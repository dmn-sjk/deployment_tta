import hashlib
from collections.abc import Callable, Sized
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tta_deploy.model import NUM_CLASSES
from tta_deploy.tent import (
    Tent,
    TentConfig,
    entropy,
    flat_parameters,
    load_flat_parameters,
)

# Bounds the forward batch when no step ends a segment sooner (source,
# or B * S above this).
MAX_SEGMENT_FRAMES = 256
Precision = Literal["fp32", "bf16"]


@dataclass
class FrameRecords:
    predictions: np.ndarray
    entropies: np.ndarray
    logits_sha256: str
    params_sha256: str
    steps: int


def adaptation_frames(num_frames: int, config: TentConfig | None) -> np.ndarray:
    """Per frame: whether it is an adaptation frame.

    Source has none.
    """
    if config is None:
        return np.zeros(num_frames, dtype=bool)
    return np.array([config.is_adaptation_frame(i) for i in range(num_frames)])


def steps_completed_through(num_frames: int, config: TentConfig | None) -> np.ndarray:
    """Per frame: adaptation batches completed by it and earlier
    frames."""
    if config is None:
        return np.zeros(num_frames, dtype=np.int64)
    return np.cumsum(adaptation_frames(num_frames, config)) // config.B


def reference_steps_before(num_frames: int, config: TentConfig | None) -> np.ndarray:
    """Per frame: steps the reference has taken before it, i.e.
    adaptation batches completed by earlier frames."""
    return np.concatenate([[0], steps_completed_through(num_frames, config)[:-1]])


def fixed_lag_prediction_steps(
    num_frames: int, config: TentConfig, lag: int
) -> np.ndarray:
    return np.maximum(0, reference_steps_before(num_frames, config) - lag)


def frame_segments(num_frames: int, config: TentConfig | None) -> list[list[int]]:
    """Consecutive frame ranges that share the same adaptable
    parameters: a segment ends at the adaptation frame that completes a
    step."""
    step_ends = np.diff(steps_completed_through(num_frames, config), prepend=0) > 0
    segments, segment = [], []
    for i in range(num_frames):
        segment.append(i)
        if step_ends[i] or len(segment) == MAX_SEGMENT_FRAMES:
            segments.append(segment)
            segment = []
    if segment:
        segments.append(segment)
    return segments


def params_sha256(params: list[nn.Parameter]) -> str:
    return hashlib.sha256(flat_parameters(params).cpu().numpy().tobytes()).hexdigest()


def run_reference(
    model: nn.Module,
    dataset: Dataset,
    tent: Tent | None,
    precision: Precision = "fp32",
    num_workers: int = 8,
    # Lets tests record the parameter trajectory after every step.
    on_step: Callable[[list[nn.Parameter]], None] | None = None,
    prediction_steps: np.ndarray | None = None,
) -> FrameRecords:
    """One continual stream through this model: TENT steps with `tent`
    (built on the model), source without.

    Every frame is predicted with the adaptable parameters current at
    its arrival.

    Lag simulator: with prediction_steps, frame i is predicted with the
    adaptable parameters after prediction_steps[i] steps instead, while
    every step still starts from the newest ones, as the decoupled
    adapter does.
    """
    # torch's Dataset does not declare __len__.
    num_frames = len(cast(Sized, dataset))
    if tent is None:
        model.eval().requires_grad_(False)
        config, params = None, []
    else:
        config, params = tent.config, tent.adaptable_parameters
    segments = frame_segments(num_frames, config)
    is_adaptation_frame = adaptation_frames(num_frames, config)
    loader = DataLoader(
        dataset,
        batch_sampler=segments,
        num_workers=num_workers,
        pin_memory=True,
    )
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16")
    if prediction_steps is None:
        prediction_steps = reference_steps_before(num_frames, config)
    assert (
        prediction_steps.shape == (num_frames,)
        and (np.diff(prediction_steps) >= 0).all()
        and prediction_steps[0] == 0
    )
    # Adaptable parameters after each step that a later frame is still
    # predicted with. Source has none.
    versions = {0: flat_parameters(params).clone()} if params else {}

    predictions, entropies, logits_hash, steps = [], [], hashlib.sha256(), 0
    for segment, (images, _) in zip(segments, loader, strict=True):
        images = images.cuda(non_blocking=True)
        is_adaptation = torch.from_numpy(is_adaptation_frame[segment]).cuda()
        predicted_at = torch.from_numpy(prediction_steps[segment]).cuda()
        assert (predicted_at <= steps).all(), (
            "a frame cannot be predicted with parameters from a step that has not run yet"
        )
        is_lagged = predicted_at < steps
        logits = torch.empty(len(segment), NUM_CLASSES, device="cuda")
        with autocast:
            # Before the grad forward: the in-place swap would
            # invalidate the tensors autograd saved.
            for step in predicted_at[is_lagged].unique().tolist():
                load_flat_parameters(params, versions[step])
                with torch.no_grad():
                    logits[predicted_at == step] = model(
                        images[predicted_at == step]
                    ).float()
            if is_lagged.any():
                load_flat_parameters(params, versions[steps])
            if not (is_adaptation | is_lagged).all():
                with torch.no_grad():
                    logits[~is_adaptation & ~is_lagged] = model(
                        images[~is_adaptation & ~is_lagged]
                    ).float()
            if is_adaptation.any():
                adaptation_logits = model(images[is_adaptation]).float()
                logits[is_adaptation & ~is_lagged] = adaptation_logits.detach()[
                    ~is_lagged[is_adaptation]
                ]

        stepped = False
        if is_adaptation.any():
            assert tent is not None, "source has no adaptation frames"
            stepped = tent.adapt(adaptation_logits)
        if stepped:
            steps += 1
            if on_step is not None:
                on_step(params)
            versions[steps] = flat_parameters(params).clone()
            next_frame = segment[-1] + 1
            oldest_needed = (
                prediction_steps[next_frame] if next_frame < num_frames else steps
            )
            versions = {k: v for k, v in versions.items() if k >= oldest_needed}

        logits = logits.cpu()
        logits_hash.update(logits.numpy().tobytes())
        predictions.append(logits.argmax(1).numpy())
        entropies.append(entropy(logits).numpy())

    return FrameRecords(
        predictions=np.concatenate(predictions),
        entropies=np.concatenate(entropies),
        logits_sha256=logits_hash.hexdigest(),
        params_sha256=params_sha256(params) if params else "",
        steps=steps,
    )
