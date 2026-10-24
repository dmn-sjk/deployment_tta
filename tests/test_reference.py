import copy
import hashlib

import pytest
import torch
from conftest import NoiseFrames
from torch import nn

from tta_deploy.model import IMAGENET_MEAN, IMAGENET_STD, WEIGHTS
from tta_deploy.reference import (
    FrameRecords,
    frame_segments,
    run_reference,
)
from tta_deploy.tent import (
    ADAPTABLE_VALUES,
    Tent,
    TentConfig,
    configure_tent,
    entropy,
    flat_parameters,
    tent_optimizer,
)

STEPS = 50


def recording_run(
    model: nn.Module, num_frames: int, config: TentConfig
) -> tuple[FrameRecords, list[torch.Tensor]]:
    trajectory = []
    records = run_reference(
        model,
        NoiseFrames(num_frames),
        Tent(model, config),
        num_workers=2,
        on_step=lambda p: trajectory.append(flat_parameters(p)),
    )
    return records, trajectory


def roid_tent(
    model: nn.Module, num_frames: int, B: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # Transcription of ROID's methods/tent.py forward_and_adapt.
    params = configure_tent(model)
    optimizer = tent_optimizer(params, TentConfig(B, 1))
    frames = NoiseFrames(num_frames)
    logits, trajectory = [], []
    for start in range(0, num_frames, B):
        x = torch.stack([frames[i][0] for i in range(start, start + B)]).cuda()
        outputs = model(x)
        loss = entropy(outputs).mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        logits.append(outputs.detach())
        trajectory.append(flat_parameters(params))
    return torch.cat(logits), trajectory


def per_frame_tent(
    model: nn.Module, num_frames: int, config: TentConfig
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # Independent oracle for the stride semantics: one frame at a time,
    # gradient accumulation over B adaptation frames.
    params = configure_tent(model)
    optimizer = tent_optimizer(params, config)
    frames = NoiseFrames(num_frames)
    logits, trajectory, accumulated = [], [], 0
    for i in range(num_frames):
        x = frames[i][0][None].cuda()
        if i % config.S == 0:
            out = model(x)
            (entropy(out).sum() / config.B).backward()
            accumulated += 1
            if accumulated == config.B:
                optimizer.step()
                optimizer.zero_grad()
                accumulated = 0
                trajectory.append(flat_parameters(params))
        else:
            with torch.no_grad():
                out = model(x)
        logits.append(out.detach())
    return torch.cat(logits), trajectory


def test_preprocessing_matches_checkpoint_transforms():
    transforms = WEIGHTS.transforms()
    assert tuple(transforms.mean) == IMAGENET_MEAN
    assert tuple(transforms.std) == IMAGENET_STD


def test_adaptable_parameters_are_all_layernorm_affine(source_model):
    params = configure_tent(source_model)
    names = {n for n, p in source_model.named_parameters() if p.requires_grad}
    expected = {
        f"encoder.layers.encoder_layer_{i}.{ln}.{w}"
        for i in range(12)
        for ln in ("ln_1", "ln_2")
        for w in ("weight", "bias")
    }
    expected |= {"encoder.ln.weight", "encoder.ln.bias"}
    assert names == expected
    assert sum(p.numel() for p in params) == ADAPTABLE_VALUES == 38_400


@pytest.mark.parametrize("B, lr", [(1, 2.5e-4 / 64), (64, 2.5e-4)])
def test_lr_scales_linearly_with_B(source_model, B, lr):
    optimizer = tent_optimizer(configure_tent(source_model), TentConfig(B, 1))
    group = optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(lr, rel=1e-12)
    assert (
        group["momentum"],
        group["nesterov"],
        group["dampening"],
        group["weight_decay"],
    ) == (0.9, True, 0.0, 0.0)


def test_train_and_eval_forward_are_identical(source_model):
    x = NoiseFrames(4)[0][0][None].repeat(4, 1, 1, 1).cuda()
    eval_logits = source_model.eval()(x)
    train_logits = source_model.train()(x)
    assert torch.equal(eval_logits, train_logits)


@pytest.mark.parametrize("B, S", [(1, 1), (4, 1), (4, 3), (16, 8)])
def test_segments_end_at_step_triggers(source_model, B, S):
    num_frames = 1000
    segments = frame_segments(num_frames, TentConfig(B, S))
    assert sum(segments, []) == list(range(num_frames))
    adaptation_counts = [sum(i % S == 0 for i in s) for s in segments]
    full_batches = len(range(0, num_frames, S)) // B
    assert adaptation_counts[:full_batches] == [B] * full_batches
    assert all(s[-1] % S == 0 for s in segments[:full_batches])


def test_short_horizon_bit_identical_to_roid_tent(source_model):
    B = 16
    records, trajectory = recording_run(
        copy.deepcopy(source_model), STEPS * B, TentConfig(B, 1)
    )
    roid_logits, roid_trajectory = roid_tent(source_model, STEPS * B, B)
    assert records.steps == STEPS
    assert all(
        torch.equal(a, b) for a, b in zip(trajectory, roid_trajectory, strict=True)
    )
    assert (records.predictions == roid_logits.argmax(1).cpu().numpy()).all()


@pytest.mark.parametrize("B, S", [(1, 2), (4, 1), (4, 3)])
def test_short_horizon_equivalence_with_per_frame_oracle(source_model, B, S):
    num_frames = STEPS * B * S
    records, trajectory = recording_run(
        copy.deepcopy(source_model), num_frames, TentConfig(B, S)
    )
    oracle_logits, oracle_trajectory = per_frame_tent(
        source_model, num_frames, TentConfig(B, S)
    )
    assert records.steps == len(oracle_trajectory) == STEPS
    relative = [
        ((ours - oracle).abs().max() / oracle.abs().max()).item()
        for ours, oracle in zip(trajectory, oracle_trajectory, strict=True)
    ]
    print(
        f"B={B} S={S}: max relative L-inf parameter difference over {STEPS} steps {max(relative):.2e}"
    )
    assert max(relative) < 1e-5
    assert (records.predictions == oracle_logits.argmax(1).cpu().numpy()).mean() > 0.99
    oracle_entropy = entropy(oracle_logits).cpu().numpy()
    assert abs(records.entropies - oracle_entropy).max() < 1e-3


@pytest.mark.parametrize("config", [None, TentConfig(4, 2)])
def test_reruns_are_bit_identical(source_model, config):
    first_model = copy.deepcopy(source_model)
    first = run_reference(
        first_model,
        NoiseFrames(160),
        Tent(first_model, config) if config else None,
        num_workers=2,
    )
    second = run_reference(
        source_model,
        NoiseFrames(160),
        Tent(source_model, config) if config else None,
        num_workers=2,
    )
    assert (first.logits_sha256, first.params_sha256) == (
        second.logits_sha256,
        second.params_sha256,
    )


def test_source_never_adapts_and_predicts_with_the_plain_model(source_model):
    records = run_reference(source_model, NoiseFrames(8), None, num_workers=2)
    with torch.no_grad():
        logits = source_model(
            torch.stack([NoiseFrames(8)[i][0] for i in range(8)]).cuda()
        ).cpu()
    assert records.steps == 0
    assert records.logits_sha256 == hashlib.sha256(logits.numpy().tobytes()).hexdigest()


def test_adapt_rejects_logits_that_cross_a_step(source_model):
    tent = Tent(source_model, TentConfig(4, 1))
    x = torch.stack([NoiseFrames(3)[i][0] for i in range(3)]).cuda()
    assert not tent.adapt(source_model(x))
    with pytest.raises(AssertionError, match="crosses a step"):
        tent.adapt(source_model(x[:2]))
