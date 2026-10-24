import json
import logging
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tta_deploy.data.imagenet_c import (
    Frame,
    decoded_frames,
    stream_labels,
    verified_stream,
)
from tta_deploy.eval.gpu import device_memory_gb, gpu_sampler
from tta_deploy.eval.metrics import HEALTH_WINDOW, per_corruption_accuracy
from tta_deploy.eval.replay import REPLAY_WARMUP, ReplayResult, replay
from tta_deploy.methods.configs import MethodConfig, build_method, run_config
from tta_deploy.methods.frame_path import FramePath
from tta_deploy.methods.method import Method
from tta_deploy.model import NUM_CLASSES, set_fps_precision

log = logging.getLogger(__name__)
HEALTH_LOG_SECONDS = 1.0


def consume_prediction(frame_index: int, logits: torch.Tensor) -> None:
    """What a live run does with each frame's logits, without keeping
    them: bisection replays time the same work as the live run."""
    prediction_and_entropy(logits.numpy()[0])


def prediction_and_entropy(logits: np.ndarray) -> tuple[int, float]:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    total = exp.sum()
    return int(shifted.argmax()), float(np.log(total) - (exp * shifted).sum() / total)


class FrameRecorder:
    """Every frame's prediction and entropy, in stream order."""

    def __init__(self, capacity: int) -> None:
        self.predictions = np.zeros(capacity, dtype=np.int64)
        self.entropies = np.zeros(capacity)
        self.frames = 0

    def __call__(self, frame_index: int, logits: torch.Tensor) -> None:
        assert frame_index == self.frames, "frames are recorded in stream order"
        self.predictions[frame_index], self.entropies[frame_index] = (
            prediction_and_entropy(logits.numpy()[0])
        )
        # Last: the health thread reads only frames already written.
        self.frames += 1


def health(recorder: FrameRecorder, method: Method) -> dict:
    """Rolling collapse indicators over the last HEALTH_WINDOW frames,
    plus the method's counters."""
    n = recorder.frames
    window = slice(max(0, n - HEALTH_WINDOW), n)
    histogram = np.bincount(recorder.predictions[window], minlength=NUM_CLASSES) / max(
        1, window.stop - window.start
    )
    histogram = histogram[histogram > 0]
    return {
        "frames": n,
        "prediction_entropy": float(recorder.entropies[window].mean()) if n else 0.0,
        "class_histogram_entropy": float(-(histogram * np.log(histogram)).sum()),
        **method.summary(),
    }


@contextmanager
def health_logging(recorder: FrameRecorder, method: Method) -> Iterator[None]:
    """Logs health once per HEALTH_LOG_SECONDS from its own thread, so
    the frame thread never formats or writes logs."""
    stopped = threading.Event()
    errors: list[Exception] = []

    def run() -> None:
        try:
            while not stopped.wait(HEALTH_LOG_SECONDS):
                log.info("health %s", health(recorder, method))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run, name="health", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
    # A thread's exception never reaches the caller on its own.
    if errors:
        raise errors[0]


def configure_logging(path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, mode="w"), logging.StreamHandler()],
        force=True,
    )


def method_summary(method: Method, replayed_seconds: float) -> dict:
    """The method's counters, plus steps per second of replayed
    stream."""
    summary: dict = method.summary()
    if (steps := summary.get("steps")) is not None:
        summary["steps_per_s"] = steps / replayed_seconds
    return summary


def lag_summary(traces: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Lag statistics over a run, for methods with a lag trace."""
    if "prediction_steps" not in traces:
        return {}
    lag = traces["accepted_batches"] - traces["prediction_steps"]
    return {
        "lag_mean": float(lag.mean()),
        "lag_p99": float(np.percentile(lag, 99)),
        "lag_max": int(lag.max()),
    }


@dataclass(frozen=True)
class LiveRun:
    result: ReplayResult
    recorder: FrameRecorder
    # The method's counters at the last answered frame.
    summary: dict
    gpu: dict[str, float]


def live_run(method: Method, frames: Sequence[torch.Tensor], fps: float) -> LiveRun:
    """The whole stream through a method at a fixed input rate, health
    logged once per second.

    Closes the method. A failed run leaves it to the worker's exit,
    which ends the daemon threads.
    """
    frame_path = FramePath(method)
    # The replay settles back to source, so warm-up only warms kernels
    # and the allocator.
    for frame in frames[:REPLAY_WARMUP]:
        frame_path(frame)
    recorder = FrameRecorder(len(frames))
    with health_logging(recorder, method):
        with gpu_sampler() as gpu:
            result = replay(frame_path, frames, fps, warmup=0, on_frame=recorder)
        summary = method_summary(method, len(frames) / fps) | lag_summary(
            method.traces()
        )
        # Drains the adapter before the health thread stops, so its log
        # covers the last steps.
        method.close()
    assert recorder.frames == len(frames), (
        f"recorded {recorder.frames} of {len(frames)} frames"
    )
    return LiveRun(result, recorder, summary, gpu)


def save_live_run(
    out: Path,
    stream: list[Frame],
    config: MethodConfig,
    method: Method,
    run: LiveRun,
    **metrics: object,
) -> dict:
    """Per-frame records, the method's traces and the run's metrics."""
    labels, corruptions = stream_labels(stream)
    recorder, result = run.recorder, run.result
    accuracy = per_corruption_accuracy(recorder.predictions, labels, corruptions)
    summary = {
        "method": method.name,
        "config": asdict(run_config(config)),
        "frames": len(stream),
        **metrics,
        "accuracy": accuracy,
        **result.summary(),
        **run.summary,
        **run.gpu,
        **device_memory_gb(),
        "entropies_finite": bool(np.isfinite(recorder.entropies).all()),
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "frames.npz",
        predictions=recorder.predictions,
        entropies=recorder.entropies,
        latency_ms=result.latency_ms,
        labels=labels,
        corruptions=corruptions,
        **method.traces(),
    )
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(
        f"{method.name} at {result.fps:.1f} fps: average {accuracy['average']:.2f}%, p99 {result.p99_ms:.2f} ms (interval {result.interval_ms:.2f}), sustainable {result.sustainable}"
    )
    return summary


def run_live(
    config: MethodConfig, subset: str, seed: int, fps: float, out: Path
) -> tuple[Method, dict]:
    """Worker body: the whole stream of a subset through a freshly built
    method at a fixed input rate, saved to `out`.

    Returns the closed method, for checks on its final state.
    """
    out.mkdir(parents=True, exist_ok=True)
    configure_logging(out / "health.log")
    set_fps_precision()
    stream = verified_stream(subset, seed)
    frames = decoded_frames(stream)
    method = build_method(config, len(stream))
    run = live_run(method, frames, fps)
    return method, save_live_run(out, stream, config, method, run, seed=seed)
