import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import torch

from tta_deploy.methods.method import AdaptationProgress

# ADR 0002: bisection and repeat replays run on the first REPLAY_FRAMES
# frames of the dev stream, each after REPLAY_WARMUP warm-up frames.
REPLAY_FRAMES = 2000
REPLAY_WARMUP = 200
# Max fps resolution of the bisection.
FPS_RESOLUTION = 0.01
# Below this the frame path is broken, not slow: stop instead of
# replaying for hours.
MIN_FPS = 1.0
# ADR 0002: the bisected rate must hold on this many repeat replays in a
# row, stepping down by CONFIRM_STEP until it does.
CONFIRM_REPLAYS = 2
CONFIRM_STEP = 0.98
# ADR 0002: a whole-stream live run that still fails after this many
# step-downs is not a rate problem, so stop and look.
LIVE_MAX_STEP_DOWNS = 5
# ADR 0002: in spec mode the adapter keeps up iff
# stepped >= KEEPS_UP_FRACTION * arrived - 1. The -1 is the batch
# normally in flight.
KEEPS_UP_FRACTION = 0.98


def keeps_up(progress: AdaptationProgress) -> bool:
    return (
        not progress.blocked
        and progress.stepped_batches >= KEEPS_UP_FRACTION * progress.arrived_batches - 1
    )


class ReplayedFramePath(Protocol):
    def __call__(self, frame: torch.Tensor) -> torch.Tensor: ...

    def event_span_ms(self) -> float: ...

    def settle(self) -> None: ...

    def adaptation_progress(self) -> AdaptationProgress | None: ...


@dataclass
class ReplayResult:
    fps: float
    latency_ms: np.ndarray
    service_ms: np.ndarray
    event_span_ms: np.ndarray
    # None when no adapter must keep up: inference only, sync TENT,
    # deploy mode.
    adaptation: AdaptationProgress | None

    @property
    def interval_ms(self) -> float:
        return 1e3 / self.fps

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.latency_ms, 50))

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.latency_ms, 99))

    @property
    def p999_ms(self) -> float:
        return float(np.percentile(self.latency_ms, 99.9))

    @property
    def sustainable(self) -> bool:
        """ADR 0002: p99 end-to-end latency within one frame interval,
        and in spec mode the adapter keeps up."""
        return self.p99_ms <= self.interval_ms and (
            self.adaptation is None or keeps_up(self.adaptation)
        )

    def summary(self) -> dict[str, float | bool | None]:
        adaptation = self.adaptation
        progress = (
            asdict(adaptation) | {"keeps_up": keeps_up(adaptation)}
            if adaptation
            else dict.fromkeys(
                ("arrived_batches", "stepped_batches", "blocked", "keeps_up")
            )
        )
        return {
            "fps": self.fps,
            "interval_ms": self.interval_ms,
            "sustainable": self.sustainable,
            "latency_p50_ms": self.p50_ms,
            "latency_p99_ms": self.p99_ms,
            "latency_p999_ms": self.p999_ms,
            "service_p50_ms": float(np.median(self.service_ms)),
            "service_p99_ms": float(np.percentile(self.service_ms, 99)),
            "event_span_p50_ms": float(np.median(self.event_span_ms)),
            **progress,
        }


@dataclass
class MaxFps:
    bisected_fps: float
    fps: float
    confirm_step_downs: int
    result: ReplayResult

    def summary(self) -> dict[str, float | bool | None]:
        return self.result.summary() | {
            "bisected_fps": self.bisected_fps,
            "max_fps": self.fps,
            "confirm_step_downs": self.confirm_step_downs,
        }


def wait_until(deadline: float) -> None:
    # Never sleep: an idle core slows the next frame's eager launches
    # (ADR 0002). sleep(0) only hands over the GIL.
    while time.perf_counter() < deadline:
        time.sleep(0)


def replay(
    frame_path: ReplayedFramePath,
    frames: Sequence[torch.Tensor],
    fps: float,
    warmup: int,
    on_frame: Callable[[int, torch.Tensor], None] | None = None,
) -> ReplayResult:
    """Frame i arrives at start + i / fps, after warm-up and a settle.

    Latency runs from arrival to prediction on host, so waiting behind a
    late frame counts. `on_frame(i, logits)` sees every replayed frame,
    never a warm-up one: consuming the prediction is part of the
    frame's latency.
    """
    for i in range(warmup):
        frame_path(frames[i % len(frames)])
    frame_path.settle()
    latency, service, span = (np.empty(len(frames)) for _ in range(3))
    start = time.perf_counter() + 1 / fps
    for i, frame in enumerate(frames):
        arrival = start + i / fps
        wait_until(arrival)
        begin = time.perf_counter()
        logits = frame_path(frame)
        if on_frame is not None:
            on_frame(i, logits)
        done = time.perf_counter()
        latency[i], service[i], span[i] = (
            (done - arrival) * 1e3,
            (done - begin) * 1e3,
            frame_path.event_span_ms(),
        )
    # Read before anything waits for the adapter: stepped counts the
    # steps finished by the time the last frame is answered.
    return ReplayResult(fps, latency, service, span, frame_path.adaptation_progress())


def step_down_until_held(
    fps: float,
    holds: Callable[[float], bool],
    max_step_downs: int | None = None,
) -> tuple[float, int]:
    """ADR 0002: multiply the rate by CONFIRM_STEP until it holds.

    Returns that rate and the number of step-downs.
    """
    step_downs = 0
    while not holds(fps):
        assert max_step_downs is None or step_downs < max_step_downs, (
            f"still not sustainable after {step_downs} step-downs, at {fps:.1f} fps"
        )
        fps *= CONFIRM_STEP
        step_downs += 1
        assert fps >= MIN_FPS, f"not sustainable at any rate down to {MIN_FPS} fps"
    return fps, step_downs


@dataclass
class LiveStepDown:
    max_fps: float
    live_step_downs: int
    # Rate, latency and adaptation progress of every live run tried.
    attempts: list[dict]


def live_step_down(
    confirmed_fps: float, live_run: Callable[[float], dict]
) -> LiveStepDown:
    """ADR 0002: whole-stream live runs (`live_run`) from the confirmed
    rate down by CONFIRM_STEP until one is sustainable."""
    attempts = []

    def holds(fps: float) -> bool:
        live = live_run(fps)
        assert live["fps"] == fps, (
            f"live run metrics are from a run at {live['fps']} fps, not {fps}"
        )
        attempts.append(
            {
                k: live[k]
                for k in (
                    "fps",
                    "interval_ms",
                    "latency_p99_ms",
                    "arrived_batches",
                    "stepped_batches",
                    "blocked",
                    "sustainable",
                )
            }
        )
        return live["sustainable"]

    fps, step_downs = step_down_until_held(confirmed_fps, holds, LIVE_MAX_STEP_DOWNS)
    return LiveStepDown(fps, step_downs, attempts)


def max_sustainable_fps(
    frame_path: ReplayedFramePath,
    frames: Sequence[torch.Tensor],
    warmup: int,
    on_frame: Callable[[int, torch.Tensor], None] | None = None,
) -> MaxFps:
    """Bisect the highest sustainable replay rate, then step down until
    it holds on repeat replays (ADR 0002).

    `on_frame` as in `replay`, so the bisection times the same work as
    the live run.
    """
    unthrottled = replay(frame_path, frames, 1e9, warmup, on_frame)
    high = 1.2e3 / float(np.median(unthrottled.service_ms))
    while replay(frame_path, frames, high, warmup, on_frame).sustainable:
        high *= 2
    low = high / 2
    best = replay(frame_path, frames, low, warmup, on_frame)
    while not best.sustainable:
        low /= 2
        assert low >= MIN_FPS, f"not sustainable at any rate down to {MIN_FPS} fps"
        best = replay(frame_path, frames, low, warmup, on_frame)
    while high / low > 1 + FPS_RESOLUTION:
        middle = (low * high) ** 0.5
        result = replay(frame_path, frames, middle, warmup, on_frame)
        if result.sustainable:
            low = middle
        else:
            high = middle
    result = best

    def holds_on_repeat_replays(fps: float) -> bool:
        nonlocal result
        for _ in range(CONFIRM_REPLAYS):
            result = replay(frame_path, frames, fps, warmup, on_frame)
            if not result.sustainable:
                return False
        return True

    fps, step_downs = step_down_until_held(low, holds_on_repeat_replays)
    return MaxFps(low, fps, step_downs, result)
