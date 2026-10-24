import itertools
import time

import numpy as np
import pytest
import torch

from tta_deploy.eval.replay import (
    ReplayResult,
    keeps_up,
    max_sustainable_fps,
    replay,
    step_down_until_held,
)
from tta_deploy.methods.method import AdaptationProgress

SERVICE_SECONDS = 0.004


class FixedServiceFramePath:
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        end = time.perf_counter() + SERVICE_SECONDS
        while time.perf_counter() < end:
            pass
        return frame

    def event_span_ms(self) -> float:
        return 0.0

    def settle(self) -> None:
        pass

    def adaptation_progress(self) -> AdaptationProgress | None:
        return None


FRAMES = [torch.zeros(1)] * 300


def test_rate_below_service_rate_is_sustainable():
    result = replay(
        FixedServiceFramePath(), FRAMES, fps=0.8 / SERVICE_SECONDS, warmup=10
    )
    assert result.sustainable
    assert result.p99_ms == pytest.approx(SERVICE_SECONDS * 1e3, rel=0.25)


def test_rate_above_service_rate_builds_a_backlog():
    result = replay(
        FixedServiceFramePath(), FRAMES, fps=1.2 / SERVICE_SECONDS, warmup=10
    )
    assert not result.sustainable
    # Each frame waits for the ones before it, so latency grows along
    # the replay.
    assert result.latency_ms[-1] > 10 * result.latency_ms[0]


def test_max_sustainable_fps_finds_the_service_rate():
    found = max_sustainable_fps(FixedServiceFramePath(), FRAMES, warmup=10)
    assert (
        found.result.sustainable and found.result.fps == found.fps <= found.bisected_fps
    )
    assert found.fps == pytest.approx(1 / SERVICE_SECONDS, rel=0.1)


class StallingFramePath(FixedServiceFramePath):
    def __init__(self):
        self.calls = 0

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        # Every 10th frame stalls for 50 ms, so p99 exceeds the interval
        # at every rate above 20 fps.
        self.calls += 1
        if self.calls % 10 == 0:
            time.sleep(0.05)
        return super().__call__(frame)


def test_max_sustainable_fps_stops_at_the_floor(monkeypatch):
    monkeypatch.setattr("tta_deploy.eval.replay.MIN_FPS", 50.0)
    with pytest.raises(AssertionError, match="not sustainable"):
        max_sustainable_fps(StallingFramePath(), FRAMES[:100], warmup=0)


def test_max_sustainable_fps_steps_down_until_repeat_replays_pass(monkeypatch):
    # Between 95 and 100 fps every other replay fails, like a rate at
    # the edge of sustainability.
    flaky_passes = itertools.cycle([True, False])

    def edge_replay(frame_path, frames, fps, warmup, on_frame):
        passes = fps <= 95 or (fps <= 100 and next(flaky_passes))
        latency_ms = 1e3 / fps * (0.5 if passes else 2)
        return ReplayResult(
            fps, np.full(10, latency_ms), np.full(10, 6.0), np.zeros(10), None
        )

    monkeypatch.setattr("tta_deploy.eval.replay.replay", edge_replay)
    found = max_sustainable_fps(FixedServiceFramePath(), FRAMES, warmup=0)
    assert found.result.sustainable and found.result.fps == found.fps
    assert 95 * 0.98 < found.fps <= 95 < found.bisected_fps
    assert found.fps == pytest.approx(
        found.bisected_fps * 0.98**found.confirm_step_downs
    )


@pytest.mark.parametrize(
    "arrived, stepped, blocked, expected",
    [
        (100, 97, False, True),
        (100, 96, False, False),
        (1, 0, False, True),
        (2, 0, False, False),
        (100, 100, True, False),
        (0, 0, False, True),
    ],
)
def test_the_adapter_keeps_up_within_2_percent_plus_the_batch_in_flight_and_never_when_the_frame_path_blocked(
    arrived, stepped, blocked, expected
):
    assert keeps_up(AdaptationProgress(arrived, stepped, blocked)) == expected


def test_a_replay_within_the_latency_bound_is_not_sustainable_when_the_adapter_falls_behind():
    def result(adaptation):
        return ReplayResult(
            100.0, np.full(10, 5.0), np.full(10, 5.0), np.zeros(10), adaptation
        )

    assert (
        result(None).sustainable
        and result(AdaptationProgress(100, 98, False)).sustainable
    )
    assert not result(AdaptationProgress(100, 90, False)).sustainable
    assert not result(AdaptationProgress(100, 100, True)).sustainable


def test_step_down_multiplies_by_0_98_until_the_rate_holds():
    tried = []
    fps, step_downs = step_down_until_held(
        100.0, lambda fps: tried.append(fps) or fps < 95
    )
    assert step_downs == 3 and fps == pytest.approx(100 * 0.98**3)
    assert tried == pytest.approx([100 * 0.98**k for k in range(4)])


def test_step_down_stops_loudly_at_the_cap():
    with pytest.raises(AssertionError, match="after 2 step-downs"):
        step_down_until_held(100.0, lambda fps: False, max_step_downs=2)


class TrackedFramePath(FixedServiceFramePath):
    def __init__(self, progress: AdaptationProgress):
        self.progress, self.frames = progress, 0
        self.calls: list[tuple[str, int]] = []

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        self.frames += 1
        return super().__call__(frame)

    def settle(self) -> None:
        self.calls.append(("settle", self.frames))

    def adaptation_progress(self) -> AdaptationProgress:
        self.calls.append(("progress", self.frames))
        return self.progress


@pytest.mark.parametrize(
    "progress, sustainable",
    [
        (AdaptationProgress(10, 10, False), True),
        (AdaptationProgress(10, 10, True), False),
        (AdaptationProgress(10, 5, False), False),
    ],
)
def test_replay_settles_after_warmup_and_reads_adaptation_progress_after_the_last_frame(
    progress, sustainable
):
    frame_path = TrackedFramePath(progress)
    result = replay(frame_path, FRAMES[:50], fps=0.5 / SERVICE_SECONDS, warmup=10)
    # Warmup blocking must not count, and stepped counts what finished
    # by the time the last frame is answered.
    assert frame_path.calls == [("settle", 10), ("progress", 60)]
    assert result.adaptation == progress and result.sustainable == sustainable


def test_on_frame_sees_every_replayed_frame_in_order_inside_its_latency():
    seen = []

    def on_frame(i: int, logits: torch.Tensor) -> None:
        seen.append(i)
        end = time.perf_counter() + SERVICE_SECONDS
        while time.perf_counter() < end:
            pass

    result = replay(
        FixedServiceFramePath(),
        FRAMES[:20],
        fps=0.25 / SERVICE_SECONDS,
        warmup=5,
        on_frame=on_frame,
    )
    assert seen == list(range(20))
    # Consuming the prediction is part of the frame's service time.
    assert result.service_ms.min() >= 2 * SERVICE_SECONDS * 1e3
