import copy
import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest
import torch
from conftest import NoiseFrames
from torch import nn

from tta_deploy.eval.live import FrameRecorder, prediction_and_entropy
from tta_deploy.export.onnx_export import ExportConfig
from tta_deploy.methods.frame_path import (
    FramePath,
    normalization_constants,
    preprocess_on_device,
)
from tta_deploy.methods.method import AdaptationProgress
from tta_deploy.methods.source import Source
from tta_deploy.methods.trt import TrtTent
from tta_deploy.reference import reference_steps_before, run_reference
from tta_deploy.runtime.adapter import SPEC_SLOTS, Adapter, AdapterConfig
from tta_deploy.runtime.adapter_step import AdapterStep
from tta_deploy.runtime.trt_engine import TrtEngine
from tta_deploy.tent import (
    Tent,
    TentConfig,
    adaptable_parameter_names,
    flat_parameters,
    packed_adaptable_parameters,
)

STEPS = 50


def noise_batches(num_steps: int, B: int) -> Iterator[torch.Tensor]:
    frames = NoiseFrames(num_steps * B)
    for start in range(0, num_steps * B, B):
        yield torch.stack([frames[i][0] for i in range(start, start + B)]).cuda()


def stepped_trajectory(
    step: AdapterStep, batches: Iterator[torch.Tensor]
) -> list[torch.Tensor]:
    trajectory = []
    for images in batches:
        step.images.copy_(images)
        step()
        trajectory.append(step.packed.clone())
    return trajectory


@pytest.mark.parametrize("micro_batch, graph", [(16, True), (4, True), (16, False)])
def test_fp32_adapter_step_matches_the_reference_over_a_short_horizon(
    source_model, micro_batch, graph
):
    config = TentConfig(16, 1)
    reference_model = copy.deepcopy(source_model)
    reference = []
    run_reference(
        reference_model,
        NoiseFrames(STEPS * config.B),
        Tent(reference_model, config),
        num_workers=2,
        on_step=lambda p: reference.append(flat_parameters(p).clone()),
    )
    ours = stepped_trajectory(
        AdapterStep(source_model, config, micro_batch, bf16=False, graph=graph),
        noise_batches(STEPS, config.B),
    )
    relative = [
        ((a - b).abs().max() / b.abs().max()).item()
        for a, b in zip(ours, reference, strict=True)
    ]
    print(
        f"m={micro_batch} graph={graph}: max relative L-inf parameter difference over {STEPS} steps {max(relative):.2e}"
    )
    assert max(relative) < 1e-5


def test_bf16_graph_replays_the_eager_step(source_model):
    config = TentConfig(4, 1)
    eager = stepped_trajectory(
        AdapterStep(copy.deepcopy(source_model), config, 4, bf16=True, graph=False),
        noise_batches(10, config.B),
    )
    graph = stepped_trajectory(
        AdapterStep(source_model, config, 4, bf16=True, graph=True),
        noise_batches(10, config.B),
    )
    assert all(torch.equal(a, b) for a, b in zip(eager, graph, strict=True))


def test_revert_undoes_a_non_finite_step_exactly(source_model):
    step = AdapterStep(source_model, TentConfig(4, 1), 4, bf16=True, graph=True)
    stepped_trajectory(step, noise_batches(2, 4))
    params, momentum = step.packed.clone(), torch.cat(step.momentum).clone()
    step.images[0, 0, 0, 0] = float("nan")
    step()
    assert not step.finite
    step.revert()
    assert torch.equal(step.packed, params) and torch.equal(
        flat_parameters(step.params), params
    )
    assert torch.equal(torch.cat(step.momentum), momentum)
    step.reset_to_source()
    assert (
        torch.equal(flat_parameters(step.params), step.source)
        and not torch.cat(step.momentum).any()
    )


def uint8_frames(num_frames: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [
        torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8, generator=generator)
        for _ in range(num_frames)
    ]


@pytest.fixture
def decoupled(source_model: nn.Module, engines: dict, non_default_stream) -> Iterator:
    def build(
        config: AdapterConfig,
        capacity: int,
        poisoned_batches: frozenset[int] = frozenset(),
    ) -> tuple[FramePath, TrtTent]:
        model = copy.deepcopy(source_model)
        packed = packed_adaptable_parameters(model)
        engine = TrtEngine(
            engines[ExportConfig()],
            adaptable_parameter_names(model),
            [packed, packed.clone()],
        )
        tent = TrtTent(
            "trt_tent", Adapter(model, config, engine, poisoned_batches), capacity
        )
        tents.append(tent)
        return FramePath(tent), tent

    tents: list[TrtTent] = []
    yield build
    for tent in tents:
        # A dead adapter thread fails close() too, as a test checks.
        if tent.adapter.error is None:
            tent.close()


def engine_predictions(
    source_model: nn.Module,
    engines: dict,
    versions: list[torch.Tensor],
    frames: list[torch.Tensor],
    prediction_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    engine = TrtEngine(
        engines[ExportConfig()],
        adaptable_parameter_names(source_model),
        versions,
    )
    frame_path = FramePath(Source("trt_source", engine))
    predictions, entropies = [], []
    for frame, step in zip(frames, prediction_steps, strict=True):
        ready = torch.cuda.Event()
        ready.record()
        engine.publish(int(step), ready, int(step))
        prediction, entropy = prediction_and_entropy(frame_path(frame).numpy()[0])
        predictions.append(prediction)
        entropies.append(entropy)
    return np.array(predictions), np.array(entropies)


def test_spec_mode_steps_on_every_adaptation_frame_in_order_and_traces_the_parameters_of_each_prediction(
    source_model, engines, decoupled
):
    config = AdapterConfig(TentConfig(4, 2), "spec", micro_batch=4)
    frames = uint8_frames(96)
    frame_path, tent = decoupled(config, len(frames))
    recorder = FrameRecorder(len(frames))
    for i, frame in enumerate(frames):
        recorder(i, frame_path(frame))
    adapter = tent.adapter
    adapter.drain()
    assert adapter.steps == 12 and adapter.dropped_batches == 0 and adapter.faults == 0

    # Spec mode is the reference loop with lag: every adaptation frame
    # once, in arrival order.
    mean, std = normalization_constants()
    adaptation = [
        preprocess_on_device(frames[i].pin_memory(), mean, std)
        for i in range(0, len(frames), 2)
    ]
    offline = AdapterStep(
        copy.deepcopy(source_model), config.tent, 4, bf16=True, graph=True
    )
    trajectory = stepped_trajectory(
        offline,
        (torch.cat(adaptation[b : b + 4]) for b in range(0, len(adaptation), 4)),
    )
    assert torch.equal(adapter.engine.published_buffer, trajectory[-1])

    steps = tent.prediction_steps
    assert (np.diff(steps) >= 0).all()
    assert (
        tent.accepted_batches == reference_steps_before(len(frames), config.tent)
    ).all()
    traces = tent.traces()
    lag = traces["accepted_batches"] - traces["prediction_steps"]
    print(
        f"unthrottled spec mode: lag max {lag.max()}, mean {lag.mean():.2f} steps, blocked {adapter.blocked}"
    )
    assert (lag >= 0).all() and lag.max() > 0
    # The trace is exact: re-running every frame with the parameters it
    # names gives the same prediction.
    versions = [packed_adaptable_parameters(source_model), *trajectory]
    predictions, entropies = engine_predictions(
        source_model, engines, versions, frames, steps
    )
    assert (predictions == recorder.predictions).all()
    assert np.array_equal(entropies, recorder.entropies)


def test_deploy_mode_drops_the_oldest_batch_and_never_blocks_the_frame_path(
    decoupled,
):
    frame_path, tent = decoupled(
        AdapterConfig(TentConfig(1, 1), "deploy", micro_batch=1), 300
    )
    for frame in uint8_frames(300):
        frame_path(frame)
    adapter = tent.adapter
    adapter.drain()
    effective_stride = 300 / (adapter.steps * adapter.config.tent.B)
    print(
        f"unthrottled deploy mode: {adapter.steps} steps, {adapter.dropped_batches} dropped, effective stride {effective_stride:.2f}"
    )
    assert adapter.dropped_batches > 0 and not adapter.blocked
    assert adapter.steps + adapter.dropped_batches == adapter.completed_batches == 300
    assert frame_path.adaptation_progress() is None


def test_spec_mode_blocks_the_frame_path_once_every_slot_is_taken_and_reports_it(
    decoupled,
):
    config = AdapterConfig(TentConfig(2, 1), "spec", micro_batch=2)
    frame_path, tent = decoupled(config, 2 * SPEC_SLOTS)
    adapter = tent.adapter
    frames = uint8_frames(2 * SPEC_SLOTS)
    release = threading.Event()

    def hold_adapter() -> None:
        with adapter.held():
            release.wait()

    holder = threading.Thread(target=hold_adapter)
    holder.start()
    while not adapter.holding:
        time.sleep(0.001)
    # SPEC_SLOTS - 1 batches queue and the last slot is filling: no wait
    # yet.
    for frame in frames[: 2 * (SPEC_SLOTS - 1)]:
        frame_path(frame)
    assert frame_path.adaptation_progress() == AdaptationProgress(
        SPEC_SLOTS - 1, 0, False
    )
    start = time.perf_counter()
    threading.Timer(0.2, release.set).start()
    for frame in frames[2 * (SPEC_SLOTS - 1) :]:
        frame_path(frame)
    assert time.perf_counter() - start >= 0.2
    holder.join()
    adapter.drain()
    assert frame_path.adaptation_progress() == AdaptationProgress(
        SPEC_SLOTS, SPEC_SLOTS, True
    )


def test_deploy_mode_reverts_an_injected_nan_and_keeps_predicting(decoupled):
    frame_path, tent = decoupled(
        AdapterConfig(TentConfig(4, 1), "deploy", micro_batch=4),
        64,
        poisoned_batches=frozenset({2}),
    )
    adapter = tent.adapter
    recorder = FrameRecorder(64)
    for i, frame in enumerate(uint8_frames(64)):
        recorder(i, frame_path(frame))
        # Every batch reaches the adapter, so the poisoned one is batch
        # 2 of the stream.
        adapter.drain()
    assert (
        adapter.faults == 1
        and adapter.steps == 64 // 4 - 1
        and adapter.dropped_batches == 0
    )
    assert np.isfinite(recorder.entropies).all()
    assert adapter.engine.published_buffer.isfinite().all()


def test_disable_keeps_the_last_parameters_and_reset_returns_to_source(
    source_model, decoupled
):
    frame_path, tent = decoupled(
        AdapterConfig(TentConfig(1, 1), "deploy", micro_batch=1), 90
    )
    adapter = tent.adapter
    frames = uint8_frames(90)
    for frame in frames[:30]:
        frame_path(frame)
        adapter.drain()
    adapter.disable()
    steps = adapter.steps
    for frame in frames[30:60]:
        frame_path(frame)
    assert adapter.steps == steps == 30
    assert (tent.prediction_steps[30:60] == steps).all()
    adapter.reset_to_source()
    assert torch.equal(
        adapter.engine.published_buffer,
        packed_adaptable_parameters(source_model),
    )
    adapter.enable()
    for frame in frames[60:]:
        frame_path(frame)
        adapter.drain()
    assert adapter.steps == 60


def test_settle_after_a_warm_up_with_a_fault_matches_a_fresh_adapter(
    source_model, decoupled
):
    # Live runs warm up and then settle: counters, published γ/β, lag
    # trace and poisoned batch indices must all start over.
    config = AdapterConfig(TentConfig(4, 2), "spec", micro_batch=4)
    frames = uint8_frames(96)
    published, faults = [], []
    for warm_up in (False, True):
        frame_path, tent = decoupled(
            config, len(frames), poisoned_batches=frozenset({1})
        )
        adapter = tent.adapter
        if warm_up:
            for frame in frames[:40][::-1]:
                frame_path(frame)
            adapter.drain()
            assert adapter.faults == 1 and adapter.steps == 4
            frame_path.settle()
            assert (
                adapter.steps
                == adapter.faults
                == adapter.completed_batches
                == tent.frame_index
                == 0
            )
            assert torch.equal(
                adapter.engine.published_buffer,
                packed_adaptable_parameters(source_model),
            )
        for frame in frames:
            frame_path(frame)
        adapter.drain()
        assert tent.prediction_steps[0] == 0
        assert (
            tent.accepted_batches == reference_steps_before(len(frames), config.tent)
        ).all()
        published.append(adapter.engine.published_buffer.clone())
        faults.append(adapter.faults)
    assert faults == [1, 1] and torch.equal(published[0], published[1])


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
@pytest.mark.parametrize("mode", ["spec", "deploy"])
def test_a_failed_adapter_thread_fails_the_frame_path_instead_of_hanging_it(
    decoupled, monkeypatch, mode
):
    config = AdapterConfig(TentConfig(2, 1), mode, micro_batch=2)
    frame_path, tent = decoupled(config, 4 * SPEC_SLOTS)
    adapter = tent.adapter

    def failing_step(batch, batch_index: int) -> None:
        raise ValueError("injected adapter failure")

    monkeypatch.setattr(adapter, "step_on", failing_step)
    # Spec mode runs out of slots once the adapter stops taking batches,
    # deploy mode never does: both fail on the next completed batch.
    with pytest.raises(RuntimeError, match="adapter thread failed"):
        for frame in uint8_frames(4 * SPEC_SLOTS):
            frame_path(frame)
    assert isinstance(adapter.error, ValueError)
    with pytest.raises(RuntimeError, match="adapter thread failed"):
        tent.close()
