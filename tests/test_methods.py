import copy
import json

import numpy as np
import pytest
import torch
from conftest import Uint8NoiseFrames

from tta_deploy.eval.live import FrameRecorder
from tta_deploy.export import engine as engine_module
from tta_deploy.export.engine import PARITY_ATOL, built_engine, write_engine_record
from tta_deploy.export.onnx_export import ExportConfig
from tta_deploy.methods.configs import (
    PytorchSourceConfig,
    PytorchTentConfig,
    TrtSourceConfig,
    TrtTentConfig,
    build_method,
)
from tta_deploy.methods.frame_path import FramePath
from tta_deploy.methods.method import Method
from tta_deploy.methods.source import Source
from tta_deploy.reference import run_reference
from tta_deploy.runtime.adapter import AdapterConfig
from tta_deploy.tent import TentConfig

pytestmark = pytest.mark.usefixtures("non_default_stream")
NUM_FRAMES = 32


@pytest.fixture
def built_engines(engines: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The test engines, with records, where build_method looks."""
    directory = engines[ExportConfig()].parent
    monkeypatch.setattr(engine_module, "ENGINE_DIR", directory)
    for export in engines:
        write_engine_record(export, parity=0.0)


METHODS = [
    PytorchSourceConfig(),
    PytorchTentConfig(TentConfig(4, 2)),
    TrtSourceConfig(),
    TrtSourceConfig(ExportConfig(layernorm_inputs=False)),
    TrtTentConfig(AdapterConfig(TentConfig(4, 2), "spec", micro_batch=4)),
    TrtTentConfig(AdapterConfig(TentConfig(4, 2), "deploy", micro_batch=4)),
]


def recorded_pass(frame_path: FramePath) -> FrameRecorder:
    recorder = FrameRecorder(NUM_FRAMES)
    frames = Uint8NoiseFrames(NUM_FRAMES, preprocess=False)
    for i in range(NUM_FRAMES):
        recorder(i, frame_path(frames[i][0]))
    return recorder


@pytest.mark.parametrize("config", METHODS, ids=lambda c: c.name)
@pytest.mark.usefixtures("built_engines")
def test_every_method_runs_through_the_same_frame_path(config):
    method = build_method(config, NUM_FRAMES)
    try:
        # Checks attribute names only: mypy checks the signatures.
        assert isinstance(method, Method) and method.name == config.name
        frame_path = FramePath(method)
        recorder = recorded_pass(frame_path)
        assert np.isfinite(recorder.entropies).all()
        assert frame_path.event_span_ms() > 0
    finally:
        method.close()
    if isinstance(config, PytorchTentConfig | TrtTentConfig):
        assert method.summary()["steps"] > 0


@pytest.mark.usefixtures("built_engines")
def test_trt_tent_traces_every_frame_and_counts_every_step_after_close():
    config = METHODS[4]
    method = build_method(config, NUM_FRAMES)
    recorded_pass(FramePath(method))
    method.close()
    summary, traces = method.summary(), method.traces()
    # 16 adaptation frames at S = 2, 4 batches of B = 4, all stepped in
    # spec mode once closed.
    assert summary["steps"] == summary["completed_batches"] == 4
    assert summary["dropped_batches"] == summary["faults"] == 0
    assert summary["effective_stride"] == NUM_FRAMES / (4 * 4)
    assert (
        len(traces["prediction_steps"]) == len(traces["accepted_batches"]) == NUM_FRAMES
    )
    assert len(traces["step_lag"]) == len(traces["step_published_s"]) == 4
    assert (traces["accepted_batches"] >= traces["prediction_steps"]).all()


def test_an_engine_that_does_not_match_its_record_is_refused(built_engines, engines):
    path = engines[ExportConfig()]
    record = json.loads(path.with_suffix(".json").read_text())
    path.with_suffix(".json").write_text(
        json.dumps(record | {"engine_sha256": "0" * 64})
    )
    with pytest.raises(AssertionError, match="does not match its record"):
        built_engine(ExportConfig())
    write_engine_record(ExportConfig(), parity=PARITY_ATOL)
    with pytest.raises(AssertionError, match="does not match its record"):
        built_engine(ExportConfig())


def test_pytorch_source_predicts_like_the_reference_source(source_model):
    reference = run_reference(
        copy.deepcopy(source_model),
        Uint8NoiseFrames(NUM_FRAMES, preprocess=True),
        None,
        num_workers=2,
    )
    recorder = recorded_pass(
        FramePath(Source("pytorch_source", source_model.requires_grad_(False)))
    )
    # Batch-1 vs batch kernels and GPU vs CPU preprocessing round
    # differently.
    assert (recorder.predictions == reference.predictions).all()
    assert np.abs(recorder.entropies - reference.entropies).max() < 1e-3


def test_method_names_encode_every_config_field():
    tent = TentConfig(16, 1)
    adapter = AdapterConfig(tent, "spec", micro_batch=16)
    names = [
        PytorchSourceConfig().name,
        PytorchTentConfig(tent).name,
        TrtSourceConfig().name,
        TrtSourceConfig(ExportConfig(layernorm_fp32=False)).name,
        TrtTentConfig(adapter).name,
        TrtTentConfig(AdapterConfig(tent, "spec", 16, graph=False)).name,
        TrtTentConfig(AdapterConfig(tent, "deploy", 16)).name,
        TrtTentConfig(adapter, poisoned_batches=frozenset({1})).name,
        TrtTentConfig(adapter, ExportConfig(softmax_fp32=False)).name,
    ]
    assert len(set(names)) == len(names)
    assert names[:3] == ["pytorch_source", "pytorch_tent_B16_S1", "trt_source"]


def test_trt_tent_needs_layernorm_inputs():
    with pytest.raises(AssertionError, match="γ/β as inputs"):
        TrtTentConfig(
            AdapterConfig(TentConfig(4, 1), "spec", 4),
            ExportConfig(layernorm_inputs=False),
        )


def test_frame_recorder_rejects_out_of_order_frames():
    recorder = FrameRecorder(2)
    with pytest.raises(AssertionError, match="stream order"):
        recorder(1, torch.zeros(1, 1000))
