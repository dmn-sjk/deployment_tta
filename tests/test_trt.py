import time
from pathlib import Path

import onnx
import pytest
import torch
from conftest import ready
from polygraphy.backend.trt import TrtRunner
from polygraphy.comparator import Comparator, CompareFunc
from torch import nn

from tta_deploy.data.imagenet_c import FrameDataset, decode
from tta_deploy.export.engine import PARITY_ATOL
from tta_deploy.export.onnx_export import (
    IMAGE_INPUT,
    LOGITS_OUTPUT,
    ExportConfig,
)
from tta_deploy.methods.frame_path import (
    FramePath,
    normalization_constants,
    preprocess_on_device,
)
from tta_deploy.methods.source import Source
from tta_deploy.runtime.trt_engine import (
    TrtEngine,
    engine_logits,
    load_cuda_engine,
)
from tta_deploy.tent import (
    adaptable_parameter_names,
    configure_tent,
    packed_adaptable_parameters,
)

pytestmark = pytest.mark.usefixtures("non_default_stream")


def source_engine(path: Path, model: nn.Module) -> TrtEngine:
    return TrtEngine(
        path,
        adaptable_parameter_names(model),
        [packed_adaptable_parameters(model)],
    )


def test_gpu_preprocess_matches_cpu_preprocess(dev_stream_sample):
    cpu = FrameDataset(dev_stream_sample)
    decoded = FrameDataset(dev_stream_sample, transform=decode)
    mean, std = normalization_constants()
    for i in range(len(dev_stream_sample)):
        frame, label = decoded[i]
        assert (
            frame.dtype == torch.uint8
            and frame.shape == (224, 224, 3)
            and label == cpu[i][1]
        )
        on_device = preprocess_on_device(frame.pin_memory(), mean, std)
        torch.testing.assert_close(on_device.cpu()[0], cpu[i][0], rtol=0, atol=1e-6)


def test_engine_matches_pytorch_fp32_with_polygraphy(
    source_model, engines, dev_stream_sample
):
    names = adaptable_parameter_names(source_model)
    state = dict(source_model.named_parameters())
    images = torch.stack([image for image, _ in FrameDataset(dev_stream_sample)])
    feeds = [
        {
            IMAGE_INPUT: images[i : i + 1].numpy(),
            **{n: state[n].detach().cpu().numpy() for n in names},
        }
        for i in range(len(images))
    ]
    trt_engine = load_cuda_engine(engines[ExportConfig()])
    results = Comparator.run(
        [TrtRunner(trt_engine, name="trt_fp16_ln32")], data_loader=feeds
    )
    # Polygraphy 0.49's PytRunner is broken (it passes Polygraphy dtypes
    # to NumPy), so the PyTorch outputs are added directly.
    with torch.no_grad():
        pytorch_logits = source_model(images.cuda()).cpu().numpy()
    results.add(
        [{LOGITS_OUTPUT: pytorch_logits[i : i + 1]} for i in range(len(images))],
        runner_name="pytorch_fp32",
    )
    assert bool(
        Comparator.compare_accuracy(
            results,
            compare_func=CompareFunc.simple(atol=PARITY_ATOL, rtol=0.0),
        )
    )
    trt_top1 = torch.tensor(
        [int(out[LOGITS_OUTPUT].argmax()) for out in results["trt_fp16_ln32"]]
    )
    top2 = torch.from_numpy(pytorch_logits).topk(2, dim=1).values
    # Near-ties may flip under FP16. A frame whose FP32 margin exceeds 2
    # * atol cannot.
    clear = top2[:, 0] - top2[:, 1] > 2 * PARITY_ATOL
    agreement = (trt_top1 == torch.from_numpy(pytorch_logits).argmax(1)).float()
    print(
        f"top-1 agreement {agreement.mean():.3f} on {len(agreement)} frames, {agreement[clear].mean():.3f} on {int(clear.sum())} clear-margin frames"
    )
    assert agreement[clear].all()


def test_attention_softmax_is_exported_in_fp32(engines):
    graph = onnx.load(str(engines[ExportConfig()].with_suffix(".onnx"))).graph
    producers = {output: node for node in graph.node for output in node.output}
    softmaxes = [node for node in graph.node if node.op_type == "Softmax"]
    assert len(softmaxes) == 12
    for softmax in softmaxes:
        cast = producers[softmax.input[0]]
        assert cast.op_type == "Cast" and cast.attribute[0].i == onnx.TensorProto.FLOAT


def test_sub_fp16_layernorm_change_reaches_fp32_layernorm_engine_only(
    source_model, engines
):
    names = adaptable_parameter_names(source_model)
    packed = packed_adaptable_parameters(source_model)
    # A γ/β change that rounds to the same FP16 values: only an engine
    # that keeps γ/β and LayerNorm in FP32 can see it.
    moved = packed * (1 + 1e-4)
    moved = torch.where(moved.half() == packed.half(), moved, packed).contiguous()
    assert (
        torch.equal(moved.half(), packed.half())
        and (moved != packed).float().mean() > 0.5
    )
    images = torch.randn(
        1,
        3,
        224,
        224,
        device="cuda",
        generator=torch.Generator("cuda").manual_seed(0),
    )
    changes = {}
    for config in (ExportConfig(), ExportConfig(layernorm_fp32=False)):
        engine = TrtEngine(engines[config], names, [packed, moved])
        before = engine(images).clone()
        engine.publish(1, ready(), step=1)
        changes[config.layernorm_fp32] = (engine(images) - before).abs().max().item()
    print(
        f"max |Δlogits| for a sub-FP16 γ/β change: FP32 LayerNorm {changes[True]:.2e}, FP16 LayerNorm {changes[False]:.2e}"
    )
    assert changes[True] > 0
    assert changes[False] == 0


def test_rebound_layernorm_parameters_match_pytorch_with_the_same_parameters(
    source_model, engines, dev_stream_sample
):
    names = adaptable_parameter_names(source_model)
    images = torch.stack([image for image, _ in FrameDataset(dev_stream_sample)]).cuda()
    adapted = torch.empty_like(packed_adaptable_parameters(source_model))
    engine = TrtEngine(
        engines[ExportConfig()],
        names,
        [packed_adaptable_parameters(source_model), adapted],
    )
    source_logits = engine_logits(engine, images)

    generator = torch.Generator("cuda").manual_seed(0)
    with torch.no_grad():
        for param in configure_tent(source_model):
            param.mul_(
                1 + 0.05 * torch.randn(param.shape, device="cuda", generator=generator)
            )
        adapted_reference = source_model(images)
    # Written on a side stream: the engine's stream must wait on the
    # published event, as in the adapter's handoff.
    writer = torch.cuda.Stream()
    writer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(writer):
        adapted.copy_(packed_adaptable_parameters(source_model))
        engine.publish(1, ready(), step=1)
    adapted_logits = engine_logits(engine, images)

    error = (adapted_logits - adapted_reference).abs().max().item()
    shift = (adapted_reference - source_logits).abs().max().item()
    print(
        f"adapted γ/β: TRT vs PyTorch max |Δ| {error:.3f}, shift from source {shift:.3f}"
    )
    assert error < PARITY_ATOL < shift


def test_layernorm_inputs_and_constants_give_the_same_logits(
    source_model, engines, dev_stream_sample
):
    names = adaptable_parameter_names(source_model)
    images = torch.stack([image for image, _ in FrameDataset(dev_stream_sample)]).cuda()
    with_inputs = source_engine(engines[ExportConfig()], source_model)
    constants = TrtEngine(engines[ExportConfig(layernorm_inputs=False)], names)
    assert not constants.layernorm_names
    difference = (
        (engine_logits(with_inputs, images) - engine_logits(constants, images))
        .abs()
        .max()
        .item()
    )
    print(f"γ/β inputs vs constants: max |Δlogits| {difference:.2e}")
    assert difference < 0.05


def test_frame_path_runs_the_engine_on_decoded_frames(
    source_model, engines, dev_stream_sample
):
    engine = source_engine(engines[ExportConfig()], source_model)
    frame_path = FramePath(Source("trt_source", engine))
    decoded = FrameDataset(dev_stream_sample, transform=decode)
    images = torch.stack([image for image, _ in FrameDataset(dev_stream_sample)]).cuda()
    expected = engine_logits(engine, images).cpu()
    for i in range(len(decoded)):
        torch.testing.assert_close(
            frame_path(decoded[i][0]).clone(),
            expected[i : i + 1],
            rtol=0,
            atol=1e-6,
        )
        assert frame_path.event_span_ms() > 0


def test_publishing_a_buffer_is_cheap_on_the_host(source_model, engines):
    names = adaptable_parameter_names(source_model)
    buffers = [packed_adaptable_parameters(source_model) for _ in range(2)]
    engine = TrtEngine(engines[ExportConfig()], names, buffers)
    event = ready()
    start = time.perf_counter()
    for i in range(1000):
        engine.publish(i % 2, event, step=i)
    per_publish_us = (time.perf_counter() - start) * 1e3
    print(f"publish: {per_publish_us:.2f} µs per swap")
    # Rebinding 50 γ/β addresses took ~350 µs, a quarter of a frame
    # interval at max fps.
    assert per_publish_us < 20


def test_writer_waiting_on_the_release_event_never_changes_an_in_flight_inference(
    source_model, engines
):
    names = adaptable_parameter_names(source_model)
    buffers = [packed_adaptable_parameters(source_model) for _ in range(2)]
    engine = TrtEngine(engines[ExportConfig()], names, buffers)
    images = torch.randn(
        1,
        3,
        224,
        224,
        device="cuda",
        generator=torch.Generator("cuda").manual_seed(0),
    )
    expected = engine(images).clone()
    writer = torch.cuda.Stream()
    in_flight = {}
    for wait in (True, False):
        buffers[0].copy_(buffers[1])
        engine.publish(0, ready(), step=0)
        # Hold the frame stream for ~30 ms so the inference reading
        # buffer 0 is still queued when the adapter writes it.
        torch.cuda._sleep(50_000_000)
        in_flight[wait] = engine(images).clone()
        engine.publish(1, ready(), step=1)
        with pytest.raises(AssertionError, match="still in use"):
            engine.release_event(1)
        with torch.cuda.stream(writer):
            if wait:
                writer.wait_event(engine.release_event(0))
            buffers[0].fill_(float("nan"))
        torch.cuda.synchronize()
    assert torch.equal(in_flight[True], expected)
    # Without the wait the write overtakes the inference: the test can
    # see the race it guards against.
    assert in_flight[False].isnan().all()
