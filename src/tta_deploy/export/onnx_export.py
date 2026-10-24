import copy
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from torch import nn
from torch.nn import functional as F

from tta_deploy.tent import adaptable_parameter_names

# opset 17 is the first version with a native LayerNormalization op
OPSET = 17
IMAGE_INPUT = "images"
LOGITS_OUTPUT = "logits"
IMAGE_SHAPE = (1, 3, 224, 224)


@dataclass(frozen=True)
class ExportConfig:
    layernorm_inputs: bool = True
    layernorm_fp32: bool = True
    # FP16 softmax moved fog's top-1 agreement below bf16 autocast's.
    softmax_fp32: bool = True

    @property
    def name(self) -> str:
        return f"vit_fp16_ln{'inputs' if self.layernorm_inputs else 'const'}_ln{'32' if self.layernorm_fp32 else '16'}_sm{'32' if self.softmax_fp32 else '16'}"


MAIN_EXPORT = ExportConfig()
ENGINE_VARIANTS = (
    ExportConfig(softmax_fp32=False),
    ExportConfig(layernorm_fp32=False),
    ExportConfig(layernorm_inputs=False),
    ExportConfig(layernorm_inputs=False, layernorm_fp32=False),
)
EXPORTS = (MAIN_EXPORT, *ENGINE_VARIANTS)
EXPORTS_BY_NAME = {e.name: e for e in EXPORTS}


class MixedPrecisionLayerNorm(nn.Module):
    """FP16 activations in and out, γ/β in FP32.

    With compute_fp32 the normalization itself also runs in FP32.
    """

    def __init__(self, layernorm: nn.LayerNorm, compute_fp32: bool):
        super().__init__()
        self.normalized_shape, self.eps, self.compute_fp32 = (
            layernorm.normalized_shape,
            layernorm.eps,
            compute_fp32,
        )
        # Buffers, not parameters: the traced γ/β inputs replace them
        # by name in forward, which a Parameter would refuse.
        self.register_buffer("weight", layernorm.weight.detach().float().clone())
        self.register_buffer("bias", layernorm.bias.detach().float().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.compute_fp32:
            return F.layer_norm(
                x.float(),
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ).half()

        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight.half(),
            self.bias.half(),
            self.eps,
        )


class Fp16Export(nn.Module):
    """FP16 copy of the model with explicit LayerNorm precision, FP32
    image input and logits, optionally γ/β as FP32 inputs."""

    def __init__(self, model: nn.Module, config: ExportConfig):
        super().__init__()
        self.layernorm_inputs = (
            adaptable_parameter_names(model) if config.layernorm_inputs else []
        )

        # Lower the precision before initializing the
        # MixedPrecisionLayerNorm.
        self.model = copy.deepcopy(model).half()

        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                # Module.set_submodule in PyTorch 2.5 fails on nested
                # names.
                parent, _, child = name.rpartition(".")
                setattr(
                    self.model.get_submodule(parent),
                    child,
                    MixedPrecisionLayerNorm(module, config.layernorm_fp32),
                )

    def forward(
        self, images: torch.Tensor, *layernorm_parameters: torch.Tensor
    ) -> torch.Tensor:
        # torch.func.functional_call refuses to run under the ONNX
        # tracer, so the graph inputs replace the γ/β buffers in place.
        for name, value in zip(
            self.layernorm_inputs, layernorm_parameters, strict=True
        ):
            module, _, buffer = name.rpartition(".")
            setattr(self.model.get_submodule(module), buffer, value)
        return self.model(images.half()).float()


def cast_softmax_to_fp32(graph: onnx.GraphProto) -> None:
    # The exporter emits attention softmax in FP16 and PyTorch has no
    # hook for its precision, so the casts are added to the graph.
    nodes = []
    for node in graph.node:
        if node.op_type != "Softmax":
            nodes.append(node)
            continue
        x, y = node.input[0], node.output[0]
        nodes.append(
            onnx.helper.make_node(
                "Cast",
                [x],
                [f"{x}_fp32"],
                name=f"{node.name}_to_fp32",
                to=onnx.TensorProto.FLOAT,
            )
        )
        node.input[0], node.output[0] = f"{x}_fp32", f"{y}_fp32"
        nodes.append(node)
        nodes.append(
            onnx.helper.make_node(
                "Cast",
                [f"{y}_fp32"],
                [y],
                name=f"{node.name}_to_fp16",
                to=onnx.TensorProto.FLOAT16,
            )
        )
    del graph.node[:]
    graph.node.extend(nodes)


def export_onnx(model: nn.Module, path: Path, config: ExportConfig) -> None:
    """Batch-1 export whose tensor types fix every precision, for a
    strongly typed TensorRT network."""
    # wrap to be able to pass LN weights as inputs
    wrapped = Fp16Export(model, config)

    images = torch.zeros(IMAGE_SHAPE, device=next(model.parameters()).device)
    state = model.state_dict()

    args = (images, *(state[n] for n in wrapped.layernorm_inputs))
    # LN weights as inputs
    input_names = [IMAGE_INPUT, *wrapped.layernorm_inputs]
    path.parent.mkdir(parents=True, exist_ok=True)

    # The no-grad MultiheadAttention fast path is a fused aten op that
    # has no ONNX symbolic.
    # Make torch use non fused path (F.multi_head_attention_forward)
    # for ONNX to correctly export it.
    # It won't add latency since TensorRT fuses those ops by itself.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)

    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapped,
                args,
                str(path),
                opset_version=OPSET,
                input_names=input_names,
                output_names=[LOGITS_OUTPUT],
                do_constant_folding=True,
            )
    finally:
        # make sure it is set to the previous value
        torch.backends.mha.set_fastpath_enabled(fastpath)

    modules = list(model.modules())
    layernorms = sum(isinstance(m, nn.LayerNorm) for m in modules)
    attentions = sum(isinstance(m, nn.MultiheadAttention) for m in modules)

    exported = onnx.load(str(path))
    graph = exported.graph
    assert (
        sum(node.op_type == "LayerNormalization" for node in graph.node) == layernorms
    ), "LayerNorm not exported as native LayerNormalization"
    assert sum(node.op_type == "Softmax" for node in graph.node) == attentions, (
        "one softmax per attention"
    )

    if config.softmax_fp32:
        cast_softmax_to_fp32(graph)
        onnx.checker.check_model(exported)
        onnx.save(exported, str(path))

    assert [i.name for i in graph.input] == input_names
    assert all(
        i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT for i in graph.input
    ), "image and γ/β inputs must be FP32"
