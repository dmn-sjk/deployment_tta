import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from tta_deploy.export.engine import built_engine
from tta_deploy.export.onnx_export import EXPORTS_BY_NAME, MAIN_EXPORT, ExportConfig
from tta_deploy.methods.method import Method
from tta_deploy.methods.pytorch import PytorchTent
from tta_deploy.methods.source import Source
from tta_deploy.methods.trt import TrtTent
from tta_deploy.model import load_source_model
from tta_deploy.runtime.adapter import Adapter, AdapterConfig
from tta_deploy.runtime.trt_engine import TrtEngine
from tta_deploy.tent import (
    Tent,
    TentConfig,
    adaptable_parameter_names,
    packed_adaptable_parameters,
)


def export_suffix(export: ExportConfig) -> str:
    return "" if export == MAIN_EXPORT else f"_{export.name}"


@dataclass(frozen=True, slots=True)
class PytorchSourceConfig:
    @property
    def name(self) -> str:
        return "pytorch_source"


@dataclass(frozen=True, slots=True)
class PytorchTentConfig:
    tent: TentConfig

    @property
    def name(self) -> str:
        return f"pytorch_{self.tent.name}"


@dataclass(frozen=True, slots=True)
class TrtSourceConfig:
    export: ExportConfig = MAIN_EXPORT

    @property
    def name(self) -> str:
        return f"trt_source{export_suffix(self.export)}"


@dataclass(frozen=True, slots=True)
class TrtTentConfig:
    adapter: AdapterConfig
    export: ExportConfig = MAIN_EXPORT
    # Adapter batch indices, counted from the last settle, whose input
    # gets a NaN: exercises fault handling.
    poisoned_batches: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        assert self.export.layernorm_inputs, "the adapter publishes γ/β as inputs"

    @property
    def name(self) -> str:
        poisoned = "_poisoned" if self.poisoned_batches else ""
        return f"trt_{self.adapter.name}{export_suffix(self.export)}{poisoned}"


MethodConfig = PytorchSourceConfig | PytorchTentConfig | TrtSourceConfig | TrtTentConfig


def build_method(config: MethodConfig, capacity: int) -> Method:
    """Loads the model and the built engine, starting from source.

    `capacity` bounds the frames a lag trace records.
    """
    match config:
        case PytorchSourceConfig():
            return Source(config.name, load_source_model().requires_grad_(False))
        case PytorchTentConfig(tent=tent):
            model = load_source_model()
            return PytorchTent(config.name, model, Tent(model, tent))
        case TrtSourceConfig(export=export):
            # Only the γ/β reach the device.
            model = load_source_model("cpu")
            buffers = (
                [packed_adaptable_parameters(model).cuda()]
                if export.layernorm_inputs
                else []
            )
            engine = TrtEngine(
                built_engine(export), adaptable_parameter_names(model), buffers
            )
            return Source(config.name, engine)
        case TrtTentConfig(adapter=adapter, export=export):
            model = load_source_model()
            packed = packed_adaptable_parameters(model)
            engine = TrtEngine(
                built_engine(export),
                adaptable_parameter_names(model),
                [packed, packed.clone()],
            )
            return TrtTent(
                config.name,
                Adapter(model, adapter, engine, config.poisoned_batches),
                capacity,
            )
    raise AssertionError(f"unknown method config {config}")


Kind = Literal[
    "pytorch_source", "pytorch_tent", "trt_source", "trt_tent_spec", "trt_tent_deploy"
]
KINDS: tuple[Kind, ...] = get_args(Kind)
# Deploy mode drops instead of falling behind, so it has no max fps
# (ADR 0002).
MAX_FPS_KINDS = tuple(k for k in KINDS if k != "trt_tent_deploy")


def kind(config: MethodConfig) -> Kind:
    match config:
        case PytorchSourceConfig():
            return "pytorch_source"
        case PytorchTentConfig():
            return "pytorch_tent"
        case TrtSourceConfig():
            return "trt_source"
        case TrtTentConfig(adapter=AdapterConfig(mode="spec")):
            return "trt_tent_spec"
        case TrtTentConfig(adapter=AdapterConfig(mode="deploy")):
            return "trt_tent_deploy"
    raise AssertionError(config)


def tent_config(config: MethodConfig) -> TentConfig | None:
    match config:
        case PytorchTentConfig(tent=tent):
            return tent
        case TrtTentConfig(adapter=adapter):
            return adapter.tent
    return None


def spec_trace_method(tent: TentConfig) -> TrtTentConfig:
    """The spec method of the main table, whose max-fps live run gives
    the lag trace the simulator replays."""
    return TrtTentConfig(AdapterConfig(tent, "spec", tent.B))


@dataclass(frozen=True)
class RunConfig:
    """What the report needs to know of a method, saved with its runs,
    so a report never parses method names."""

    kind: Kind
    # The reference config the method is compared with, None for
    # source.
    tent: str | None
    # The spec run of the main table at its (B, S), whose lag trace the
    # simulator replays.
    gated: bool
    B: int | None
    S: int | None
    micro_batch: int | None
    graph: bool | None
    export: str | None


def run_config(config: MethodConfig) -> RunConfig:
    adapter: AdapterConfig | None = None
    export: ExportConfig | None = None
    match config:
        case TrtTentConfig():
            adapter, export = config.adapter, config.export
        case TrtSourceConfig():
            export = config.export
    tent = tent_config(config)
    return RunConfig(
        kind=kind(config),
        tent=tent.name if tent else None,
        gated=tent is not None and config == spec_trace_method(tent),
        B=tent.B if tent else None,
        S=tent.S if tent else None,
        micro_batch=adapter.micro_batch if adapter else None,
        graph=adapter.graph if adapter else None,
        export=export.name if export else None,
    )


# Fields each method takes, from its CLI flags or experiment file.
METHOD_FIELDS = {
    "pytorch_source": set(),
    "pytorch_tent": {"B", "S"},
    "trt_source": {"export"},
    "trt_tent": {"B", "S", "mode", "micro_batch", "graph", "export"},
}
REQUIRED_FIELDS = {"pytorch_tent": {"B", "S"}, "trt_tent": {"B", "S", "mode"}}


def method_config(fields: Mapping[str, Any]) -> MethodConfig:
    # Fields set to None count as not given.
    given = {k: v for k, v in fields.items() if v is not None}
    method = given.pop("method")
    assert method in METHOD_FIELDS, f"unknown method {method}"
    extra = given.keys() - METHOD_FIELDS[method]
    assert not extra, f"{method} does not take {sorted(extra)}, given {given}"
    missing = REQUIRED_FIELDS.get(method, set()) - given.keys()
    assert not missing, f"{method} needs {sorted(missing)}, given {given}"
    export = EXPORTS_BY_NAME[given.get("export", MAIN_EXPORT.name)]
    match method:
        case "pytorch_source":
            return PytorchSourceConfig()
        case "pytorch_tent":
            return PytorchTentConfig(TentConfig(given["B"], given["S"]))
        case "trt_source":
            return TrtSourceConfig(export)
        case "trt_tent":
            tent = TentConfig(given["B"], given["S"])
            adapter = AdapterConfig(
                tent,
                given["mode"],
                given.get("micro_batch", tent.B),
                given.get("graph", True),
            )
            return TrtTentConfig(adapter, export)
    raise AssertionError(method)


def add_method_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=METHOD_FIELDS, required=True)
    parser.add_argument("--B", type=int)
    parser.add_argument("--S", type=int)
    parser.add_argument("--mode", choices=["spec", "deploy"])
    parser.add_argument("--micro-batch", type=int, help="default: B")
    parser.add_argument(
        "--graph", action=argparse.BooleanOptionalAction, help="default: graph"
    )
    parser.add_argument("--export", choices=EXPORTS_BY_NAME, help="default: main")


def method_from_args(args: argparse.Namespace) -> MethodConfig:
    fields = {"method"}.union(*METHOD_FIELDS.values())
    return method_config({k: v for k, v in vars(args).items() if k in fields})


def max_fps_method_from_args(args: argparse.Namespace) -> MethodConfig:
    method = method_from_args(args)
    # Deploy mode reports no adaptation progress, so the keeps-up check
    # of the bisection would silently not apply.
    assert kind(method) in MAX_FPS_KINDS, f"{method.name} has no max fps (ADR 0002)"
    return method
