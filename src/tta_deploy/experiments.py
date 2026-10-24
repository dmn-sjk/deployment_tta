import argparse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tta_deploy.export.engine import TRTEXEC_BUILT
from tta_deploy.export.onnx_export import EXPORTS_BY_NAME
from tta_deploy.methods.configs import (
    add_method_args,
    max_fps_method_from_args,
    method_from_args,
)
from tta_deploy.paths import reference_set
from tta_deploy.tent import TentConfig

SEEDS = {"dev": (1,), "full": (1, 2, 3)}
# Same-rate comparison: every method sees the same camera rate.
SAME_RATE_FPS = 30.0
LAGS = (0, 1, 2, 4, 8, 16)
TRTEXEC_ENGINES = (*EXPORTS_BY_NAME, TRTEXEC_BUILT)


def add_stream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--subset", choices=SEEDS, default="dev")
    parser.add_argument("--seed", type=int, default=1)


def input_rate(text: str) -> float:
    fps = float(text)
    if fps <= 0:
        raise argparse.ArgumentTypeError(f"input rate {text} must be > 0")
    return fps


def add_realtime_args(parser: argparse.ArgumentParser) -> None:
    add_method_args(parser)
    parser.add_argument("--fps", type=input_rate, required=True, help="input rate")


def add_reference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--B", type=int, help="with --S, unless --source")
    parser.add_argument("--S", type=int)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--rerun",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="for the determinism check",
    )


def reference_from_args(args: argparse.Namespace) -> TentConfig | None:
    if args.source:
        assert args.B is None and args.S is None, "source takes no B, S"
        return None
    assert args.B is not None and args.S is not None, "need --B, --S or --source"
    return TentConfig(args.B, args.S)


def reference_name(tent: TentConfig | None) -> str:
    return tent.name if tent else "source"


def reference_run_name(args: argparse.Namespace) -> str:
    references = reference_set(args.precision, args.rerun)
    return f"{references}/{reference_name(reference_from_args(args))}"


def add_lag_sim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--S", type=int, required=True)
    parser.add_argument(
        "--trace-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="replay a new live trace, keep the fixed-lag results",
    )


def add_trtexec_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=TRTEXEC_ENGINES, required=True)


@dataclass(frozen=True)
class AdapterBench:
    """A standalone adapter step: only B, the micro-batch and the CUDA
    graph matter."""

    B: int
    micro_batch: int
    graph: bool

    def __post_init__(self) -> None:
        assert self.B >= 1, f"B {self.B} must be >= 1"
        assert self.micro_batch >= 1 and self.B % self.micro_batch == 0, (
            f"B {self.B} must be a multiple of the micro-batch {self.micro_batch}"
        )

    @property
    def name(self) -> str:
        return f"B{self.B}_m{self.micro_batch}_{'graph' if self.graph else 'eager'}"


def add_adapter_bench_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--micro-batch", type=int, help="default: B")
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)


def adapter_bench_from_args(args: argparse.Namespace) -> AdapterBench:
    m = args.B if args.micro_batch is None else args.micro_batch
    return AdapterBench(args.B, m, args.graph)


@dataclass(frozen=True)
class ExperimentKind:
    script: str
    # The flags an experiment file sets, without --subset and --seed.
    add_args: Callable[[argparse.ArgumentParser], None]
    # Checks the parsed flags and names the run, unique in a file.
    run_name: Callable[[argparse.Namespace], str]
    # Reads the stream: one run per seed of the subset. Benchmarks run
    # once, whatever the subset.
    per_seed: bool


EXPERIMENT_KINDS = {
    "realtime": ExperimentKind(
        "run.py",
        add_realtime_args,
        lambda args: f"{method_from_args(args).name}/fps{args.fps:g}",
        per_seed=True,
    ),
    # The run name is its output dir name.
    "max_fps": ExperimentKind(
        "max_fps_bisect.py",
        add_method_args,
        lambda args: max_fps_method_from_args(args).name,
        per_seed=True,
    ),
    "reference": ExperimentKind(
        "reference_run.py",
        add_reference_args,
        reference_run_name,
        per_seed=True,
    ),
    "lag_sim": ExperimentKind(
        "lag_sim.py",
        add_lag_sim_args,
        lambda args: TentConfig(args.B, args.S).name,
        per_seed=True,
    ),
    "trtexec": ExperimentKind(
        "trtexec_bench.py",
        add_trtexec_bench_args,
        lambda args: args.engine,
        per_seed=False,
    ),
    "adapter": ExperimentKind(
        "adapter_bench.py",
        add_adapter_bench_args,
        lambda args: adapter_bench_from_args(args).name,
        per_seed=False,
    ),
}


@dataclass(frozen=True)
class ExperimentRun:
    name: str
    # The run script's flags, without --subset and --seed.
    argv: tuple[str, ...]
    # max_fps only: measurements of the same max fps, each in its own
    # output dir.
    repeats: int


def fields_argv(fields: Mapping[str, Any]) -> list[str]:
    argv = []
    for key, value in fields.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            argv.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            argv += [f"--{flag}", str(value)]
    return argv


def read_experiment(path: Path) -> tuple[str, list[ExperimentRun]]:
    """An experiment file: what it measures and the runs it makes.

    A run's fields are the flags of its script. A run with a `tent` list
    expands to one run per TENT config, an `fps` list to one run per
    input rate. `repeats` (max_fps only) measures each run that many
    times. `grids` only holds YAML anchors for the runs to reuse.
    """
    spec = yaml.safe_load(path.read_text())
    extra = spec.keys() - {"experiment", "grids", "runs"}
    assert not extra, f"{path}: unknown keys {sorted(extra)}"
    experiment = spec["experiment"]
    assert experiment in EXPERIMENT_KINDS, f"{path}: unknown experiment {experiment}"
    kind = EXPERIMENT_KINDS[experiment]
    assert spec["runs"], f"{path}: no runs"
    runs = []
    for entry in spec["runs"]:
        # Fields are checked by the same parser as a command line, which
        # names the entry on an error.
        parser = argparse.ArgumentParser(prog=f"{path}: {entry}", allow_abbrev=False)
        kind.add_args(parser)
        fields = dict(entry)
        tents = fields.pop("tent", [{}])
        assert tents, f"{path}: empty tent list in {entry}"
        rates = fields.pop("fps", None)
        assert rates is None or isinstance(rates, list), (
            f"{path}: fps takes a list: {entry}"
        )
        assert rates != [], f"{path}: empty fps list in {entry}"
        repeats = fields.pop("repeats", 1)
        assert isinstance(repeats, int) and repeats >= 1, f"{path}: repeats {entry}"
        assert repeats == 1 or experiment == "max_fps", (
            f"{path}: repeats is for max_fps only: {entry}"
        )
        for tent in tents:
            assert tent.keys() <= {"B", "S"}, f"{path}: TENT config {tent}"
            assert not tent.keys() & fields.keys(), f"{path}: B or S twice: {entry}"
            for rate in [{}] if rates is None else [{"fps": f} for f in rates]:
                argv = fields_argv(fields | rate | tent)
                name = kind.run_name(parser.parse_args(argv))
                runs.append(ExperimentRun(name, tuple(argv), repeats))
    names = [r.name for r in runs]
    repeated = sorted({n for n in names if names.count(n) > 1})
    assert not repeated, f"{path}: runs listed more than once: {repeated}"
    return experiment, runs
