import argparse
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tta_deploy import paths
from tta_deploy.eval.replay import LiveStepDown
from tta_deploy.experiments import (
    EXPERIMENT_KINDS,
    AdapterBench,
    adapter_bench_from_args,
    add_adapter_bench_args,
    read_experiment,
    reference_from_args,
)
from tta_deploy.export.engine import TRTEXEC_BUILT
from tta_deploy.export.onnx_export import MAIN_EXPORT
from tta_deploy.methods.configs import (
    TrtTentConfig,
    add_method_args,
    method_config,
    method_from_args,
    spec_trace_method,
    tent_config,
)
from tta_deploy.report import load
from tta_deploy.report.load import load_max_fps, load_trtexec
from tta_deploy.report.tables import step_in_flight_latency
from tta_deploy.runtime.adapter import AdapterConfig
from tta_deploy.tent import TentConfig

EXPERIMENTS = Path(__file__).parents[1] / "experiments"
SCRIPTS = Path(__file__).parents[1] / "scripts"


def method_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_method_args(parser)
    return parser


def run_names(path: Path) -> list[str]:
    return [r.name for r in read_experiment(path)[1]]


def test_every_experiment_file_loads() -> None:
    for path in sorted(EXPERIMENTS.glob("*.yaml")):
        assert read_experiment(path)[1], path


def run_tents(name: str) -> tuple[set[TentConfig | None], set[TentConfig]]:
    # The TENT configs (None for source) of the runs, and of the gated
    # spec runs (the main table's).
    experiment, runs = read_experiment(EXPERIMENTS / name)
    tents: set[TentConfig | None] = set()
    gated = set()
    for run in runs:
        parser = argparse.ArgumentParser()
        EXPERIMENT_KINDS[experiment].add_args(parser)
        args = parser.parse_args(run.argv)
        match experiment:
            case "realtime" | "max_fps":
                method = method_from_args(args)
                tent = tent_config(method)
                if tent and method == spec_trace_method(tent):
                    gated.add(tent)
            case "reference":
                tent = reference_from_args(args)
            case "lag_sim":
                tent = TentConfig(args.B, args.S)
        tents.add(tent)
    return tents, gated


def test_every_live_config_has_a_reference() -> None:
    references = run_tents("reference.yaml")[0]
    assert run_tents("reference_checks.yaml")[0] == references
    for name in (
        "realtime.yaml",
        "realtime_variants.yaml",
        "max_fps.yaml",
        "max_fps_variants.yaml",
    ):
        assert run_tents(name)[0] <= references, name
    # Every reference (B, S) has a main-table spec row, whose trace the
    # simulator replays.
    gated = run_tents("max_fps.yaml")[1]
    assert gated == references - {None}
    assert run_tents("lag_sim.yaml")[0] == gated


def test_realtime_runs_expand_per_tent_config_and_rate(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text(
        trt_tent_run("mode: deploy, tent: [{B: 1, S: 1}, {B: 4, S: 2}]", "[30, 60]")
    )
    experiment, runs = read_experiment(path)
    assert [r.name for r in runs] == [
        f"trt_tent_{t}_deploy_m{m}_graph/fps{fps}"
        for t, m in (("B1_S1", 1), ("B4_S2", 4))
        for fps in (30, 60)
    ]
    assert runs[1].argv == (
        *("--method", "trt_tent", "--mode", "deploy", "--fps", "60"),
        *("--B", "1", "--S", "1"),
    )


def test_method_flag_defaults() -> None:
    argv = ["--method", "trt_tent", "--mode", "spec", "--B", "16", "--S", "1"]
    assert method_from_args(method_parser().parse_args(argv)) == TrtTentConfig(
        AdapterConfig(TentConfig(16, 1), "spec", 16, graph=True), MAIN_EXPORT
    )


def test_adapter_bench_flag_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_adapter_bench_args(parser)
    bench = adapter_bench_from_args(parser.parse_args(["--B", "16"]))
    assert bench == AdapterBench(16, 16, graph=True)
    assert bench.name == "B16_m16_graph"
    assert AdapterBench(64, 4, graph=False).name == "B64_m4_eager"


def test_rejects_fields_a_method_does_not_take_or_needs() -> None:
    with pytest.raises(AssertionError, match="does not take"):
        method_config({"method": "pytorch_source", "mode": "spec"})
    with pytest.raises(AssertionError, match="does not take"):
        method_config({"method": "trt_source", "B": 16})
    with pytest.raises(AssertionError, match=r"needs \['mode'\]"):
        method_config({"method": "trt_tent", "B": 16, "S": 1})


def test_off_grid_tent_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "off_grid.yaml"
    path.write_text(trt_tent_run("mode: spec, B: 8, S: 3"))
    assert run_names(path) == ["trt_tent_B8_S3_spec_m8_graph/fps30"]


ONE_RUN = "experiment: realtime\nruns:\n  - {method: trt_source, fps: [30]}\n"


def trt_tent_run(fields: str, fps: str = "[30]") -> str:
    return (
        f"experiment: realtime\nruns:\n  - {{method: trt_tent, fps: {fps}, {fields}}}\n"
    )


@pytest.mark.parametrize(
    ("text", "error"),
    [
        (ONE_RUN + "fsp: 30\n", "unknown keys"),
        ("experiment: realtme\nruns:\n  - {}\n", "unknown experiment"),
        ("experiment: realtime\nruns: []\n", "no runs"),
        (
            "experiment: max_fps\nruns:\n"
            "  - {method: trt_tent, mode: deploy, tent: [{B: 1, S: 1}]}\n",
            "no max fps",
        ),
        (
            "experiment: realtime\nruns:\n  - {method: trt_source, fps: [30, 30]}\n",
            "more than once",
        ),
        (ONE_RUN + "  - {method: trt_source, fps: [60, 30]}\n", "more than once"),
        (trt_tent_run("mode: spec, tent: []"), "empty tent list"),
        (trt_tent_run("mode: spec, B: 1, S: 1", fps="[]"), "empty fps list"),
        (trt_tent_run("mode: spec, B: 1, S: 1", fps="30"), "fps takes a list"),
        (trt_tent_run("mode: spec, B: 1, tent: [{B: 4, S: 1}]"), "B or S twice"),
        (trt_tent_run("mode: spec, B: 16, S: 1, micro_batch: 3"), "multiple"),
        (trt_tent_run("mode: spec, B: 1, S: 0"), "must be >= 1"),
        (trt_tent_run("mode: spec, B: 1, S: 1, repeats: 2"), "max_fps only"),
        (
            "experiment: max_fps\nruns:\n  - {method: trt_source, repeats: 0}\n",
            "repeats",
        ),
        ("experiment: reference\nruns:\n  - {source: true, B: 1}\n", "no B, S"),
        (
            "experiment: adapter\nruns:\n  - {B: 16, micro_batch: 3, graph: true}\n",
            "multiple",
        ),
        ("experiment: adapter\nruns:\n  - {B: 16, micro_batch: 0}\n", "multiple"),
        ("experiment: adapter\nruns:\n  - {B: 0}\n", ">= 1"),
        ("experiment: reference\nruns:\n  - {B: 1}\n", "need --B, --S"),
        (
            "experiment: reference\nruns:\n"
            "  - {source: true}\n  - {source: true, precision: fp32}\n",
            "more than once",
        ),
    ],
)
def test_rejects_bad_experiment_files(tmp_path: Path, text: str, error: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(AssertionError, match=error):
        read_experiment(path)


@pytest.mark.parametrize(
    "text",
    [
        trt_tent_run("mode: Spec, B: 1, S: 1"),
        trt_tent_run("mode: spec, B: 16.5, S: 1"),
        trt_tent_run("mode: spec, B: null, S: 1"),
        # A quoted "false" must not pass as true.
        trt_tent_run("mode: spec, B: 1, S: 1, graph: 'false'"),
        # An abbreviation of --micro-batch must not pass as a typo.
        trt_tent_run("mode: spec, B: 16, S: 1, micro: 4"),
        "experiment: realtime\nruns:\n  - {method: trt_source}\n",
        "experiment: realtime\nruns:\n  - {method: trt_source, fps: [0]}\n",
        "experiment: max_fps\nruns:\n  - {method: trt_source, fps: [30]}\n",
        "experiment: reference\nruns:\n  - {source: true, fps: [30]}\n",
        "experiment: lag_sim\nruns:\n  - {B: 1}\n",
    ],
)
def test_bad_flags_fail_at_load(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(SystemExit):
        read_experiment(path)


def test_report_reads_only_current_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    for engine in (TRTEXEC_BUILT, "vit_removed_engine"):
        out = paths.trtexec_bench_path(engine)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"gpu_median_ms": 1.0}))
    assert load_trtexec() == {TRTEXEC_BUILT: {"gpu_median_ms": 1.0}}


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recorded(
    orchestrate: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[str, ...]]:
    """Commands orchestrate issues, each parsed by its script's own
    parser, so the two cannot drift."""
    issued: list[tuple[str, ...]] = []

    def run_script(script: Path, *argv: str) -> None:
        load_script(script.stem).parser().parse_args(argv)
        issued.append((script.name, *argv))

    monkeypatch.setattr(orchestrate, "run_script", run_script)
    return issued


def orchestrate_main(
    orchestrate: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    text: str,
    *argv: str,
) -> None:
    path.write_text(text)
    monkeypatch.setattr(sys, "argv", ["orchestrate.py", str(path), *argv])
    orchestrate.main()


STREAM = ("--subset", "dev", "--seed", "1")


@pytest.mark.parametrize(
    ("text", "argv", "commands"),
    [
        (
            "experiment: realtime\nruns:\n  - {method: pytorch_source, fps: [30, 60]}\n",
            (),
            [
                ("run.py", "--method", "pytorch_source", "--fps", "30", *STREAM),
                ("run.py", "--method", "pytorch_source", "--fps", "60", *STREAM),
            ],
        ),
        (
            "experiment: realtime\nruns:\n  - {method: pytorch_source, fps: [30]}\n",
            ("--subset", "full"),
            [
                (
                    *("run.py", "--method", "pytorch_source", "--fps", "30"),
                    *("--subset", "full", "--seed", str(seed)),
                )
                for seed in (1, 2, 3)
            ],
        ),
        (
            "experiment: reference\nruns:\n"
            "  - {source: true}\n  - {tent: [{B: 16, S: 1}], precision: bf16}\n",
            (),
            [
                ("reference_run.py", "--source", *STREAM),
                (
                    *("reference_run.py", "--precision", "bf16"),
                    *("--B", "16", "--S", "1", *STREAM),
                ),
            ],
        ),
        (
            "experiment: lag_sim\nruns:\n"
            "  - {B: 1, S: 1}\n  - {B: 16, S: 1, trace_only: true}\n",
            (),
            [
                ("lag_sim.py", "--B", "1", "--S", "1", *STREAM),
                ("lag_sim.py", "--B", "16", "--S", "1", "--trace-only", *STREAM),
            ],
        ),
        # Benchmarks run once, for no seed, whatever the subset.
        (
            "experiment: adapter\nruns:\n  - {B: 16, micro_batch: 4, graph: false}\n",
            ("--subset", "full"),
            [("adapter_bench.py", "--B", "16", "--micro-batch", "4", "--no-graph")],
        ),
        (
            "experiment: trtexec\nruns:\n  - {engine: vit_fp16_lninputs_ln32_sm32}\n",
            ("--subset", "full"),
            [("trtexec_bench.py", "--engine", "vit_fp16_lninputs_ln32_sm32")],
        ),
    ],
)
def test_orchestrate_issues_each_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    argv: tuple[str, ...],
    commands: list[tuple[str, ...]],
) -> None:
    orchestrate = load_script("orchestrate")
    issued = recorded(orchestrate, monkeypatch)
    orchestrate_main(orchestrate, monkeypatch, tmp_path / "x.yaml", text, *argv)
    assert issued == commands


def test_orchestrate_max_fps_bisects_then_steps_down_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrate = load_script("orchestrate")
    out = tmp_path / "max_fps"
    out.mkdir()
    (out / "max_fps.json").write_text("{}")
    monkeypatch.setattr(
        orchestrate, "max_fps_dir", lambda subset, name, seed, repeat: out
    )
    issued = recorded(orchestrate, monkeypatch)
    record = orchestrate.run_script

    def run_script(script: Path, *argv: str) -> None:
        record(script, *argv)
        if script.name == "max_fps_bisect.py":
            bisection = {
                "bisected_fps": 110.0,
                "max_fps": 100.0,
                "confirm_step_downs": 1,
            }
            (out / "bisection.json").write_text(json.dumps(bisection))
        else:
            assert not (out / "max_fps.json").exists(), "stale max fps left in place"
            (out / "metrics.json").write_text(json.dumps({"fps": float(argv[-3])}))

    def live_step_down(
        confirmed_fps: float, live_run: Callable[[float], dict]
    ) -> LiveStepDown:
        assert live_run(confirmed_fps)["fps"] == confirmed_fps
        return LiveStepDown(confirmed_fps, 0, [])

    monkeypatch.setattr(orchestrate, "run_script", run_script)
    monkeypatch.setattr(orchestrate, "live_step_down", live_step_down)
    orchestrate_main(
        orchestrate,
        monkeypatch,
        tmp_path / "max_fps.yaml",
        "experiment: max_fps\nruns:\n  - {method: pytorch_source}\n",
    )
    method = ("--method", "pytorch_source", *STREAM)
    assert issued == [
        ("max_fps_bisect.py", *method, "--out", str(out)),
        ("run.py", *method, "--fps", "100.0", "--out", str(out)),
    ]
    held = json.loads((out / "max_fps.json").read_text())
    assert (held["confirmed_fps"], held["max_fps"], held["live_step_downs"]) == (
        100.0,
        100.0,
        0,
    )


def test_orchestrate_max_fps_repeats_each_into_its_own_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrate = load_script("orchestrate")
    monkeypatch.setattr(
        orchestrate,
        "max_fps_dir",
        lambda subset, name, seed, repeat: tmp_path / f"repeat{repeat}",
    )
    issued = recorded(orchestrate, monkeypatch)
    record = orchestrate.run_script

    def run_script(script: Path, *argv: str) -> None:
        record(script, *argv)
        out = Path(argv[-1])
        out.mkdir()
        bisection = {"bisected_fps": 100.0, "max_fps": 100.0, "confirm_step_downs": 0}
        (out / "bisection.json").write_text(json.dumps(bisection))

    monkeypatch.setattr(orchestrate, "run_script", run_script)
    monkeypatch.setattr(
        orchestrate, "live_step_down", lambda fps, live_run: LiveStepDown(fps, 0, [])
    )
    orchestrate_main(
        orchestrate,
        monkeypatch,
        tmp_path / "max_fps.yaml",
        "experiment: max_fps\nruns:\n  - {method: pytorch_source, repeats: 2}\n",
    )
    method = ("--method", "pytorch_source", *STREAM)
    assert issued == [
        ("max_fps_bisect.py", *method, "--out", str(tmp_path / f"repeat{repeat}"))
        for repeat in (1, 2)
    ]


def test_max_fps_reports_the_median_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(load, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    for repeat, fps in ((1, 271.0), (2, 290.0), (3, 258.0)):
        out = paths.max_fps_dir("dev", "trt_tent_B16_S1", 1, repeat)
        out.mkdir(parents=True)
        (out / "bisection.json").write_text(json.dumps({"repeat": repeat}))
        (out / "max_fps.json").write_text(json.dumps({"max_fps": fps}))
        (out / "metrics.json").write_text(json.dumps({"fps": fps, "sustainable": True}))
    # A repeat whose live run never held is left out.
    paths.max_fps_dir("dev", "trt_tent_B16_S1", 1, 4).mkdir()
    run = load_max_fps("dev", 1)["trt_tent_B16_S1"]
    assert (run.held["max_fps"], run.bisection["repeat"]) == (271.0, 1)
    assert run.repeat_fps == (258.0, 271.0, 290.0)


def test_step_in_flight_latency_splits_frames() -> None:
    frames = {
        "latency_ms": np.array([1.0, 3.0, 3.0, 1.0]),
        "accepted_batches": np.array([0, 1, 1, 1]),
        "prediction_steps": np.array([0, 0, 0, 1]),
    }
    in_flight, idle = step_in_flight_latency(frames)
    assert in_flight.tolist() == [3.0, 3.0] and idle.tolist() == [1.0, 1.0]
