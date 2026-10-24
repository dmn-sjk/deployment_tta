import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from tta_deploy.experiments import TRTEXEC_ENGINES
from tta_deploy.methods.configs import RunConfig
from tta_deploy.paths import (
    RUNS_DIR,
    bench_dir,
    max_fps_repeat_dirs,
    offline_dir,
    trtexec_bench_path,
)


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@dataclass(frozen=True)
class Run:
    """One saved run: its metrics and per-frame arrays."""

    metrics: dict
    frames_npz: Path

    @property
    def config(self) -> RunConfig:
        """Live runs only: the reference has no method."""
        assert "config" in self.metrics, (
            f"{self.frames_npz.parent}: saved without its run config, rerun it"
        )
        return RunConfig(**self.metrics["config"])

    def frames(self) -> dict[str, np.ndarray]:
        with np.load(self.frames_npz) as frames:
            return dict(frames)


def load_reference(subset: str, seed: int, name: str) -> dict[str, Run]:
    """Reference runs by config name ('source', 'tent_B16_S1', ...),
    from offline/seed<n>/<name>, e.g. reference_fp32."""
    runs = {}
    for path in sorted((offline_dir(subset, seed) / name).glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        runs[metrics["config"]] = Run(metrics, path.parent / "frames.npz")
    # Every comparison is against source.
    assert not runs or "source" in runs, (
        f"{offline_dir(subset, seed) / name}: no source run, run reference_run.py --source"
    )
    return runs


def load_realtime(subset: str) -> list[Run]:
    """Every real-time run of the subset, any method, rate and seed."""
    return [
        Run(json.loads(path.read_text()), path.parent / "frames.npz")
        for path in sorted((RUNS_DIR / subset / "realtime").glob("*/*/metrics.json"))
    ]


@dataclass(frozen=True)
class MaxFpsRun:
    bisection: dict
    held: dict
    live: Run
    # Max fps of every repeat that held, ascending.
    repeat_fps: tuple[float, ...]


def held_repeat(run_dir: Path) -> MaxFpsRun | None:
    """One repeat's max fps and the live run that held it, or None if no
    step-down completed."""
    if not (run_dir / "max_fps.json").exists():
        return None
    bisection, held, live = (
        json.loads((run_dir / f).read_text())
        for f in ("bisection.json", "max_fps.json", "metrics.json")
    )
    assert live["fps"] == held["max_fps"] and live["sustainable"], (
        f"{run_dir}: metrics are not from the live run that held"
    )
    return MaxFpsRun(
        bisection, held, Run(live, run_dir / "frames.npz"), (held["max_fps"],)
    )


def load_held_max_fps(subset: str, name: str, seed: int) -> MaxFpsRun | None:
    """The repeat with the median max fps (the lower one of an even
    count) among those whose live run held."""
    repeats = sorted(
        (
            run
            for run_dir in max_fps_repeat_dirs(subset, name, seed)
            if (run := held_repeat(run_dir)) is not None
        ),
        key=lambda run: run.held["max_fps"],
    )
    if not repeats:
        return None
    median = repeats[(len(repeats) - 1) // 2]
    return replace(median, repeat_fps=tuple(r.held["max_fps"] for r in repeats))


def load_max_fps(subset: str, seed: int) -> dict[str, MaxFpsRun]:
    """Max-fps results by method name, only where a live run held."""
    runs = {}
    for name_dir in sorted((RUNS_DIR / subset / "max_fps").glob("*")):
        if (run := load_held_max_fps(subset, name_dir.name, seed)) is not None:
            runs[name_dir.name] = run
    return runs


def load_lag_sim(subset: str, seed: int) -> dict[str, dict]:
    """By config: the fixed lags, L0_matches_reference and the trace."""
    return {
        path.parent.name: json.loads(path.read_text())
        | {"trace": json.loads((path.parent / "trace.json").read_text())}
        for path in sorted(
            (offline_dir(subset, seed) / "lag_sim").glob("*/fixed_lags.json")
        )
    }


def load_trtexec() -> dict[str, dict]:
    """By engine, the engines that exist now: a removed engine's result
    is left out."""
    return {
        engine: result
        for engine in TRTEXEC_ENGINES
        if (result := load_json(trtexec_bench_path(engine)))
    }


def load_adapter_bench() -> dict[tuple[int, int, bool], dict]:
    return {
        (r["B"], r["micro_batch"], r["graph"]): r
        for r in (
            json.loads(p.read_text())
            for p in sorted((bench_dir() / "adapter").glob("*.json"))
        )
    }
