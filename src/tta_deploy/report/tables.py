import numpy as np

from tta_deploy.data.imagenet_c import CORRUPTIONS
from tta_deploy.experiments import LAGS
from tta_deploy.export.onnx_export import MAIN_EXPORT
from tta_deploy.methods.configs import KINDS, RunConfig
from tta_deploy.report.load import MaxFpsRun, Run

SHORT = {c: c.split("_")[0][:5] for c in CORRUPTIONS}
assert len(set(SHORT.values())) == len(CORRUPTIONS), (
    f"ambiguous corruption abbreviations: {SHORT}"
)


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return "\n".join([*lines, *("| " + " | ".join(r) + " |" for r in rows)])


def order(runs: dict[str, Run]) -> list[str]:
    """Reference config names: source first, then by S and B."""
    return ["source"] + sorted(
        (n for n in runs if n != "source"),
        key=lambda n: (runs[n].metrics["S"], runs[n].metrics["B"]),
    )


def average(run: Run) -> float:
    return run.metrics["accuracy"]["average"]


def reference_name(run: Run) -> str:
    """The reference config a live run is compared with."""
    return run.config.tent or "source"


def is_main(config: RunConfig) -> bool:
    """Of the main table: one config per method and (B, S), no engine or
    adapter variants."""
    match config.kind:
        case "trt_source":
            return config.export == MAIN_EXPORT.name
        case "trt_tent_spec":
            return config.gated
        case "trt_tent_deploy":
            return False
    return True


def reference_accuracies(
    reference_by_seed: dict[int, dict[str, Run]],
) -> dict[str, list[float]]:
    """By reference config, its average accuracy per seed."""
    accuracies: dict[str, list[float]] = {}
    for runs in reference_by_seed.values():
        for name, run in runs.items():
            accuracies.setdefault(name, []).append(average(run))
    return accuracies


def gated_run(runs: dict[str, MaxFpsRun], tent: str) -> MaxFpsRun | None:
    """The spec run gated against the reference config `tent`."""
    gated = [
        r for r in runs.values() if r.live.config.gated and r.live.config.tent == tent
    ]
    assert len(gated) <= 1, f"{tent}: more than one gated run"
    return gated[0] if gated else None


def cell(value: float | int | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def collapsed_configs(reference: dict[str, Run]) -> set[str]:
    source = average(reference["source"])
    return {n for n in reference if n != "source" and average(reference[n]) < source}


def reference_table(reference: dict[str, Run]) -> str:
    source, collapsed = average(reference["source"]), collapsed_configs(reference)
    rows = []
    for name in order(reference):
        m, acc = reference[name].metrics, reference[name].metrics["accuracy"]
        rows.append(
            [
                name,
                str(m["steps"]),
                *(f"{acc[c]:.1f}" for c in CORRUPTIONS),
                f"**{acc['average']:.2f}**",
                f"{acc['average'] - source:+.2f}",
                "" if name == "source" else ("yes" if name in collapsed else "no"),
                f"{m['offline_seconds']:.0f}",
                f"{m['offline_peak_gpu_memory_gb']:.1f}",
            ]
        )
    header = ["config", "steps", *(SHORT[c] for c in CORRUPTIONS)]
    header += ["avg", "Δ source", "collapsed", "offline s", "offline peak GB"]
    return table(header, rows)


def precision_table(fp32: dict[str, Run], bf16: dict[str, Run]) -> str:
    rows = [
        [n, f"{average(fp32[n]):.2f}", f"{average(bf16[n]):.2f}"]
        + [f"{average(bf16[n]) - average(fp32[n]):+.2f}"]
        for n in order(bf16)
    ]
    return table(["config", "fp32 avg", "bf16 avg", "Δ"], rows)


def determinism_table(runs: dict[str, Run], reruns: dict[str, Run]) -> tuple[str, bool]:
    rows, all_identical = [], True
    for name in order(reruns):
        a, b = runs[name].metrics, reruns[name].metrics
        identical = (a["logits_sha256"], a["params_sha256"]) == (
            b["logits_sha256"],
            b["params_sha256"],
        )
        all_identical &= identical
        rows.append(
            [
                name,
                f"`{a['logits_sha256'][:16]}`",
                f"`{a['params_sha256'][:16] or '-'}`",
                "yes" if identical else "**NO**",
            ]
        )
    header = ["config", "logits sha256", "params sha256", "bit-identical"]
    return table(header, rows), all_identical


def agreement(run: Run, reference: Run) -> float:
    return float(
        (run.frames()["predictions"] == reference.frames()["predictions"]).mean() * 100
    )


def same_rate_table(runs: list[Run], reference: dict[str, Run]) -> str:
    """Every method at the same input rate: accuracy against the
    reference at the same (B, S), latency, adaptation counters,
    resources."""
    source = average(reference["source"])
    rows = []
    for run in sorted(runs, key=lambda r: r.metrics["method"]):
        m = run.metrics
        ref = reference.get(reference_name(run))
        rows.append(
            [
                m["method"],
                f"{average(run):.2f}",
                f"{average(run) - source:+.2f}",
                f"{average(run) - average(ref):+.2f}" if ref else "-",
                f"{agreement(run, ref):.2f}" if ref else "-",
                *(cell(m[f"latency_{p}_ms"]) for p in ("p50", "p99", "p999")),
                cell(m["service_p50_ms"]),
                str(m["sustainable"]),
                str(m.get("steps", "-")),
                cell(m.get("lag_mean")),
                str(m.get("lag_max", "-")),
                cell(m["device_memory_used_gb"]),
                cell(m["power_mean_w"], 0),
            ]
        )
    header = ["method", "accuracy", "Δ source", "Δ reference", "agreement %"]
    header += ["p50 ms", "p99 ms", "p99.9 ms", "service p50 ms", "sustainable"]
    header += ["steps", "lag mean", "lag max", "device GB", "power W"]
    return table(header, rows)


def deploy_table(runs: list[Run]) -> str:
    rows = [
        [
            r.metrics["method"],
            f"{r.metrics['fps']:g}",
            f"{average(r):.2f}",
            cell(r.metrics["effective_stride"]),
            cell(r.metrics["drop_rate"], 3),
            cell(r.metrics["lag_mean"]),
            cell(r.metrics["latency_p99_ms"]),
            str(r.metrics["sustainable"]),
        ]
        for r in sorted(runs, key=lambda r: (r.metrics["method"], r.metrics["fps"]))
    ]
    header = ["method", "input fps", "accuracy", "effective stride", "drop rate"]
    return table(header + ["lag mean", "p99 ms", "sustainable"], rows)


def repeat_range(run: MaxFpsRun) -> str:
    fps = run.repeat_fps
    return f"{fps[0]:.0f} to {fps[-1]:.0f} ({len(fps)})" if len(fps) > 1 else "-"


def max_fps_table(runs: dict[str, MaxFpsRun], reference: dict[str, Run]) -> str:
    rows = []
    for name in sorted(runs):
        run, h, live = runs[name], runs[name].held, runs[name].live.metrics
        ref = reference.get(reference_name(run.live))
        batches = (
            f"{live['arrived_batches']} / {live['stepped_batches']}"
            if live["arrived_batches"] is not None
            else "-"
        )
        rows.append(
            [
                name,
                f"{average(run.live):.2f}",
                f"{average(run.live) - average(ref):+.2f}" if ref else "-",
                f"{h['bisected_fps']:.1f}",
                f"{h['confirmed_fps']:.1f}",
                str(h["confirm_step_downs"]),
                f"**{h['max_fps']:.1f}**",
                repeat_range(run),
                str(h["live_step_downs"]),
                cell(live["latency_p99_ms"]),
                cell(live["interval_ms"]),
                batches,
                cell(live.get("lag_mean")),
            ]
        )
    header = ["method", "live accuracy", "Δ reference", "bisected fps"]
    header += ["after repeats", "repeat step-downs", "max fps", "range (repeats)"]
    header += ["live step-downs", "live p99 ms", "interval ms", "arrived / stepped"]
    return table(header + ["lag mean"], rows)


def step_in_flight_latency(
    frames: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Frame latencies while a step is in flight (an accepted batch not
    yet published), and while none is."""
    in_flight = frames["accepted_batches"] > frames["prediction_steps"]
    return frames["latency_ms"][in_flight], frames["latency_ms"][~in_flight]


def step_in_flight_table(runs: dict[str, MaxFpsRun]) -> str:
    rows = []
    for name in sorted(runs):
        run = runs[name]
        if run.live.config.kind != "trt_tent_spec":
            continue
        in_flight, idle = step_in_flight_latency(run.live.frames())
        share = in_flight.size / (in_flight.size + idle.size)
        rows.append(
            [
                name,
                f"{run.held['max_fps']:.0f}",
                cell(run.live.metrics["interval_ms"]),
                f"{share * 100:.0f}",
                *(cell(np.percentile(in_flight, p)) for p in (50, 99)),
                *(cell(np.percentile(idle, p)) if idle.size else "-" for p in (50, 99)),
            ]
        )
    header = ["method", "max fps", "interval ms", "frames with a step in flight %"]
    header += ["p50 ms in flight", "p99 ms in flight", "p50 ms none", "p99 ms none"]
    return table(header, rows)


def spec_resources_table(runs: dict[str, MaxFpsRun]) -> str:
    rows = []
    for name in sorted(runs):
        if runs[name].live.config.kind != "trt_tent_spec":
            continue
        b = runs[name].bisection
        rows.append(
            [
                name,
                f"{b['max_fps']:.0f}",
                cell(b["confirm"]["latency_p99_ms"]),
                cell(b.get("steps_per_s"), 1),
                f"{b['confirm']['arrived_batches']} / {b['confirm']['stepped_batches']}",
                cell(b["gpu_utilisation_mean_pct"], 0),
                f"{b['power_mean_w']:.0f} / {b['power_max_w']:.0f}",
                cell(b["device_memory_used_gb"]),
                cell(b["pytorch_peak_gb"]),
            ]
        )
    header = ["method", "max fps", "p99 ms", "steps/s", "arrived / stepped"]
    header += ["GPU util %", "power W (mean / max)", "device GB", "PyTorch peak GB"]
    return table(header, rows)


def lag_sim_table(lag_sim: dict[str, dict], live: dict[str, MaxFpsRun]) -> str:
    rows = []
    for name, sim in sorted(lag_sim.items()):
        spec = gated_run(live, name)
        rows.append(
            [
                name,
                *(f"{sim[f'L{lag}']['accuracy']['average']:.2f}" for lag in LAGS),
                f"{sim['trace']['accuracy']['average']:.2f}",
                f"{average(spec.live) - sim['trace']['accuracy']['average']:+.2f}"
                if spec
                else "-",
                str(sim["L0_matches_reference"]),
            ]
        )
    header = ["config", *(f"L={lag}" for lag in LAGS), "trace"]
    return table(header + ["live − trace", "L=0 = reference"], rows)


def engines_table(records: dict[str, dict], trtexec: dict) -> str:
    rows = []
    for name, record in records.items():
        t = trtexec.get(name, {})
        rows.append(
            [
                name,
                cell(t.get("gpu_median_ms"), 3),
                cell(t.get("gpu_p99_ms"), 3),
                cell(t.get("engine_mb"), 0),
                cell(t.get("activation_mb"), 0),
                cell(record["max_abs_logit_difference_vs_pytorch"], 3),
            ]
        )
    header = ["engine", "trtexec GPU median ms", "trtexec GPU p99 ms", "engine MB"]
    return table(header + ["activations MB", "max |Δlogits| vs PyTorch"], rows)


def adapter_bench_table(bench: dict[tuple[int, int, bool], dict]) -> str:
    rows = []
    for B, m in sorted({(B, m) for B, m, _ in bench}):
        e, g = bench.get((B, m, False), {}), bench.get((B, m, True), {})
        rows.append(
            [
                str(B),
                str(m),
                cell(e.get("step_ms")),
                cell(g.get("step_ms")),
                f"{e['step_ms'] / g['step_ms']:.2f}×" if e and g else "-",
                cell(e.get("pytorch_peak_gb")),
                cell(g.get("pytorch_peak_gb")),
            ]
        )
    header = ["B", "m", "eager ms", "graph ms", "graph speedup", "eager peak GB"]
    return table(header + ["graph peak GB"], rows)


def reference_seeds_table(accuracies: dict[str, list[float]]) -> str:
    rows = [
        [
            name,
            str(len(a)),
            f"**{np.mean(a):.2f}**",
            f"{min(a):.2f}",
            f"{max(a):.2f}",
            f"{max(a) - min(a):.2f}",
        ]
        for name, a in sorted(accuracies.items())
    ]
    return table(["config", "seeds", "mean", "min", "max", "range pp"], rows)


def accuracy_cell(accuracies: list[float] | None) -> str:
    """Mean ± half the seed range, and the number of seeds."""
    if not accuracies:
        return "-"
    half_range = (max(accuracies) - min(accuracies)) / 2
    return f"{np.mean(accuracies):.2f} ± {half_range:.2f} ({len(accuracies)})"


def main_table(
    at_30_fps: list[Run],
    max_fps: dict[str, MaxFpsRun],
    accuracies: dict[str, list[float]],
) -> str:
    """Per main method config: the 30 fps live run, the max-fps live run
    that held, and the reference accuracy at its (B, S)."""
    at_30 = {r.metrics["method"]: r for r in at_30_fps if is_main(r.config)}
    held = {n: r for n, r in max_fps.items() if is_main(r.live.config)}

    def config(name: str) -> RunConfig:
        return (at_30[name] if name in at_30 else held[name].live).config

    def sort_key(name: str) -> tuple[int, int, int]:
        c = config(name)
        return KINDS.index(c.kind), c.S or 0, c.B or 0

    rows = []
    for name in sorted(at_30.keys() | held.keys(), key=sort_key):
        c, m30 = config(name), at_30[name].metrics if name in at_30 else None
        live = held[name].live.metrics if name in held else None
        sustains = "-" if m30 is None else ("yes" if m30["sustainable"] else "**no**")
        rows.append(
            [
                name,
                str(c.B or "-"),
                str(c.S or "-"),
                sustains,
                "-"
                if m30 is None
                else f"{cell(m30['latency_p50_ms'])} / {cell(m30['latency_p99_ms'])}",
                f"**{held[name].held['max_fps']:.0f}**"
                + (f" ({r})" if (r := repeat_range(held[name])) != "-" else "")
                if name in held
                else "-",
                cell(live["latency_p99_ms"]) if live else "-",
                cell(max(m["device_memory_used_gb"] for m in (m30, live) if m)),
                accuracy_cell(accuracies.get(c.tent or "source")),
            ]
        )
    header = ["method", "B", "S", "sustains 30 fps", "p50 / p99 ms at 30 fps"]
    header += ["max fps (range, repeats)", "p99 ms at max fps", "device GB"]
    return table(header + ["top-1 reference (seeds)"], rows)
