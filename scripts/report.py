"""Report of the reference of a subset and of the dev timing runs:
writes report.md and plots, skipping sections whose runs do not exist
yet."""

import argparse

import numpy as np

from tta_deploy.experiments import SAME_RATE_FPS, SEEDS
from tta_deploy.export.engine import TRTEXEC_BUILT, engine_path
from tta_deploy.export.onnx_export import EXPORTS
from tta_deploy.methods.configs import RunConfig
from tta_deploy.paths import fault_dir, reference_set, refit_bench_path, report_dir
from tta_deploy.report.load import (
    load_adapter_bench,
    load_json,
    load_lag_sim,
    load_max_fps,
    load_realtime,
    load_reference,
    load_trtexec,
)
from tta_deploy.report.plots import (
    plot_accuracy_vs_fps,
    plot_cost_vs_knobs,
    plot_knobs,
    plot_over_time,
)
from tta_deploy.report.tables import (
    adapter_bench_table,
    average,
    collapsed_configs,
    deploy_table,
    determinism_table,
    engines_table,
    is_main,
    lag_sim_table,
    main_table,
    max_fps_table,
    precision_table,
    reference_accuracies,
    reference_name,
    reference_seeds_table,
    reference_table,
    same_rate_table,
    spec_resources_table,
    step_in_flight_table,
)

# ROID Table 1 (continual IN-C severity 5, ViT-B/16, error %): source
# and TENT at B = 64.
ROID_ACCURACY = {"source": 100 - 60.2, "tent_B64_S1": 100 - 54.5}
# Timing does not depend on the images or the seed, so every timing run
# is dev, seed 1. Only the reference follows --subset.
TIMING_SUBSET, TIMING_SEED = "dev", 1


def section(title: str, *body: str) -> list[str]:
    return ["", f"## {title}", "", *body]


def tent_points(
    runs: list[tuple[RunConfig, float]],
) -> dict[str, list[tuple[int, int, float]]]:
    """By method kind: (B, S, value) of the main TENT configs."""
    points: dict[str, list[tuple[int, int, float]]] = {}
    for config, value in runs:
        if config.tent and is_main(config):
            assert config.B is not None and config.S is not None
            points.setdefault(config.kind, []).append((config.B, config.S, value))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=SEEDS, default="dev")
    parser.add_argument(
        "--seed", type=int, default=1, help="seed of the per-run reference tables"
    )
    args = parser.parse_args()
    out = report_dir(args.subset)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    assert seed in SEEDS[args.subset], f"seed {seed} not in {args.subset} seeds"
    lines = [
        f"# Edge TTA deployment: reference on the {args.subset} subset, "
        f"timing on the {TIMING_SUBSET} subset, seed {TIMING_SEED}"
    ]
    lines += [
        "",
        f"Live-run accuracies are {TIMING_SUBSET}-stream numbers and check correctness only, never accuracy claims.",
    ]

    reference_by_seed = {
        s: runs
        for s in SEEDS[args.subset]
        if (runs := load_reference(args.subset, s, reference_set("fp32", False)))
    }
    accuracies = reference_accuracies(reference_by_seed)
    # Live accuracy is compared with the reference on the same stream.
    timing_reference = load_reference(
        TIMING_SUBSET, TIMING_SEED, reference_set("fp32", False)
    )
    realtime = [
        r for r in load_realtime(TIMING_SUBSET) if r.metrics["seed"] == TIMING_SEED
    ]
    deploy = [r for r in realtime if r.config.kind == "trt_tent_deploy"]
    same_rate = [
        r
        for r in realtime
        if r.metrics["fps"] == SAME_RATE_FPS and r.config.kind != "trt_tent_deploy"
    ]
    max_fps = load_max_fps(TIMING_SUBSET, TIMING_SEED)
    if same_rate or max_fps:
        lines += section(
            "Main table",
            f"Timing: whole {TIMING_SUBSET} stream, seed {TIMING_SEED}. Sustains {SAME_RATE_FPS:g} fps: the {SAME_RATE_FPS:g} fps live run holds ADR 0002. "
            "Max fps: the rate whose live run held. Device GB: the higher of the two runs. "
            f"Top-1 is the reference's at the method's (B, S), {args.subset} subset: mean ± half the seed range (seeds).",
            "",
            main_table(same_rate, max_fps, accuracies),
        )
    if len(reference_by_seed) > 1:
        lines += section(
            f"Reference over seeds {sorted(reference_by_seed)}",
            reference_seeds_table(accuracies),
        )

    reference = reference_by_seed.get(seed, {})
    if reference:
        collapsed = collapsed_configs(reference)
        lines += section(
            "Reference (PyTorch FP32, deterministic, offline)",
            reference_table(reference),
            "",
            f"Collapsed (below source, marked, never dropped): {', '.join(sorted(collapsed)) or 'none'}.",
            "ROID Table 1 (report-only): "
            + ", ".join(
                f"{n} {average(reference[n]):.2f} vs {a:.1f}"
                for n, a in ROID_ACCURACY.items()
                if n in reference
            )
            + ".",
        )
        if bf16 := load_reference(args.subset, seed, reference_set("bf16", False)):
            lines += section(
                "Precision delta (bf16 autocast vs FP32)",
                precision_table(reference, bf16),
            )
        if rerun := load_reference(args.subset, seed, reference_set("fp32", True)):
            text, identical = determinism_table(reference, rerun)
            lines += section(
                f"Rerun determinism: {'all bit-identical' if identical else 'MISMATCH'}",
                text,
            )
        plot_knobs(reference, out / "accuracy_vs_B_and_S.png")
        plot_over_time(reference, out / "accuracy_and_health_over_time.png")

    if same_rate and timing_reference:
        lines += section(
            f"Deployed vs reference at {SAME_RATE_FPS:g} fps (report-only)",
            f"Every method sees the same input rate, the least lag. Δ reference and agreement are against the {TIMING_SUBSET} reference at the same (B, S). "
            "A gap far above dev noise (about 0.6 pp) points to a bug.",
            "",
            same_rate_table(same_rate, timing_reference),
        )
    if deploy:
        lines += section("Deploy mode vs input fps (drop-oldest)", deploy_table(deploy))

    trtexec = load_trtexec()
    bench = load_adapter_bench()
    if max_fps and timing_reference:
        lines += section(
            "Max sustainable fps (ADR 0002)",
            f"Bisection and repeat replays on 2,000 frames. Max fps is the rate whose whole-{TIMING_SUBSET}-stream live run held, and live accuracy (report-only), lag and p99 come from that run. "
            "With repeats of the whole measurement, the run with the median max fps is reported, next to the range over repeats.",
            "",
            max_fps_table(max_fps, timing_reference),
        )
        lines += section(
            "Spec mode at max fps: frame latency with a step in flight",
            "A step is in flight from the adaptation batch's acceptance to its publish. ADR 0002 caps max fps at 1 / p99 latency.",
            "",
            step_in_flight_table(max_fps),
        )
        lines += section(
            "Spec mode at the confirmed rate (2,000 frames): resources",
            spec_resources_table(max_fps),
        )
    if max_fps and accuracies:
        points: dict[str, list[tuple[float, float, str]]] = {}
        for name, run in max_fps.items():
            config = run.live.config
            if is_main(config) and (a := accuracies.get(reference_name(run.live))):
                points.setdefault(config.kind, []).append(
                    (run.held["max_fps"], float(np.mean(a)), config.tent or name)
                )
        plot_accuracy_vs_fps(
            points, float(np.mean(accuracies["source"])), out / "accuracy_vs_fps.png"
        )
    if max_fps or same_rate:
        plot_cost_vs_knobs(
            tent_points([(r.live.config, r.held["max_fps"]) for r in max_fps.values()]),
            tent_points([(r.config, r.metrics["latency_p99_ms"]) for r in same_rate]),
            {
                r.live.config.kind: r.held["max_fps"]
                for r in max_fps.values()
                if r.live.config.tent is None and is_main(r.live.config)
            },
            out / "cost_vs_B_and_S.png",
        )
    if lag_sim := load_lag_sim(TIMING_SUBSET, TIMING_SEED):
        lines += section(
            "Lag simulator (reference numerics, dev tool)",
            lag_sim_table(lag_sim, max_fps),
        )

    records = {
        name: record
        for name in (*(e.name for e in EXPORTS), TRTEXEC_BUILT)
        if (record := load_json(engine_path(name).with_suffix(".json")))
    }
    if records:
        lines += section("Engines", engines_table(records, trtexec))
    if bench:
        lines += section(
            "Adapter step (bf16 autocast, FP32 adaptable parameters, standalone)",
            adapter_bench_table(bench),
        )
    if refit := load_json(refit_bench_path()):
        lines += section(
            "Refit alternative (microbenchmark)",
            f"Refit of the 50 γ/β from device memory: {refit['refit_host_ms_median']:.2f} ms host, {refit['refit_gpu_ms_median']:.3f} ms GPU (median). "
            f"The handoff would need two refittable engines: {2 * refit['refittable_engine_mb']:.0f} MB vs {refit['main_engine_mb']:.0f} MB for γ/β inputs, whose publish is a pointer swap.",
        )
    if fault := load_json(fault_dir(args.subset) / "fault.json"):
        lines += section(
            "Fault handling",
            f"{fault['method']} at {fault['fps']:g} fps with NaNs injected: {fault['faults']} faults caught and reverted, "
            f"predictions finite: {fault['entropies_finite']}, published γ/β finite: {fault['published_parameters_finite']}, accuracy {fault['accuracy']['average']:.2f}%.",
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
