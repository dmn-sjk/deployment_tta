"""One lag simulation (reference numerics, no real time) of one TENT
config, in this process: the reference loop at fixed lags, and replaying
the lag trace of the config's spec-mode max-fps live run (the median
repeat)."""

import argparse
import json
from pathlib import Path

import numpy as np

from tta_deploy.data.imagenet_c import (
    Frame,
    FrameDataset,
    stream_labels,
    verified_stream,
)
from tta_deploy.eval.metrics import per_corruption_accuracy
from tta_deploy.experiments import LAGS, add_lag_sim_args, add_stream_args
from tta_deploy.methods.configs import spec_trace_method
from tta_deploy.model import enable_determinism, load_source_model
from tta_deploy.paths import lag_sim_dir, reference_dir
from tta_deploy.reference import fixed_lag_prediction_steps, run_reference
from tta_deploy.report.load import load_held_max_fps
from tta_deploy.tent import Tent, TentConfig


def simulated(
    stream: list[Frame],
    config: TentConfig,
    name: str,
    prediction_steps: np.ndarray,
    out: Path,
) -> dict:
    labels, corruptions = stream_labels(stream)
    model = load_source_model()
    records = run_reference(
        model,
        FrameDataset(stream),
        Tent(model, config),
        prediction_steps=prediction_steps,
    )
    np.save(out / f"{name}_predictions.npy", records.predictions)
    accuracy = per_corruption_accuracy(records.predictions, labels, corruptions)
    print(f"lag sim {config.name} {name}: average {accuracy['average']:.2f}%")
    return {
        "accuracy": accuracy,
        "logits_sha256": records.logits_sha256,
        "params_sha256": records.params_sha256,
    }


def lag_sim(subset: str, seed: int, config: TentConfig, trace_only: bool) -> None:
    reference_metrics = (
        reference_dir(subset, seed, "fp32", rerun=False) / config.name / "metrics.json"
    )
    # The fixed lags check L0 against the reference.
    assert trace_only or reference_metrics.exists(), (
        f"{reference_metrics}: no reference, run reference_run.py first"
    )
    stream = verified_stream(subset, seed)
    out = lag_sim_dir(subset, seed, config)
    assert not trace_only or (out / "fixed_lags.json").exists(), (
        f"{out}: no fixed-lag results yet, run without --trace-only"
    )
    name = spec_trace_method(config).name
    # The report sets the trace against this same live run.
    live = load_held_max_fps(subset, name, seed)
    assert live is not None, f"{name}: no live run held, nothing to replay"
    trace = live.live.frames()["prediction_steps"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace.json").write_text(
        json.dumps(simulated(stream, config, "trace", trace, out), indent=2)
    )
    # Fixed lags do not depend on the live run, so a new live run only
    # needs its trace replayed.
    if trace_only:
        return
    fixed = {
        f"L{lag}": simulated(
            stream,
            config,
            f"L{lag}",
            fixed_lag_prediction_steps(len(stream), config, lag),
            out,
        )
        for lag in LAGS
    }
    matches_reference = (
        fixed["L0"]["logits_sha256"]
        == json.loads(reference_metrics.read_text())["logits_sha256"]
    )
    (out / "fixed_lags.json").write_text(
        json.dumps({**fixed, "L0_matches_reference": matches_reference}, indent=2)
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_lag_sim_args(parser)
    add_stream_args(parser)
    return parser


def main() -> None:
    args = parser().parse_args()
    enable_determinism()
    lag_sim(args.subset, args.seed, TentConfig(args.B, args.S), args.trace_only)


if __name__ == "__main__":
    main()
