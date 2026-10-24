"""One reference run (PyTorch, deterministic, no real time): source or
one TENT config over the whole stream of a subset, in this process.

The reported accuracy of every method at the same (B, S).
"""

import argparse
import json
import time

import numpy as np
import torch

from tta_deploy.data.imagenet_c import FrameDataset, stream_labels, verified_stream
from tta_deploy.eval.metrics import per_corruption_accuracy
from tta_deploy.experiments import (
    add_reference_args,
    add_stream_args,
    reference_from_args,
    reference_name,
)
from tta_deploy.model import enable_determinism, load_source_model
from tta_deploy.paths import reference_dir
from tta_deploy.reference import Precision, run_reference
from tta_deploy.tent import Tent, TentConfig


def reference(
    subset: str,
    seed: int,
    config: TentConfig | None,
    precision: Precision,
    rerun: bool,
) -> None:
    stream = verified_stream(subset, seed)
    labels, corruptions = stream_labels(stream)
    name = reference_name(config)
    start = time.perf_counter()
    model = load_source_model()
    records = run_reference(
        model,
        FrameDataset(stream),
        Tent(model, config) if config else None,
        precision,
    )
    offline_seconds = time.perf_counter() - start
    accuracy = per_corruption_accuracy(records.predictions, labels, corruptions)
    out = reference_dir(subset, seed, precision, rerun) / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "frames.npz",
        predictions=records.predictions,
        entropies=records.entropies,
        labels=labels,
        corruptions=corruptions,
    )
    metrics = {
        "config": name,
        "B": config.B if config else None,
        "S": config.S if config else None,
        "lr": config.lr if config else None,
        "subset": subset,
        "seed": seed,
        "precision": precision,
        "frames": len(stream),
        "steps": records.steps,
        "accuracy": accuracy,
        "logits_sha256": records.logits_sha256,
        "params_sha256": records.params_sha256,
        "offline_seconds": offline_seconds,
        "offline_peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"{name}: average {accuracy['average']:.2f}%, steps {records.steps}, {offline_seconds:.0f} s offline, logits {records.logits_sha256[:12]}"
    )


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_reference_args(parser)
    add_stream_args(parser)
    return parser


def main() -> None:
    args = parser().parse_args()
    enable_determinism()
    reference(
        args.subset, args.seed, reference_from_args(args), args.precision, args.rerun
    )


if __name__ == "__main__":
    main()
