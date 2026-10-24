"""One max-fps bisection (ADR 0002): bisection and repeat replays of one
method on the first 2,000 dev frames, in this process.

Writes the confirmed rate to bisection.json, the start of the whole-
stream live runs that `orchestrate.py` runs next.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tta_deploy.data.imagenet_c import decoded_frames, verified_stream
from tta_deploy.eval.gpu import device_memory_gb, gpu_sampler
from tta_deploy.eval.live import consume_prediction, method_summary
from tta_deploy.eval.replay import (
    REPLAY_FRAMES,
    REPLAY_WARMUP,
    max_sustainable_fps,
    replay,
)
from tta_deploy.experiments import add_stream_args
from tta_deploy.methods.configs import (
    MethodConfig,
    add_method_args,
    build_method,
    max_fps_method_from_args,
)
from tta_deploy.methods.frame_path import FramePath
from tta_deploy.model import set_fps_precision


def bisect(config: MethodConfig, seed: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    set_fps_precision()
    # Bisection always replays dev frames: the rate is a property of
    # the method, the live run then checks it on the subset.
    frames = decoded_frames(verified_stream("dev", seed)[:REPLAY_FRAMES])
    method = build_method(config, REPLAY_FRAMES + REPLAY_WARMUP)
    frame_path = FramePath(method)
    found = max_sustainable_fps(frame_path, frames, REPLAY_WARMUP, consume_prediction)
    # A last replay at max fps, so the method's counters and nvidia-smi
    # samples belong to it.
    with gpu_sampler() as gpu:
        confirm = replay(
            frame_path, frames, found.fps, REPLAY_WARMUP, consume_prediction
        )
    summary = method_summary(method, len(frames) / found.fps)
    method.close()
    result = found.summary() | {
        "confirm": confirm.summary(),
        **summary,
        **gpu,
        **device_memory_gb(),
    }
    print(
        f"{config.name}: max sustainable fps {found.fps:.1f} (bisected {found.bisected_fps:.1f}), p99 {found.result.p99_ms:.2f} ms, service p50 {np.median(found.result.service_ms):.2f} ms"
    )
    (out / "bisection.json").write_text(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_method_args(parser)
    add_stream_args(parser)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = parser().parse_args()
    bisect(max_fps_method_from_args(args), args.seed, args.out)


if __name__ == "__main__":
    main()
