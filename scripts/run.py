"""One live run: the whole stream of a subset through one method at a
fixed input rate, in this process."""

import argparse
from pathlib import Path

from tta_deploy.eval.live import run_live
from tta_deploy.experiments import add_realtime_args, add_stream_args
from tta_deploy.methods.configs import method_from_args
from tta_deploy.paths import realtime_dir


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_realtime_args(parser)
    add_stream_args(parser)
    parser.add_argument("--out", type=Path, help="default: the real-time run dir")
    return parser


def main() -> None:
    args = parser().parse_args()
    config = method_from_args(args)
    out = args.out or realtime_dir(args.subset, config.name, args.fps, args.seed)
    run_live(config, args.subset, args.seed, args.fps, out)


if __name__ == "__main__":
    main()
