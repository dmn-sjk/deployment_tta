"""Every run of an experiment file, each in its own process, for each
seed of the subset.

realtime: one live run (`run.py`) per method and input rate. max_fps:
per method, a bisection (`max_fps_bisect.py`), then whole-stream live
runs from the confirmed rate down until one holds (ADR 0002). reference:
one reference run (`reference_run.py`) per config. lag_sim: one lag
simulation (`lag_sim.py`) per TENT config. trtexec, adapter: one
benchmark (`trtexec_bench.py`, `adapter_bench.py`) per engine or adapter
step, once whatever the subset.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tta_deploy.eval.replay import live_step_down
from tta_deploy.eval.workers import run_script
from tta_deploy.experiments import EXPERIMENT_KINDS, SEEDS, read_experiment
from tta_deploy.paths import max_fps_dir

SCRIPTS = Path(__file__).parent


def max_fps(
    name: str, subset: str, seed: int, repeat: int, stream_argv: tuple[str, ...]
) -> None:
    out = max_fps_dir(subset, name, seed, repeat)
    run_script(SCRIPTS / "max_fps_bisect.py", *stream_argv, "--out", str(out))
    # A step-down that stops partway must not leave an older max fps
    # next to the failed run's outputs.
    (out / "max_fps.json").unlink(missing_ok=True)

    def live_run(fps: float) -> dict:
        run_script(
            SCRIPTS / "run.py", *stream_argv, "--fps", str(fps), "--out", str(out)
        )
        return json.loads((out / "metrics.json").read_text())

    bisection = json.loads((out / "bisection.json").read_text())
    held = {
        "bisected_fps": bisection["bisected_fps"],
        "confirmed_fps": bisection["max_fps"],
        "confirm_step_downs": bisection["confirm_step_downs"],
        **asdict(live_step_down(bisection["max_fps"], live_run)),
    }
    (out / "max_fps.json").write_text(json.dumps(held, indent=2))
    print(
        f"{name}: bisected {held['bisected_fps']:.1f}, confirmed {held['confirmed_fps']:.1f}, live run held at {held['max_fps']:.1f} fps after {held['live_step_downs']} step-downs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("experiment", type=Path, help="an experiments/*.yaml file")
    parser.add_argument("--subset", choices=SEEDS, default="dev")
    args = parser.parse_args()
    experiment, runs = read_experiment(args.experiment)
    kind = EXPERIMENT_KINDS[experiment]
    for run in runs:
        if not kind.per_seed:
            run_script(SCRIPTS / kind.script, *run.argv)
            continue
        for seed in SEEDS[args.subset]:
            stream_argv = (*run.argv, "--subset", args.subset, "--seed", str(seed))
            if experiment == "max_fps":
                for repeat in range(1, run.repeats + 1):
                    max_fps(run.name, args.subset, seed, repeat, stream_argv)
            else:
                run_script(SCRIPTS / kind.script, *stream_argv)


if __name__ == "__main__":
    main()
