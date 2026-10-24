"""One deploy-mode live run with NaNs injected into chosen adaptation
batches, in this process: every fault must be caught and reverted.

Seed 1 only, also on the full subset.
"""

import argparse
import json

from tta_deploy.eval.live import run_live
from tta_deploy.experiments import SEEDS
from tta_deploy.methods.configs import TrtTentConfig
from tta_deploy.methods.trt import TrtTent
from tta_deploy.paths import fault_dir
from tta_deploy.runtime.adapter import AdapterConfig
from tta_deploy.tent import TentConfig

FAULT_METHOD = TrtTentConfig(
    AdapterConfig(TentConfig(16, 1), "deploy", 16),
    poisoned_batches=frozenset({5, 20, 21}),
)
FAULT_FPS = 240.0


def fault_run(subset: str) -> None:
    out = fault_dir(subset)
    method, summary = run_live(FAULT_METHOD, subset, 1, FAULT_FPS, out)
    assert isinstance(method, TrtTent)
    assert summary["faults"] == len(FAULT_METHOD.poisoned_batches), (
        "every injected NaN must be caught and reverted"
    )
    published_finite = bool(method.adapter.engine.published_buffer.isfinite().all())
    assert summary["entropies_finite"] and published_finite
    # Written only once every check passed: the report reads this file.
    (out / "fault.json").write_text(
        json.dumps(
            summary
            | {
                "poisoned_batches": sorted(FAULT_METHOD.poisoned_batches),
                "published_parameters_finite": published_finite,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--subset", choices=SEEDS, default="dev")
    fault_run(parser.parse_args().subset)


if __name__ == "__main__":
    main()
