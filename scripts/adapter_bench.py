"""One standalone adapter step benchmark, in this process: only B, the
micro-batch m and the CUDA graph matter."""

import argparse
import json
import time

import torch

from tta_deploy.eval.gpu import device_memory_gb
from tta_deploy.experiments import (
    AdapterBench,
    adapter_bench_from_args,
    add_adapter_bench_args,
)
from tta_deploy.model import load_source_model, set_fps_precision
from tta_deploy.paths import adapter_bench_path
from tta_deploy.runtime.adapter_step import AdapterStep
from tta_deploy.tent import TentConfig

ADAPTER_BENCH_STEPS = 20


def adapter_bench(bench: AdapterBench) -> None:
    set_fps_precision()
    # S never reaches a standalone step.
    step = AdapterStep(
        load_source_model(),
        TentConfig(bench.B, 1),
        bench.micro_batch,
        bf16=True,
        graph=bench.graph,
    )
    step.images.normal_()
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ADAPTER_BENCH_STEPS):
        step()
    torch.cuda.synchronize()
    result = {
        "B": bench.B,
        "micro_batch": bench.micro_batch,
        "graph": bench.graph,
        "step_ms": (time.perf_counter() - start) / ADAPTER_BENCH_STEPS * 1e3,
        **device_memory_gb(),
    }
    print(result)
    out = adapter_bench_path(bench.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_adapter_bench_args(parser)
    return parser


def main() -> None:
    adapter_bench(adapter_bench_from_args(parser().parse_args()))


if __name__ == "__main__":
    main()
