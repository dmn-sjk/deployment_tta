"""One engine's GPU time (trtexec) and sizes, in this process."""

import argparse
import json
import re
import subprocess

from tta_deploy.experiments import add_trtexec_bench_args
from tta_deploy.export.engine import built_engine, engine_path
from tta_deploy.export.onnx_export import EXPORTS_BY_NAME
from tta_deploy.paths import trtexec_bench_path
from tta_deploy.runtime.trt_engine import load_cuda_engine

TRTEXEC_STAT = re.compile(
    r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms.*percentile\(99%\) = ([\d.]+) ms"
)


def trtexec_gpu_ms(name: str) -> dict[str, float]:
    out = subprocess.run(
        [
            "trtexec",
            f"--loadEngine={engine_path(name)}",
            "--useSpinWait",
            "--noDataTransfers",
            "--warmUp=2000",
            "--duration=10",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    match = TRTEXEC_STAT.search(out)
    assert match, out[-2000:]
    return {
        "gpu_median_ms": float(match[4]),
        "gpu_p99_ms": float(match[5]),
        "gpu_mean_ms": float(match[3]),
    }


def trtexec_bench(engine: str) -> None:
    path = (
        built_engine(EXPORTS_BY_NAME[engine])
        if engine in EXPORTS_BY_NAME
        else engine_path(engine)
    )
    result = trtexec_gpu_ms(engine) | {
        "engine_mb": path.stat().st_size / 1e6,
        "activation_mb": load_cuda_engine(path).device_memory_size_v2 / 1e6,
    }
    print(engine, result)
    out = trtexec_bench_path(engine)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_trtexec_bench_args(parser)
    return parser


def main() -> None:
    trtexec_bench(parser().parse_args().engine)


if __name__ == "__main__":
    main()
