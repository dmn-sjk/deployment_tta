"""Refit benchmark, in this process: a refittable engine with constant
γ/β, and the cost of refitting the 50 γ/β from device memory, vs the
pointer swap of the main engine (ADR 0005)."""

import argparse
import json
import time

import numpy as np
import tensorrt as trt
import torch

from tta_deploy.export.engine import LOGGER, build_engine, built_engine
from tta_deploy.export.onnx_export import ExportConfig
from tta_deploy.model import load_source_model
from tta_deploy.paths import bench_dir, refit_bench_path
from tta_deploy.runtime.trt_engine import load_cuda_engine
from tta_deploy.tent import adaptable_parameter_names, adaptable_parameters

REFIT_REPEATS = 50


def refit_bench() -> None:
    bench_dir().mkdir(parents=True, exist_ok=True)
    # Checks the constant-γ/β ONNX is the one its engine was built from.
    onnx_path = built_engine(ExportConfig(layernorm_inputs=False)).with_suffix(".onnx")
    path = bench_dir() / "refittable.engine"
    build_engine(onnx_path, path, refittable=True)
    engine = load_cuda_engine(path)
    refitter = trt.Refitter(engine, LOGGER)
    model = load_source_model()
    names = adaptable_parameter_names(model)
    params = adaptable_parameters(model)
    assert {f"model.{n}" for n in names} <= set(refitter.get_all_weights())
    stream = torch.cuda.Stream()
    host_ms, gpu_ms = [], []
    for _ in range(REFIT_REPEATS):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin = time.perf_counter()
        for name, p in zip(names, params, strict=True):
            assert refitter.set_named_weights(
                f"model.{name}",
                trt.Weights(trt.float32, p.data_ptr(), p.numel()),
                trt.TensorLocation.DEVICE,
            )
        with torch.cuda.stream(stream):
            start.record()
            assert refitter.refit_cuda_engine_async(stream.cuda_stream)
            end.record()
        host_ms.append((time.perf_counter() - begin) * 1e3)
        end.synchronize()
        gpu_ms.append(start.elapsed_time(end))
    main = built_engine(ExportConfig())
    result = {
        "refit_host_ms_median": float(np.median(host_ms[5:])),
        "refit_gpu_ms_median": float(np.median(gpu_ms[5:])),
        "refittable_engine_mb": path.stat().st_size / 1e6,
        "main_engine_mb": main.stat().st_size / 1e6,
        "main_engine_activation_mb": load_cuda_engine(main).device_memory_size_v2 / 1e6,
    }
    print(result)
    refit_bench_path().write_text(json.dumps(result, indent=2))


def main() -> None:
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args()
    refit_bench()


if __name__ == "__main__":
    main()
