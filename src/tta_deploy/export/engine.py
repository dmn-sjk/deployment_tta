import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import tensorrt as trt

from tta_deploy.export.onnx_export import MAIN_EXPORT, ExportConfig

TIMING_CACHE = Path("/data/cache/tensorrt/timing.cache")
ENGINE_DIR = Path("/data/engines")
# FP16 activations over 12 blocks: logits differ by up to ~0.1 from
# PyTorch FP32 on dev frames.
PARITY_ATOL = 0.25
# The main export built by trtexec instead of the Python API: same
# network, checked for the same parity and speed.
TRTEXEC_BUILT = f"{MAIN_EXPORT.name}_trtexec"
LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path: Path, engine_path: Path, refittable: bool = False) -> None:
    """Fixed batch-1 engine whose precisions are exactly the ONNX tensor
    types."""
    builder = trt.Builder(LOGGER)

    # Myelin ignores per-layer precision constraints and casts FP32 γ/β
    # inputs to FP16. Strongly typed networks keep the ONNX types.
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )

    parser = trt.OnnxParser(network, LOGGER)
    assert parser.parse_from_file(str(onnx_path)), [
        parser.get_error(i).desc() for i in range(parser.num_errors)
    ]

    builder_config = builder.create_builder_config()
    builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    if refittable:
        builder_config.set_flag(trt.BuilderFlag.REFIT)

    TIMING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = builder_config.create_timing_cache(
        TIMING_CACHE.read_bytes() if TIMING_CACHE.exists() else b""
    )
    builder_config.set_timing_cache(cache, ignore_mismatch=False)

    serialized = builder.build_serialized_network(network, builder_config)
    assert serialized is not None, f"engine build failed: {onnx_path}"

    # The TensorRT stubs do not mark IHostMemory as a buffer, so the
    # type checker rejects it in write_bytes. memoryview satisfies it
    # without copying the bytes.
    TIMING_CACHE.write_bytes(memoryview(builder_config.get_timing_cache().serialize()))
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(memoryview(serialized))


def engine_path(name: str) -> Path:
    return ENGINE_DIR / f"{name}.engine"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_engine_record(export: ExportConfig, parity: float) -> None:
    """Next to the engine, written last: what it was built from and its
    parity with PyTorch."""
    path = engine_path(export.name)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "export": asdict(export),
                "tensorrt": trt.__version__,
                "onnx_sha256": file_sha256(path.with_suffix(".onnx")),
                "engine_sha256": file_sha256(path),
                "max_abs_logit_difference_vs_pytorch": parity,
            },
            indent=2,
        )
    )


def built_engine(export: ExportConfig) -> Path:
    """Path of an engine built by scripts/engines.py from this export
    config with this TensorRT version.

    Runs never build engines: a build would touch the GPU of a
    measurement.
    """
    path = engine_path(export.name)
    record_path = path.with_suffix(".json")
    assert path.exists() and record_path.exists(), (
        f"{path}: not built, run scripts/engines.py"
    )
    record = json.loads(record_path.read_text())
    assert (
        record["export"] == asdict(export)
        and record["tensorrt"] == trt.__version__
        and record["engine_sha256"] == file_sha256(path)
        and record["onnx_sha256"] == file_sha256(path.with_suffix(".onnx"))
        and record["max_abs_logit_difference_vs_pytorch"] < PARITY_ATOL
    ), f"{path}: does not match its record {record}, rebuild with scripts/engines.py"
    return path
