"""Build every TensorRT engine the experiments use, each with a record
of what it was built from and its parity with PyTorch FP32.
"""

import json
import subprocess

import torch
from torch.utils.data import DataLoader

from tta_deploy.data.imagenet_c import (
    DEV_MANIFEST,
    FrameDataset,
    continual_stream,
    per_corruption_sample,
    read_manifest,
)
from tta_deploy.export.engine import (
    PARITY_ATOL,
    TIMING_CACHE,
    TRTEXEC_BUILT,
    build_engine,
    engine_path,
    write_engine_record,
)
from tta_deploy.export.onnx_export import (
    EXPORTS,
    MAIN_EXPORT,
    ExportConfig,
    export_onnx,
)
from tta_deploy.model import load_source_model
from tta_deploy.runtime.trt_engine import TrtEngine, engine_logits
from tta_deploy.tent import adaptable_parameter_names, packed_adaptable_parameters

PARITY_FRAMES_PER_CORRUPTION = 2
# Parity sample only, not an evaluation seed.
PARITY_SEED = 1


def max_abs_logit_difference(
    path_name: str, export: ExportConfig, model: torch.nn.Module, images: torch.Tensor
) -> float:
    buffers = [packed_adaptable_parameters(model)] if export.layernorm_inputs else []
    engine = TrtEngine(
        engine_path(path_name), adaptable_parameter_names(model), buffers
    )

    # enqueueV3 on the default stream adds a cudaStreamSynchronize per
    # call.
    with torch.no_grad(), torch.cuda.stream(torch.cuda.Stream()):
        return (engine_logits(engine, images) - model(images)).abs().max().item()


def main() -> None:
    model = load_source_model()

    sample = per_corruption_sample(
        continual_stream(read_manifest(DEV_MANIFEST), PARITY_SEED),
        PARITY_FRAMES_PER_CORRUPTION,
    )
    images = torch.stack(
        [image for image, _ in DataLoader(FrameDataset(sample), batch_size=None)]
    ).cuda()

    for export in EXPORTS:
        onnx_path = engine_path(export.name).with_suffix(".onnx")
        # remove the potential previous engine record
        engine_path(export.name).with_suffix(".json").unlink(missing_ok=True)

        export_onnx(model, onnx_path, export)
        build_engine(onnx_path, engine_path(export.name))

        difference = max_abs_logit_difference(export.name, export, model, images)
        assert difference < PARITY_ATOL, (
            f"{export.name}: max |Δlogits| {difference:.3f} vs PyTorch FP32"
        )

        write_engine_record(export, difference)
        print(f"{export.name}: max |Δlogits| vs PyTorch FP32 {difference:.3f}")

    # Same network through trtexec: both build routes must give the same
    # parity with PyTorch (and, in trtexec_bench.py, the same speed).
    engine_path(TRTEXEC_BUILT).with_suffix(".json").unlink(missing_ok=True)
    subprocess.run(
        [
            "trtexec",
            f"--onnx={engine_path(MAIN_EXPORT.name).with_suffix('.onnx')}",
            "--stronglyTyped",
            f"--saveEngine={engine_path(TRTEXEC_BUILT)}",
            f"--timingCacheFile={TIMING_CACHE}",
            "--skipInference",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    difference = max_abs_logit_difference(TRTEXEC_BUILT, MAIN_EXPORT, model, images)
    assert difference < PARITY_ATOL, (
        f"{TRTEXEC_BUILT}: max |Δlogits| {difference:.3f} vs PyTorch FP32"
    )
    engine_path(TRTEXEC_BUILT).with_suffix(".json").write_text(
        json.dumps({"max_abs_logit_difference_vs_pytorch": difference}, indent=2)
    )
    print(f"{TRTEXEC_BUILT}: max |Δlogits| vs PyTorch FP32 {difference:.3f}")


if __name__ == "__main__":
    main()
