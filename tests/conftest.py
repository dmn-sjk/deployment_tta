import copy
import gc
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from tta_deploy.data.imagenet_c import (
    DEV_MANIFEST,
    continual_stream,
    per_corruption_sample,
    read_manifest,
)
from tta_deploy.export.engine import build_engine
from tta_deploy.export.onnx_export import ExportConfig, export_onnx
from tta_deploy.model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    enable_determinism,
    load_source_model,
)

enable_determinism()


class NoiseFrames(Dataset):
    def __init__(self, num_frames: int):
        self.num_frames = num_frames

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.randn(
            3, 224, 224, generator=torch.Generator().manual_seed(index)
        ), 0


class Uint8NoiseFrames(Dataset):
    """Decoded uint8 HWC frames, and with preprocess the reference's
    CPU-preprocessed images of the same frames."""

    def __init__(self, num_frames: int, preprocess: bool):
        self.num_frames, self.preprocess = num_frames, preprocess

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        frame = torch.randint(
            0,
            256,
            (224, 224, 3),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(index),
        )
        if self.preprocess:
            return F.normalize(
                frame.permute(2, 0, 1).float() / 255,
                IMAGENET_MEAN,
                IMAGENET_STD,
            ), 0
        return frame, 0


@pytest.fixture(scope="session")
def _source_model() -> torch.nn.Module:
    return load_source_model()


@pytest.fixture
def source_model(_source_model: torch.nn.Module) -> Iterator[torch.nn.Module]:
    yield copy.deepcopy(_source_model)
    # The B = 64 reference needs 7.4 GB of the 10. Models of earlier
    # tests linger in reference cycles, and cuBLAS keeps a workspace for
    # every stream a test created (adapter, graph capture).
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()


@pytest.fixture
def non_default_stream():
    # enqueueV3 on the default stream adds a cudaStreamSynchronize per
    # call.
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        yield
    torch.cuda.synchronize()


@pytest.fixture(scope="session")
def engines(
    _source_model: nn.Module, tmp_path_factory: pytest.TempPathFactory
) -> dict[ExportConfig, Path]:
    out = tmp_path_factory.mktemp("engines")
    paths = {}
    for config in (
        ExportConfig(),
        ExportConfig(layernorm_fp32=False),
        ExportConfig(layernorm_inputs=False),
    ):
        export_onnx(_source_model, out / f"{config.name}.onnx", config)
        build_engine(out / f"{config.name}.onnx", out / f"{config.name}.engine")
        paths[config] = out / f"{config.name}.engine"
    return paths


@pytest.fixture(scope="session")
def dev_stream_sample() -> list:
    return per_corruption_sample(
        continual_stream(read_manifest(DEV_MANIFEST), seed=1), 2
    )


def ready() -> torch.cuda.Event:
    event = torch.cuda.Event()
    event.record()
    return event
