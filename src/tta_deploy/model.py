import os

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 1000


def load_source_model(device: str = "cuda") -> nn.Module:
    return vit_b_16(weights=WEIGHTS).to(device).eval()


def set_fps_precision() -> None:
    # Fps runs set both TF32 flags explicitly. bf16 autocast
    # work is unaffected, FP32 ops stay strict.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def enable_determinism() -> None:
    """FP32 + deterministic settings of the reference.

    Call before any CUDA work.
    """
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "0"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
