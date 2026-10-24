import torch

from tta_deploy.methods.method import AdaptationProgress, Method
from tta_deploy.model import IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES


def normalization_constants() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(IMAGENET_MEAN, device="cuda").view(3, 1, 1),
        torch.tensor(IMAGENET_STD, device="cuda").view(3, 1, 1),
    )


def preprocess_on_device(
    frame: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Pinned uint8 HWC host frame -> normalized FP32 NCHW batch of 1 on
    the current stream, same arithmetic as the CPU preprocess."""
    assert (
        frame.dtype == torch.uint8
        and frame.shape == (224, 224, 3)
        and frame.is_pinned()
    )
    images = frame.to("cuda", non_blocking=True).permute(2, 0, 1).float().div_(255)
    return images.sub_(mean).div_(std)[None].contiguous()


class FramePath:
    """One frame at a time through a method: decoded host frame -> GPU
    preprocess -> method -> logits in host memory."""

    def __init__(self, method: Method):
        self.method = method
        # Highest priority: the adapter's kernels queue behind the frame
        # path's (PyTorch clamps to the device's range).
        self.stream = torch.cuda.Stream(priority=-100)
        self.mean, self.std = normalization_constants()
        # A decoded frame lands in pageable memory: the frame path pays
        # the copy into pinned memory, as a camera loop would.
        self.staging = torch.empty(224, 224, 3, dtype=torch.uint8, pin_memory=True)
        self.logits = torch.empty(1, NUM_CLASSES, pin_memory=True)
        self.span_start = torch.cuda.Event(enable_timing=True)
        self.span_end = torch.cuda.Event(enable_timing=True)

    @torch.inference_mode()
    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        """Returns the frame path's own pinned logits buffer, valid
        until the next call."""
        assert frame.dtype == self.staging.dtype and frame.shape == self.staging.shape
        # Safe to overwrite: the previous frame's H2D copy finished
        # before its call returned.
        self.staging.copy_(frame)
        with torch.cuda.stream(self.stream):
            self.span_start.record()
            logits = self.method.infer(
                preprocess_on_device(self.staging, self.mean, self.std)
            )
            self.logits.copy_(logits, non_blocking=True)
            self.span_end.record()
        # Make the CPU wait for the work on frame path stream to end.
        self.span_end.synchronize()
        return self.logits

    def event_span_ms(self) -> float:
        """Event span of the last frame on the stream.

        From H2D copy to D2H copy. Not GPU busy time: it includes gaps
        where the GPU waits for launches.
        """
        return self.span_start.elapsed_time(self.span_end)

    def settle(self) -> None:
        self.method.settle()

    def adaptation_progress(self) -> AdaptationProgress | None:
        return self.method.adaptation_progress()
