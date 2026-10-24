import numpy as np
import torch
from torch import nn

from tta_deploy.tent import Tent


class PytorchTent:
    """Eager PyTorch, TENT in the frame path in the gradient-
    accumulation form, so an adaptation frame pays its own backward."""

    def __init__(self, name: str, model: nn.Module, tent: Tent) -> None:
        self.name = name
        self.model = model
        self.tent = tent
        self.frame_index = 0
        self.steps = 0

    def infer(self, images: torch.Tensor) -> torch.Tensor:
        i = self.frame_index
        self.frame_index += 1
        if not self.tent.config.is_adaptation_frame(i):
            with torch.no_grad():
                return self.model(images)
        # The frame path runs under inference mode, whose tensors cannot
        # enter autograd.
        with torch.inference_mode(False), torch.enable_grad():
            logits = self.model(images.clone()).float()
            self.steps += self.tent.adapt(logits)
        return logits.detach()

    def settle(self) -> None:
        self.tent.reset_to_source()
        # The frame stream does not wait on the current stream, so the
        # reset must land before the next frame.
        torch.cuda.current_stream().synchronize()
        self.frame_index = self.steps = 0

    def adaptation_progress(self) -> None:
        # The step runs inside the frame path, so the p99 bound already
        # covers it.
        return None

    def summary(self) -> dict[str, float | int | None]:
        return {"steps": self.steps}

    def traces(self) -> dict[str, np.ndarray]:
        return {}

    def close(self) -> None:
        pass
