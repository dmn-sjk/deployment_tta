from collections.abc import Callable

import numpy as np
import torch


class Source:
    """The frozen model behind any forward: eager PyTorch or a TensorRT
    engine with the source γ/β."""

    def __init__(
        self, name: str, forward: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        self.name = name
        self.forward = forward

    def infer(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(images)

    def settle(self) -> None:
        pass

    def adaptation_progress(self) -> None:
        return None

    def summary(self) -> dict[str, float | int | None]:
        return {}

    def traces(self) -> dict[str, np.ndarray]:
        return {}

    def close(self) -> None:
        pass
