from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@dataclass(frozen=True)
class AdaptationProgress:
    arrived_batches: int
    stepped_batches: int
    # The frame path waited for a free adaptation slot.
    blocked: bool


@runtime_checkable
class Method(Protocol):
    """A deployable way to predict the frame stream.

    The frame path calls `infer` once per frame on its own stream.
    """

    name: str

    def infer(self, images: torch.Tensor) -> torch.Tensor: ...

    # Between replays: back to source, counting from frame 0.
    def settle(self) -> None: ...

    # None when no adapter must keep up: inference only, sync TENT,
    # deploy mode.
    def adaptation_progress(self) -> AdaptationProgress | None: ...

    # Run counters, read from any thread: steps, faults, drops, lag.
    def summary(self) -> dict[str, float | int | None]: ...

    # Arrays saved with a run, per frame or per step.
    def traces(self) -> dict[str, np.ndarray]: ...

    # Finishes queued adaptation work, then stops threads.
    def close(self) -> None: ...
