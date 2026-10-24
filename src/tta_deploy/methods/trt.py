import numpy as np
import torch

from tta_deploy.methods.method import AdaptationProgress
from tta_deploy.runtime.adapter import Adapter


class TrtTent:
    """TensorRT with the published γ/β, every S-th frame to the
    decoupled adapter, and the lag trace per frame."""

    def __init__(self, name: str, adapter: Adapter, capacity: int) -> None:
        self.name = name
        self.adapter = adapter
        self.frame_index = 0
        self.prediction_steps = np.zeros(capacity, dtype=np.int64)
        self.accepted_batches = np.zeros(capacity, dtype=np.int64)

    def infer(self, images: torch.Tensor) -> torch.Tensor:
        i = self.frame_index
        logits = self.adapter.engine(images)
        self.prediction_steps[i] = self.adapter.engine.called_step
        self.accepted_batches[i] = self.adapter.accepted_batches
        if self.adapter.config.tent.is_adaptation_frame(i):
            self.adapter.add_adaptation_frame(images)
        # Last: the health thread reads only frames already traced.
        self.frame_index += 1
        return logits

    def settle(self) -> None:
        self.adapter.settle()
        self.frame_index = 0

    def adaptation_progress(self) -> AdaptationProgress | None:
        return self.adapter.adaptation_progress()

    def summary(self) -> dict[str, float | int | None]:
        # Counters only: the health thread reads this every second.
        n, steps = self.frame_index, self.adapter.steps
        return self.adapter.summary() | {
            "effective_stride": n / (steps * self.adapter.config.tent.B)
            if steps
            else None,
            "lag": int(self.accepted_batches[n - 1] - self.prediction_steps[n - 1])
            if n
            else 0,
        }

    def traces(self) -> dict[str, np.ndarray]:
        n, records = self.frame_index, self.adapter.step_records
        return {
            "prediction_steps": self.prediction_steps[:n],
            "accepted_batches": self.accepted_batches[:n],
            "step_lag": np.array([r.accepted_batches - r.step for r in records]),
            "step_published_s": np.array([r.published_s for r in records]),
        }

    def close(self) -> None:
        self.adapter.drain()
        self.adapter.close()
