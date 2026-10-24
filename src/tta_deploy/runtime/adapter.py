import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from tta_deploy.methods.method import AdaptationProgress
from tta_deploy.runtime.adapter_step import AdapterStep
from tta_deploy.runtime.trt_engine import BackBuffer, TrtEngine
from tta_deploy.tent import TentConfig

log = logging.getLogger(__name__)
# Spec mode: past this many adaptation batches the frame path blocks,
# which ADR 0002 never counts as sustainable.
SPEC_SLOTS = 4
# Deploy mode drops the oldest queued batch, so a filling, a queued and
# a draining slot never block.
DEPLOY_SLOTS = 3


@dataclass(frozen=True)
class AdapterConfig:
    tent: TentConfig
    mode: Literal["spec", "deploy"]
    micro_batch: int
    graph: bool = True

    def __post_init__(self) -> None:
        assert self.micro_batch >= 1 and self.tent.B % self.micro_batch == 0, (
            f"B {self.tent.B} must be a multiple of the micro-batch {self.micro_batch}"
        )

    @property
    def name(self) -> str:
        return f"{self.tent.name}_{self.mode}_m{self.micro_batch}_{'graph' if self.graph else 'eager'}"


@dataclass(frozen=True)
class PendingBatch:
    slot: int
    ready: torch.cuda.Event


@dataclass(frozen=True)
class StepRecord:
    step: int
    # Adaptation batches the frame path had completed and not dropped
    # when the step was published.
    accepted_batches: int
    published_s: float


class Adapter:
    """Runs steps on adaptation batches in its own thread and low-
    priority stream, and publishes the adaptable parameters to the
    engine's back γ/β buffer."""

    def __init__(
        self,
        model: nn.Module,
        config: AdapterConfig,
        engine: TrtEngine,
        poisoned_batches: frozenset[int] = frozenset(),
    ):
        assert len(engine.layernorm_buffers) == 2, (
            "the handoff needs a front and a back γ/β buffer"
        )
        self.config = config
        self.engine = engine
        # Batch indices, counted from the last settle, whose adapter
        # input gets a NaN: exercises fault handling.
        self.poisoned_batches = poisoned_batches
        # Lowest priority: the frame path's stream is created at the
        # highest.
        self.stream = torch.cuda.Stream(priority=0)
        with torch.cuda.stream(self.stream):
            self.step = AdapterStep(
                model,
                config.tent,
                config.micro_batch,
                bf16=True,
                graph=config.graph,
            )
        torch.cuda.current_stream().wait_stream(self.stream)
        self.drops_oldest = config.mode == "deploy"
        num_slots = DEPLOY_SLOTS if self.drops_oldest else SPEC_SLOTS
        self.slots = [
            torch.empty(config.tent.B, 3, 224, 224, device="cuda")
            for _ in range(num_slots)
        ]
        self.consumed = [torch.cuda.Event() for _ in self.slots]
        for event in self.consumed:
            event.record(self.stream)
        self.cv = threading.Condition()
        self.holding = False
        self.busy = False
        self.stopping = False
        self.enabled = True
        # Set when the adapter thread dies, so nothing waits on it
        # forever.
        self.error: Exception | None = None
        self.reset_counters()
        self.thread = threading.Thread(target=self.run, name="adapter", daemon=True)
        self.thread.start()

    def reset_counters(self) -> None:
        self.free_slots = list(range(1, len(self.slots)))
        self.filling, self.filled = 0, 0
        self.pending: deque[PendingBatch] = deque()
        self.completed_batches = self.dropped_batches = self.taken_batches = 0
        self.steps = self.faults = 0
        self.blocked = False
        self.step_records: list[StepRecord] = []

    @property
    def accepted_batches(self) -> int:
        return self.completed_batches - self.dropped_batches

    @property
    def stepped_batches(self) -> int:
        # A reverted step still consumed its batch.
        return self.steps + self.faults

    def summary(self) -> dict[str, float | int]:
        return {
            "steps": self.steps,
            "faults": self.faults,
            "pending": len(self.pending),
            "completed_batches": self.completed_batches,
            "dropped_batches": self.dropped_batches,
            "drop_rate": self.dropped_batches / max(1, self.completed_batches),
        }

    def adaptation_progress(self) -> AdaptationProgress | None:
        """None in deploy mode: it drops adaptation batches instead of
        falling behind, so keeping up does not apply."""
        if self.drops_oldest:
            return None
        return AdaptationProgress(
            self.completed_batches, self.stepped_batches, self.blocked
        )

    def add_adaptation_frame(self, images: torch.Tensor) -> None:
        """Frame thread, frame stream: copy one preprocessed adaptation
        frame into the batch being filled."""
        assert images.shape == (1, 3, 224, 224) and images.dtype == torch.float32
        if not self.enabled:
            # Frames of a batch cut short by disable() never reach a
            # step.
            self.filled = 0
            return
        stream = torch.cuda.current_stream()
        if self.filled == 0:
            # The adapter may still be copying the previous batch out of
            # this slot.
            stream.wait_event(self.consumed[self.filling])
        self.slots[self.filling][self.filled].copy_(images[0])
        self.filled += 1
        if self.filled < self.config.tent.B:
            return
        ready = torch.cuda.Event()
        ready.record(stream)
        with self.cv:
            self.completed_batches += 1
            if self.drops_oldest and self.pending:
                self.free_slots.append(self.pending.popleft().slot)
                self.dropped_batches += 1
            self.pending.append(PendingBatch(self.filling, ready))
            self.cv.notify_all()
            self.blocked |= not self.free_slots
            self.wait_for(lambda: bool(self.free_slots))
            self.filling, self.filled = self.free_slots.pop(), 0

    def wait_for(self, done: Callable[[], bool]) -> None:
        """Under `cv`, on any thread but the adapter's.

        The error check comes first: deploy mode always has a free slot,
        so a dead adapter would otherwise go unnoticed.
        """
        while True:
            if self.error is not None:
                raise RuntimeError("the adapter thread failed") from self.error
            if done():
                return
            self.cv.wait()

    def run(self) -> None:
        # A thread's exception never reaches the caller on its own.
        try:
            self.step_until_stopped()
        except Exception as error:
            with self.cv:
                self.error = error
                self.cv.notify_all()
            raise

    def step_until_stopped(self) -> None:
        with torch.cuda.stream(self.stream):
            while True:
                with self.cv:
                    while not self.stopping and (self.holding or not self.pending):
                        self.cv.wait()
                    if self.stopping:
                        return
                    batch = self.pending.popleft()
                    self.busy = True
                    batch_index = self.taken_batches
                    self.taken_batches += 1
                self.step_on(batch, batch_index)
                with self.cv:
                    self.busy = False
                    self.cv.notify_all()

    def step_on(self, batch: PendingBatch, batch_index: int) -> None:
        self.stream.wait_event(batch.ready)
        self.step.images.copy_(self.slots[batch.slot])
        self.consumed[batch.slot].record(self.stream)
        with self.cv:
            self.free_slots.append(batch.slot)
            self.cv.notify_all()
        if batch_index in self.poisoned_batches:
            self.step.images[0, 0, 0, 0] = float("nan")
        self.step()
        # Publish only once the step has finished, so the frame stream
        # never waits behind a backward pass. A non-finite step leaves
        # its parameters in the back buffer, which is never published
        # before the next write.
        back, done = self.write_back_buffer()
        done.synchronize()
        if not bool(self.step.finite):
            self.step.revert()
            self.faults += 1
            log.warning(
                "adapter fault: non-finite adaptable parameters after batch %d, step reverted",
                batch_index,
            )
            return
        self.steps += 1
        self.engine.publish(back.index, done, self.steps)
        self.step_records.append(
            StepRecord(self.steps, self.accepted_batches, time.perf_counter())
        )

    def write_back_buffer(self) -> tuple[BackBuffer, torch.cuda.Event]:
        back = self.engine.back_buffer()
        self.stream.wait_event(back.released)
        back.tensor.copy_(self.step.packed)
        done = torch.cuda.Event()
        done.record(self.stream)
        return back, done

    @contextmanager
    def held(self) -> Iterator[None]:
        """The adapter thread takes no batch and runs no step inside, so
        another thread may touch its state."""
        with self.cv:
            self.holding = True
            self.wait_for(lambda: not self.busy)
        try:
            yield
        finally:
            with self.cv:
                self.holding = False
                self.cv.notify_all()

    def reset_to_source(self) -> None:
        """Runtime operation: the adaptable parameters, momentum and the
        published γ/β go back to source.

        The step count keeps running.
        """
        with self.held():
            self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.cuda.stream(self.stream):
            self.step.reset_to_source()
            back, done = self.write_back_buffer()
            done.synchronize()
            self.engine.publish(back.index, done, self.steps)

    def drain(self) -> None:
        """Wait until every queued adaptation batch has been stepped
        on."""
        with self.cv:
            self.wait_for(lambda: not self.pending and not self.busy)

    def settle(self) -> None:
        """Between replays: drop the backlog, reset to source and count
        again from step 0."""
        with self.held():
            self.reset_counters()
            self.reset_parameters()

    def disable(self) -> None:
        """Runtime operation: stop adapting and keep the last good
        adaptable parameters."""
        with self.held():
            self.enabled = False
            self.dropped_batches += len(self.pending)
            self.free_slots += [b.slot for b in self.pending]
            self.pending.clear()

    def enable(self) -> None:
        with self.cv:
            self.enabled = True

    def close(self) -> None:
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        self.thread.join()
