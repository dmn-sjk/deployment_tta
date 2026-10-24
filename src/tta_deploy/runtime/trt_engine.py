import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import tensorrt as trt
import torch

from tta_deploy.export.engine import LOGGER
from tta_deploy.export.onnx_export import (
    IMAGE_INPUT,
    IMAGE_SHAPE,
    LOGITS_OUTPUT,
)
from tta_deploy.model import NUM_CLASSES
from tta_deploy.tent import ADAPTABLE_VALUES


@dataclass(frozen=True)
class Published:
    index: int
    # Recorded after the buffer was written. None once a call has waited
    # on it: later calls on the same stream are ordered after it anyway.
    ready: torch.cuda.Event | None
    # Step whose adaptable parameters the buffer holds.
    step: int


@dataclass(frozen=True)
class BackBuffer:
    index: int
    tensor: torch.Tensor
    # Its writer waits on this first, so no in-flight inference still
    # reads it.
    released: torch.cuda.Event


def load_cuda_engine(engine_path: Path) -> trt.ICudaEngine:
    engine = trt.Runtime(LOGGER).deserialize_cuda_engine(engine_path.read_bytes())
    assert engine is not None, f"cannot load {engine_path}"

    return engine


class TrtEngine:
    """Batch-1 TensorRT engine with one execution context pre-bound to
    each packed FP32 γ/β buffer, so publishing a buffer is a pointer
    swap."""

    def __init__(
        self,
        engine_path: Path,
        layernorm_names: list[str],
        layernorm_buffers: Sequence[torch.Tensor] = (),
    ):
        self.engine = load_cuda_engine(engine_path)

        io_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.layernorm_names = [
            n for n in io_names if n not in (IMAGE_INPUT, LOGITS_OUTPUT)
        ]

        assert self.layernorm_names in ([], layernorm_names), (
            "engine γ/β inputs do not match the adaptable parameters"
        )
        assert bool(self.layernorm_names) == bool(layernorm_buffers), (
            "γ/β buffers are needed exactly when the engine has γ/β inputs"
        )
        assert tuple(self.engine.get_tensor_shape(IMAGE_INPUT)) == IMAGE_SHAPE

        self.logits = torch.empty(1, NUM_CLASSES, device="cuda")

        # The engine reads these buffers on every call, so they must
        # outlive the contexts bound to them. They are fixed, so copy
        # them with tuple.
        self.layernorm_buffers = tuple(layernorm_buffers)

        # An engine with constant γ/β still needs its one context.
        # Running ICudaEngine requires at least one IExecutionContext
        # for per-inference state.
        self.contexts = [
            self.engine.create_execution_context()
            for _ in range(max(1, len(self.layernorm_buffers)))
        ]

        # Why one per buffer: bound addresses belong to a context, not
        # the engine. There are other options, but this is fine and
        # simple for now. The cost: each context allocates its own
        # activation scratch memory, so two contexts use twice that. For
        # batch-1 ViT-B/16 that's small compared with the weights, which
        # the contexts share.
        for context in self.contexts:
            context.set_tensor_address(LOGITS_OUTPUT, self.logits.data_ptr())

        # strict=False: a constant-γ/β engine has one context and no
        # buffers.
        for context, packed in zip(self.contexts, self.layernorm_buffers, strict=False):
            self.bind_layernorm_buffer(context, packed)

        self.released = [torch.cuda.Event() for _ in self.contexts]

        # Prevent race conditions on reading `published`, enqueueing and
        # recording the release.
        self.lock = threading.Lock()

        # Buffer 0 starts published at step 0: the first call waits for
        # the current stream's work, which wrote it.
        ready = torch.cuda.Event()
        ready.record()
        self.published = Published(0, ready, 0)

        # Step of the γ/β the last call used: the frame path's lag
        # trace.
        self.called_step = 0

    def bind_layernorm_buffer(
        self, context: trt.IExecutionContext, packed: torch.Tensor
    ) -> None:
        assert (
            packed.is_cuda
            and packed.dtype == torch.float32
            and packed.is_contiguous()
            and packed.shape == (ADAPTABLE_VALUES,)
        )

        offset = 0
        for name in self.layernorm_names:
            assert self.engine.get_tensor_dtype(name) == trt.float32

            context.set_tensor_address(
                name, packed.data_ptr() + offset * packed.element_size()
            )
            offset += trt.volume(self.engine.get_tensor_shape(name))

        assert offset == ADAPTABLE_VALUES

    def publish(self, index: int, ready: torch.cuda.Event, step: int) -> None:
        """Use γ/β buffer `index`, written by `step`, from the next call
        on.

        `ready` is recorded after the buffer was written, the next
        call's stream waits on it.
        """
        assert 0 <= index < len(self.layernorm_buffers)
        with self.lock:
            self.published = Published(index, ready, step)

    def release_event(self, index: int) -> torch.cuda.Event:
        # Recorded after the last call that read buffer `index`. Once
        # another buffer is published no later call reads it, so a
        # writer that waits on this event never overwrites γ/β an
        # in-flight inference still uses.
        with self.lock:
            assert self.published.index != index, (
                "the published γ/β buffer is still in use"
            )
            return self.released[index]

    @property
    def published_buffer(self) -> torch.Tensor:
        return self.layernorm_buffers[self.published.index]

    def back_buffer(self) -> BackBuffer:
        # Double-buffered handoff with a single writer: it stays the
        # back buffer until that writer publishes it.
        with self.lock:
            back = 1 - self.published.index
        return BackBuffer(back, self.layernorm_buffers[back], self.release_event(back))

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Enqueue on the current stream.

        Returns the engine's own logits buffer, valid until the next
        call.
        """
        assert (
            images.is_cuda
            and images.dtype == torch.float32
            and images.is_contiguous()
            and tuple(images.shape) == IMAGE_SHAPE
        )

        stream = torch.cuda.current_stream()
        with self.lock:
            published = self.published
            self.called_step = published.step
            if published.ready is not None:
                # First call since a publish: wait for its writer.
                stream.wait_event(published.ready)
                self.published = replace(published, ready=None)

            context = self.contexts[published.index]
            context.set_tensor_address(IMAGE_INPUT, images.data_ptr())

            assert context.execute_async_v3(stream.cuda_stream)

            self.released[published.index].record(stream)
        return self.logits


def engine_logits(engine: TrtEngine, images: torch.Tensor) -> torch.Tensor:
    """Logits of a batch of images, one batch-1 call per image."""
    return torch.cat(
        [engine(images[i : i + 1].contiguous()).clone() for i in range(len(images))]
    )
