import torch
from torch import nn

from tta_deploy.tent import (
    ADAPTABLE_VALUES,
    TentConfig,
    configure_tent,
    entropy,
    flat_parameters,
    load_flat_parameters,
    tent_optimizer,
)


def sequence_major_attention_input(
    module: nn.MultiheadAttention, args: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    # MHA transposes batch-first input to [197, B, 768]. Under autograd
    # matmul will not fold that non-contiguous view and broadcasts the
    # in-projection weight into a bmm, 10x slower at B = 4. Same values,
    # laid out so the transpose is contiguous.
    query, key, value = args
    assert query is key is value and module.batch_first
    x = query.transpose(0, 1).contiguous().transpose(0, 1)
    return x, x, x


class AdapterStep:
    """One TENT step on a batch of B adaptation frames: forward and
    backward in micro-batches of m, SGD on the FP32 adaptable
    parameters, optionally one CUDA graph."""

    def __init__(
        self,
        model: nn.Module,
        config: TentConfig,
        micro_batch: int,
        bf16: bool,
        graph: bool,
    ):
        assert config.B % micro_batch == 0, "B must be a multiple of the micro-batch"
        self.model, self.config, self.micro_batch, self.bf16 = (
            model,
            config,
            micro_batch,
            bf16,
        )
        self.params = configure_tent(model)
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_pre_hook(sequence_major_attention_input)
        assert all(p.dim() == 1 for p in self.params)
        self.optimizer = tent_optimizer(self.params, config)
        # SGD clones the first gradient into a new momentum buffer, a
        # Python branch a graph cannot replay. Zero buffers made up
        # front give the same first step: 0.9 * 0 + g = g.
        self.momentum = [torch.zeros_like(p) for p in self.params]
        for p, m in zip(self.params, self.momentum, strict=True):
            self.optimizer.state[p]["momentum_buffer"] = m
        self.source = flat_parameters(self.params).clone()
        self.images = torch.zeros(config.B, 3, 224, 224, device="cuda")
        self.packed = torch.empty(ADAPTABLE_VALUES, device="cuda")
        self.before_step = torch.empty(2 * ADAPTABLE_VALUES, device="cuda")
        self.finite = torch.ones((), dtype=torch.bool, device="cuda")
        self.graph = self.capture() if graph else None

    def body(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            torch.cat([*self.params, *self.momentum], out=self.before_step)
        for images in self.images.split(self.micro_batch):
            # Graph capture needs the autocast weight cache off.
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=self.bf16,
                cache_enabled=False,
            ):
                logits = self.model(images)
            # sum / B: the micro-batch gradients add up to the batch-B
            # mean, as in Tent.adapt.
            (entropy(logits.float()).sum() / self.config.B).backward()
        self.optimizer.step()
        with torch.no_grad():
            torch.cat(tuple(self.params), out=self.packed)
            self.finite.copy_(self.packed.isfinite().all())

    def capture(self) -> torch.cuda.CUDAGraph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.body()
        torch.cuda.current_stream().wait_stream(side)
        # Warm-up steps moved the parameters. Capture only records
        # kernels, so the graph starts from source.
        self.reset_to_source()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.body()
        return graph

    def __call__(self) -> None:
        """Step on `images` in the current stream.

        Leaves the new adaptable parameters in `packed` and whether they
        are finite in `finite`.
        """
        if self.graph is None:
            self.body()
        else:
            self.graph.replay()

    def revert(self) -> None:
        """Undo the last step: adaptable parameters and momentum as they
        were before it."""
        params, momentum = self.before_step.split(ADAPTABLE_VALUES)
        load_flat_parameters(self.params, params)
        load_flat_parameters(self.momentum, momentum)
        with torch.no_grad():
            torch.cat(tuple(self.params), out=self.packed)

    def reset_to_source(self) -> None:
        load_flat_parameters(self.params, self.source)
        with torch.no_grad():
            for m in self.momentum:
                m.zero_()
            self.packed.copy_(self.source)
