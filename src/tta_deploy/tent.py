from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

# ViT-B/16: 25 LayerNorms (2 per block × 12, plus the final one) × 768
# channels × (γ, β).
ADAPTABLE_VALUES = 25 * 768 * 2


@dataclass(frozen=True)
class TentConfig:
    B: int
    S: int

    def __post_init__(self) -> None:
        assert self.B >= 1 and self.S >= 1, f"B {self.B} and S {self.S} must be >= 1"

    @property
    def lr(self) -> float:
        # just constant TENT's default for now, dependent on the BS
        return 2.5e-4 * self.B / 64

    @property
    def name(self) -> str:
        return f"tent_B{self.B}_S{self.S}"

    def is_adaptation_frame(self, frame_index: int) -> bool:
        return frame_index % self.S == 0


def adaptable_parameter_names(model: nn.Module) -> list[str]:
    """LayerNorm γ/β names, in the order of configure_tent and of the
    packed γ/β buffer."""
    return [
        f"{name}.{p}"
        for name, module in model.named_modules()
        if isinstance(module, nn.LayerNorm)
        for p in ("weight", "bias")
    ]


def adaptable_parameters(model: nn.Module) -> list[nn.Parameter]:
    state = dict(model.named_parameters())
    return [state[n] for n in adaptable_parameter_names(model)]


def configure_tent(model: nn.Module) -> list[nn.Parameter]:
    # eval() as in ROID: no dropout or BatchNorm in this checkpoint, so
    # train() would give the same forward.
    model.eval()
    model.requires_grad_(False)
    params = adaptable_parameters(model)
    for p in params:
        p.requires_grad_(True)
    assert len(params) == 50 and sum(p.numel() for p in params) == ADAPTABLE_VALUES
    return params


def tent_optimizer(params: list[nn.Parameter], config: TentConfig) -> torch.optim.SGD:
    return torch.optim.SGD(
        params,
        lr=config.lr,
        momentum=0.9,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=True,
    )


def flat_parameters(params: list[nn.Parameter]) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in params])


def load_flat_parameters(params: Sequence[torch.Tensor], flat: torch.Tensor) -> None:
    """In-place copy, so optimizer state and captured CUDA graphs keep
    pointing at the same tensors."""
    assert flat.shape == (sum(p.numel() for p in params),)
    with torch.no_grad():
        for p, chunk in zip(
            params, flat.split([p.numel() for p in params]), strict=True
        ):
            p.copy_(chunk.view_as(p))


def packed_adaptable_parameters(model: nn.Module) -> torch.Tensor:
    return flat_parameters(adaptable_parameters(model))


def entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class Tent:
    def __init__(self, model: nn.Module, config: TentConfig):
        self.config = config
        self.adaptable_parameters = configure_tent(model)
        self.optimizer = tent_optimizer(self.adaptable_parameters, config)
        self.source = flat_parameters(self.adaptable_parameters).clone()
        self.accumulated = 0

    def adapt(self, adaptation_logits: torch.Tensor) -> bool:
        """Accumulate the entropy gradient of some adaptation frames,
        and run a step once B have been seen."""
        assert adaptation_logits.dtype == torch.float32, (
            "TENT loss runs in FP32: cast bf16 logits with .float()"
        )
        assert self.accumulated + len(adaptation_logits) <= self.config.B, (
            "adapt call crosses a step boundary"
        )
        self.accumulated += len(adaptation_logits)
        # sum / B instead of mean: bit-identical at batch B, and correct
        # when a step is accumulated over several calls.
        (entropy(adaptation_logits).sum() / self.config.B).backward()
        if self.accumulated < self.config.B:
            return False
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulated = 0
        return True

    def reset_to_source(self) -> None:
        # As freshly built: source adaptable parameters, no momentum, no
        # accumulated gradient.
        load_flat_parameters(self.adaptable_parameters, self.source)
        # SGD creates the momentum buffer again on the next step, as on
        # the first.
        self.optimizer.state.clear()
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulated = 0
