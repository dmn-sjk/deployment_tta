import copy

import numpy as np
import pytest
import torch
from conftest import Uint8NoiseFrames

from tta_deploy.methods.frame_path import FramePath
from tta_deploy.methods.pytorch import PytorchTent
from tta_deploy.reference import run_reference
from tta_deploy.tent import Tent, TentConfig, entropy, flat_parameters

STEPS = 50

pytestmark = pytest.mark.usefixtures("non_default_stream")


@pytest.mark.parametrize("B", [1, 4, 16, 64])
@pytest.mark.parametrize("S", [1, 4])
def test_pytorch_tent_matches_the_reference_over_a_short_horizon(source_model, B, S):
    config, num_frames = TentConfig(B, S), STEPS * B * S
    reference_model = copy.deepcopy(source_model)
    reference_trajectory = []
    reference = run_reference(
        reference_model,
        Uint8NoiseFrames(num_frames, preprocess=True),
        Tent(reference_model, config),
        num_workers=4,
        on_step=lambda p: reference_trajectory.append(flat_parameters(p).clone()),
    )

    method = Tent(source_model, config)
    frame_path = FramePath(PytorchTent("pytorch_tent", source_model, method))
    frames = Uint8NoiseFrames(num_frames, preprocess=False)
    logits, trajectory = [], []
    for i in range(num_frames):
        logits.append(frame_path(frames[i][0]).clone())
        if config.is_adaptation_frame(i) and method.accumulated == 0:
            trajectory.append(flat_parameters(method.adaptable_parameters).clone())
    logits = torch.cat(logits)

    relative = [
        ((a - b).abs().max() / b.abs().max()).item()
        for a, b in zip(trajectory, reference_trajectory, strict=True)
    ]
    entropy_error = np.abs(entropy(logits).numpy() - reference.entropies).max()
    agreement = (logits.argmax(1).numpy() == reference.predictions).mean()
    print(
        f"B={B} S={S}: max relative L-inf parameter difference over {STEPS} steps {max(relative):.2e}, max |Δentropy| {entropy_error:.1e}, top-1 agreement {agreement:.4f}"
    )
    assert len(trajectory) == STEPS
    assert max(relative) < 1e-5
    assert entropy_error < 1e-3 and agreement > 0.99


def test_settle_puts_pytorch_tent_back_to_a_fresh_start(source_model):
    # 19 adaptation frames: 4 steps, then settle also has to drop 3
    # accumulated gradients.
    config, num_frames = TentConfig(4, 2), 38
    tent = PytorchTent("pytorch_tent", source_model, Tent(source_model, config))
    frame_path = FramePath(tent)
    frames = Uint8NoiseFrames(num_frames, preprocess=False)
    passes = []
    for _ in range(2):
        passes.append(
            torch.cat([frame_path(frames[i][0]).clone() for i in range(num_frames)])
        )
        assert tent.summary() == {"steps": 4}
        frame_path.settle()
    assert torch.equal(passes[0], passes[1])
