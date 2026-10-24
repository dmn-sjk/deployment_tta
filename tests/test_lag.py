import copy

import numpy as np
import pytest
import torch
from conftest import NoiseFrames

from tta_deploy.reference import (
    fixed_lag_prediction_steps,
    reference_steps_before,
    run_reference,
)
from tta_deploy.tent import (
    Tent,
    TentConfig,
    configure_tent,
    entropy,
    flat_parameters,
    load_flat_parameters,
)


def test_reference_steps_before_counts_completed_adaptation_batches():
    config = TentConfig(B=2, S=3)
    # Adaptation frames 0, 3, 6, 9, 12: batches complete at frames 3 and
    # 9, and count from the next frame on.
    assert reference_steps_before(13, config).tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
    ]


def test_zero_lag_is_bit_identical_to_the_reference(source_model):
    config, num_frames = TentConfig(4, 2), 96
    plain_model = copy.deepcopy(source_model)
    plain = run_reference(
        plain_model,
        NoiseFrames(num_frames),
        Tent(plain_model, config),
        num_workers=2,
    )
    zero_lag = run_reference(
        source_model,
        NoiseFrames(num_frames),
        Tent(source_model, config),
        num_workers=2,
        prediction_steps=fixed_lag_prediction_steps(num_frames, config, 0),
    )
    assert (zero_lag.logits_sha256, zero_lag.params_sha256) == (
        plain.logits_sha256,
        plain.params_sha256,
    )


def fast_tent(model: torch.nn.Module, config: TentConfig) -> Tent:
    # TENT's lr moves the parameters too little to tell adjacent steps
    # apart in a few frames.
    tent = Tent(model, config)
    tent.optimizer.param_groups[0]["lr"] = 0.05
    return tent


@pytest.mark.parametrize("B, S, lag", [(1, 1, 2), (2, 3, 1)])
def test_lagged_frames_are_predicted_with_old_parameters_and_steps_are_unchanged(
    source_model, B, S, lag
):
    config, num_frames = TentConfig(B, S), 48
    plain_model = copy.deepcopy(source_model)
    trajectory = []
    plain = run_reference(
        plain_model,
        NoiseFrames(num_frames),
        fast_tent(plain_model, config),
        num_workers=2,
        on_step=lambda p: trajectory.append(flat_parameters(p).clone()),
    )
    prediction_steps = fixed_lag_prediction_steps(num_frames, config, lag)
    lagged_model = copy.deepcopy(source_model)
    lagged = run_reference(
        lagged_model,
        NoiseFrames(num_frames),
        fast_tent(lagged_model, config),
        num_workers=2,
        prediction_steps=prediction_steps,
    )
    # Steps start from the newest parameters whatever the lag, so the
    # trajectory is the reference's.
    assert lagged.params_sha256 == plain.params_sha256 and lagged.steps == plain.steps

    params = configure_tent(source_model)
    versions = [flat_parameters(params).clone(), *trajectory]
    expected = []
    for i in range(num_frames):
        load_flat_parameters(params, versions[prediction_steps[i]])
        with torch.no_grad():
            expected.append(
                entropy(source_model(NoiseFrames(num_frames)[i][0][None].cuda())).item()
            )
    lagged_error = np.abs(lagged.entropies - expected).max()
    plain_error = np.abs(plain.entropies - expected).max()
    print(
        f"B={B} S={S} L={lag}: max |Δentropy| vs per-frame oracle {lagged_error:.1e}, unlagged run {plain_error:.1e}"
    )
    assert lagged_error < 1e-4 < plain_error


def test_prediction_steps_from_the_future_are_rejected(source_model):
    config = TentConfig(1, 1)
    too_new = reference_steps_before(8, config) + np.array([0, 0, 0, 1, 0, 0, 0, 0])
    with pytest.raises(AssertionError, match="has not run yet"):
        run_reference(
            source_model,
            NoiseFrames(8),
            Tent(source_model, config),
            num_workers=0,
            prediction_steps=np.maximum.accumulate(too_new),
        )
