from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from wm_poc.local_global.losses import (  # noqa: E402
    action_smoothness,
    combined_local_loss,
    delta_mse,
    one_step_mse,
    rollout_mse,
    variance_penalty,
)
from wm_poc.local_global.models import build_local_model  # noqa: E402

PATCHES = 9
DIM = 6
ACTION = 2


def test_rollout_mse_matches_manual():
    pred = torch.ones(2, 3, 4)
    target = torch.zeros(2, 3, 4)
    assert float(rollout_mse(pred, target)) == pytest.approx(1.0)
    discounted = float(rollout_mse(pred, target, discount=0.5))
    assert discounted == pytest.approx(1.0)  # constant error: discounting is a no-op
    with pytest.raises(ValueError):
        rollout_mse(pred, target[:, :2])


def test_rollout_mse_discount_weights_early_steps():
    pred = torch.zeros(1, 2, 1)
    target = torch.stack(
        [torch.zeros(1, 1), torch.ones(1, 1)], dim=1
    )  # error only at step 2
    undiscounted = float(rollout_mse(pred, target))
    discounted = float(rollout_mse(pred, target, discount=0.5))
    assert discounted < undiscounted


def test_one_step_and_delta():
    x = torch.zeros(4, 3)
    pred = torch.full((4, 3), 0.5)
    target = torch.ones(4, 3)
    assert float(one_step_mse(pred, target)) == pytest.approx(0.25)
    assert float(delta_mse(x, pred, target)) == pytest.approx(0.25)


def test_action_smoothness_zero_for_constant():
    actions = torch.ones(2, 5, 3)
    assert float(action_smoothness(actions)) == 0.0
    actions = torch.cumsum(torch.ones(2, 5, 3), dim=1)
    assert float(action_smoothness(actions)) == pytest.approx(1.0)


def test_variance_penalty_detects_collapse():
    collapsed = torch.zeros(32, 8)
    spread = torch.randn(512, 8) * 2.0
    assert float(variance_penalty(collapsed)) == pytest.approx(1.0)
    assert float(variance_penalty(spread)) < 0.2


def _fake_batch(batch=4, ctx=2, k=3):
    return {
        "z_context": torch.randn(batch, ctx, PATCHES, DIM),
        "z_targets": torch.randn(batch, k, PATCHES, DIM),
        "actions_context": torch.randn(batch, ctx - 1, ACTION),
        "actions": torch.randn(batch, k, ACTION),
    }


def test_combined_loss_finite_with_expected_keys():
    model = build_local_model(
        patches=PATCHES,
        embed_dim=DIM,
        step_action_dim=ACTION,
        local_dim=8,
        hidden_dim=16,
        num_layers=1,
    )
    loss, metrics = combined_local_loss(
        _fake_batch(),
        model,
        {"lambda_jacobian": 0.01, "lambda_variance": 0.1},
    )
    # Detach for the scalar check; the un-detached tensor is used by backward() below.
    assert math.isfinite(float(loss.detach()))
    for key in (
        "loss_total",
        "loss_rollout",
        "loss_one_step",
        "loss_delta",
        "loss_jacobian",
        "loss_variance",
        "rollout_mse_vs_static",
    ):
        assert key in metrics and math.isfinite(metrics[key])
    loss.backward()  # must be differentiable end to end
