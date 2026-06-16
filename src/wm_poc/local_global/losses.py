"""Training losses for the local surrogate.

The default objective is a discounted multi-step rollout MSE in the compressed
local space, plus a one-step term and a delta term that keeps predicted state
changes calibrated. Pixel reconstruction is deliberately not part of the
objective (the local model has no decoder).
"""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to use local_global.losses.") from exc


def one_step_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def rollout_mse(
    pred_seq: torch.Tensor, target_seq: torch.Tensor, discount: float = 1.0
) -> torch.Tensor:
    """Discounted MSE over a ``(B, K, X)`` rollout."""
    if pred_seq.shape != target_seq.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred_seq.shape)} vs {tuple(target_seq.shape)}.")
    errors = torch.mean((pred_seq - target_seq) ** 2, dim=(0, 2))  # (K,)
    if discount == 1.0:
        return errors.mean()
    weights = discount ** torch.arange(errors.shape[0], device=errors.device, dtype=errors.dtype)
    return (errors * weights).sum() / weights.sum()


def delta_mse(
    x_prev: torch.Tensor, pred_next: torch.Tensor, target_next: torch.Tensor
) -> torch.Tensor:
    """MSE between predicted and true state changes (keeps step sizes honest)."""
    return torch.mean(((pred_next - x_prev) - (target_next - x_prev)) ** 2)


def action_smoothness(actions: torch.Tensor) -> torch.Tensor:
    """Mean squared difference between consecutive actions ``(..., K, A)``."""
    if actions.shape[-2] < 2:
        return actions.new_zeros(())
    return torch.mean((actions[..., 1:, :] - actions[..., :-1, :]) ** 2)


def variance_penalty(x: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """Hinge penalty on collapsed feature dimensions (for trainable projectors)."""
    std = x.reshape(-1, x.shape[-1]).std(dim=0)
    return torch.relu(target_std - std).mean()


def jacobian_norm_penalty(
    model: Any, x: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """Squared norm of d(step)/d(action) via a double-backward vector product."""
    action = action.detach().requires_grad_(True)
    next_x, _ = model.step(x.detach(), action)
    grad = torch.autograd.grad(next_x.sum(), action, create_graph=True)[0]
    return torch.mean(grad**2)


def combined_local_loss(
    batch: dict[str, torch.Tensor],
    model: Any,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the combined surrogate loss for one collated window batch.

    ``model`` is a :class:`wm_poc.local_global.models.LocalRolloutModel`.
    Returns the scalar loss and a detached metrics dictionary.
    """
    weights = weights or {}
    lambda_rollout = float(weights.get("lambda_rollout", 1.0))
    lambda_one_step = float(weights.get("lambda_one_step", 1.0))
    lambda_delta = float(weights.get("lambda_delta", 0.1))
    lambda_jacobian = float(weights.get("lambda_jacobian", 0.0))
    lambda_variance = float(weights.get("lambda_variance", 0.0))
    discount = float(weights.get("rollout_discount", 1.0))

    x_context = model.encode_global_latent(batch["z_context"])  # (B, C, X)
    x_targets = model.encode_global_latent(batch["z_targets"])  # (B, K, X)
    pred = model.rollout_from_context(x_context, batch["actions_context"], batch["actions"])

    loss_rollout = rollout_mse(pred, x_targets, discount=discount)
    loss_one_step = one_step_mse(pred[:, 0], x_targets[:, 0])
    loss_delta = delta_mse(x_context[:, -1], pred[:, 0], x_targets[:, 0])
    loss = lambda_rollout * loss_rollout + lambda_one_step * loss_one_step
    loss = loss + lambda_delta * loss_delta

    metrics: dict[str, float] = {
        "loss_rollout": float(loss_rollout.detach()),
        "loss_one_step": float(loss_one_step.detach()),
        "loss_delta": float(loss_delta.detach()),
    }
    if lambda_jacobian > 0:
        loss_jacobian = jacobian_norm_penalty(model, x_context[:, -1], batch["actions"][:, 0])
        loss = loss + lambda_jacobian * loss_jacobian
        metrics["loss_jacobian"] = float(loss_jacobian.detach())
    if lambda_variance > 0:
        loss_variance = variance_penalty(x_targets)
        loss = loss + lambda_variance * loss_variance
        metrics["loss_variance"] = float(loss_variance.detach())

    # Scale-free diagnostic: rollout error relative to a "predict no change" baseline.
    with torch.no_grad():
        baseline = x_context[:, -1:].expand_as(x_targets)
        baseline_mse = torch.mean((baseline - x_targets) ** 2)
        metrics["rollout_mse_vs_static"] = float(
            loss_rollout.detach() / baseline_mse.clamp_min(1e-12)
        )
    metrics["loss_total"] = float(loss.detach())
    return loss, metrics
