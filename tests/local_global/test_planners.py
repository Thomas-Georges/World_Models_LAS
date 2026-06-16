from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wm_poc.local_global.configs import PlannerConfig  # noqa: E402
from wm_poc.local_global.planners import (  # noqa: E402
    GlobalCEMPlanner,
    HybridCEMLocalRefinePlanner,
    LocalGradientPlanner,
    PlanContext,
    build_planner,
    cem_optimize,
    gradient_optimize,
    refinement_decision,
    squash_to_bounds,
    unsquash_from_bounds,
)

PATCHES = 4
DIM = 2


def _cfg(**overrides) -> PlannerConfig:
    base = dict(
        action_dim=DIM,
        action_low=(-1.0, -1.0),
        action_high=(1.0, 1.0),
        horizon=3,
        goal_steps=3,
        mpc_exec_steps=1,
        cem_population=64,
        cem_elites=8,
        cem_iters=6,
        cem_init_std=0.5,
        gd_iters=60,
        gd_lr=0.2,
        gradient_clip=10.0,
        action_smoothness=0.0,
        reject_refine_if_worse_by=0.05,
        frameskip=1,
    )
    base.update(overrides)
    return PlannerConfig(**base)


def state_to_latent(x: torch.Tensor) -> torch.Tensor:
    return x[..., None, :].expand(*x.shape[:-1], PATCHES, DIM).contiguous()


class ToyGlobalModel:
    """Exact integrator dynamics: x' = x + gain * a, latent = broadcast state."""

    frameskip = 1
    context_len = 1

    def __init__(self, gain: float = 0.1) -> None:
        self.gain = gain

    def init_state(self, z_context, proprio_context=None, actions_context=None):
        return {"x": z_context[-1].mean(dim=-2)}

    def rollout_final(self, state, actions):
        x = state["x"].unsqueeze(0).expand(actions.shape[0], -1).clone()
        for t in range(actions.shape[1]):
            x = x + self.gain * actions[:, t]
        return state_to_latent(x)

    def advance(self, state, actions):
        x = state["x"].clone()
        step_latents = []
        for t in range(actions.shape[0]):
            x = x + self.gain * actions[t]
            step_latents.append(state_to_latent(x))
        return {"x": x, "step_latents": torch.stack(step_latents, dim=0)}

    def current_latent(self, state):
        return state_to_latent(state["x"])


class ToyLocalModel:
    """Linear local surrogate; ``gain`` may disagree with the global model."""

    def __init__(self, gain: float = 0.1) -> None:
        self.gain = gain

    def encode_global_latent(self, z):
        return z.mean(dim=-2)

    def rollout_from_context(self, x_context, actions_context, actions):
        x = x_context[:, -1]
        states = []
        for t in range(actions.shape[1]):
            x = x + self.gain * actions[:, t]
            states.append(x)
        return torch.stack(states, dim=1)


def _toy_context(x0=(0.0, 0.0), x_goal=(0.25, -0.2), seed=0) -> PlanContext:
    z_context = state_to_latent(torch.tensor([list(x0)], dtype=torch.float32))
    z_goal = state_to_latent(torch.tensor(list(x_goal), dtype=torch.float32))
    return PlanContext(
        global_state=None,
        z_context=z_context,
        z_goal=z_goal,
        actions_context=torch.zeros(0, DIM),
        seed=seed,
    )


def test_squash_roundtrip_and_bounds():
    low = torch.tensor([-1.0, -2.0])
    high = torch.tensor([1.0, 0.5])
    raw = torch.randn(5, 4, 2) * 3
    actions = squash_to_bounds(raw, low, high)
    assert torch.all(actions >= low) and torch.all(actions <= high)
    recovered = squash_to_bounds(unsquash_from_bounds(actions, low, high), low, high)
    torch.testing.assert_close(recovered, actions, atol=1e-4, rtol=1e-4)


def test_cem_optimize_quadratic():
    cfg = _cfg()
    target = torch.full((cfg.horizon, cfg.step_action_dim), 0.3)

    def cost_fn(candidates):
        return torch.mean((candidates - target) ** 2, dim=(1, 2))

    actions, best_cost, trace = cem_optimize(
        cost_fn, cfg, device=torch.device("cpu"), seed=0
    )
    assert best_cost < trace[0]["best_cost"] or best_cost < 1e-3
    assert best_cost < 0.02
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
    assert len(trace) == cfg.cem_iters


def test_cem_is_deterministic_per_seed():
    cfg = _cfg()

    def cost_fn(candidates):
        return torch.mean(candidates**2, dim=(1, 2))

    a1, c1, _ = cem_optimize(cost_fn, cfg, device=torch.device("cpu"), seed=5)
    a2, c2, _ = cem_optimize(cost_fn, cfg, device=torch.device("cpu"), seed=5)
    torch.testing.assert_close(a1, a2)
    assert c1 == c2


def test_gradient_optimize_reduces_cost():
    cfg = _cfg()
    target = torch.full((cfg.horizon, cfg.step_action_dim), -0.4)

    def cost_fn(actions):
        cost = torch.mean((actions - target) ** 2)
        return cost, {"goal_cost": float(cost.detach())}

    actions, best_cost, best_components, trace, backward_steps = gradient_optimize(
        cost_fn, cfg, device=torch.device("cpu")
    )
    assert "goal_cost" in best_components
    assert trace[-1]["cost"] < trace[0]["cost"]
    assert best_cost < 0.01
    assert backward_steps == cfg.gd_iters
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


def test_global_cem_planner_reaches_toy_goal():
    cfg = _cfg()
    planner = GlobalCEMPlanner(ToyGlobalModel(), cfg)
    result = planner.plan(_toy_context())
    assert result.planner_name == "global_cem"
    assert result.costs["goal_cost"] <= result.costs["first_iter_cost"]
    assert result.costs["goal_cost"] < 1e-3
    # population x iters for the search, +1 for the final pure-goal re-score
    assert result.metadata["num_global_forward_calls"] == cfg.cem_population * cfg.cem_iters + 1
    assert result.metadata["num_backward_steps"] == 0


def test_local_gradient_planner_cost_decreases():
    cfg = _cfg()
    planner = LocalGradientPlanner(ToyLocalModel(), cfg, optimizer="adam")
    result = planner.plan(_toy_context())
    assert result.planner_name == "local_adam"
    assert result.trace[-1]["cost"] < result.trace[0]["cost"]
    assert result.costs["goal_cost"] < 1e-3
    assert result.metadata["num_backward_steps"] == cfg.gd_iters
    low = torch.tensor(cfg.step_action_low)
    high = torch.tensor(cfg.step_action_high)
    assert torch.all(result.actions >= low) and torch.all(result.actions <= high)


def test_refinement_decision_three_cases():
    """The bounded-worsening gate, covering all three regimes explicitly.

    The gate rejects only refinements that worsen the global total by *more
    than* the tolerance; ``improved`` is the stricter non-worsening test.
    """
    cem_total = 1.0
    tol = 0.05  # absolute tolerance for cem_total == 1.0 (i.e. 5%)

    # (1) strictly improved -> accepted and improved
    accepted, improved = refinement_decision(cem_total, 0.8, tol, global_rescore=True)
    assert accepted is True and improved is True

    # (2) worse but within tolerance -> accepted but NOT improved (Option B)
    accepted, improved = refinement_decision(cem_total, 1.04, tol, global_rescore=True)
    assert accepted is True and improved is False

    # (3) worse by more than tolerance -> rejected (and not improved)
    accepted, improved = refinement_decision(cem_total, 1.20, tol, global_rescore=True)
    assert accepted is False and improved is False

    # Boundary: exactly at the tolerance edge is still accepted (strict >).
    accepted, _ = refinement_decision(cem_total, cem_total + tol, tol, global_rescore=True)
    assert accepted is True

    # Without the gate, even a large worsening is accepted (but flagged not improved).
    accepted, improved = refinement_decision(cem_total, 5.0, tol, global_rescore=False)
    assert accepted is True and improved is False


def test_hybrid_accepts_good_refinement():
    cfg = _cfg()
    planner = HybridCEMLocalRefinePlanner(
        ToyGlobalModel(), ToyLocalModel(), cfg, global_rescore=True
    )
    result = planner.plan(_toy_context())
    assert result.planner_name == "hybrid_cem_local_refine_global_rescore"
    # Local model is exact here, so refinement should not be rejected.
    assert result.metadata["accepted_refinement"] is True
    assert result.costs["goal_cost"] <= result.costs["cem_global_cost"] * 1.05 + 1e-9


def test_hybrid_rejects_bad_refinement_with_rescore():
    cfg = _cfg()
    # Sign-flipped surrogate: local gradients push actions the wrong way.
    planner = HybridCEMLocalRefinePlanner(
        ToyGlobalModel(gain=0.1), ToyLocalModel(gain=-0.1), cfg, global_rescore=True
    )
    result = planner.plan(_toy_context())
    assert result.metadata["accepted_refinement"] is False
    assert result.costs["goal_cost"] == result.costs["cem_global_cost"]
    assert result.costs["global_rescore_cost"] > result.costs["cem_global_cost"]


def test_hybrid_without_rescore_keeps_refinement():
    cfg = _cfg()
    planner = HybridCEMLocalRefinePlanner(
        ToyGlobalModel(gain=0.1), ToyLocalModel(gain=-0.1), cfg, global_rescore=False
    )
    result = planner.plan(_toy_context())
    assert result.planner_name == "hybrid_cem_local_refine"
    assert result.metadata["accepted_refinement"] is True
    assert result.costs["goal_cost"] == result.costs["global_rescore_cost"]


def test_build_planner_registry():
    cfg = _cfg()
    toy_global, toy_local = ToyGlobalModel(), ToyLocalModel()
    for name in (
        "global_cem",
        "local_cem",
        "local_gd",
        "local_adam",
        "hybrid_cem_local_refine",
        "hybrid_cem_local_refine_global_rescore",
    ):
        planner = build_planner(name, global_model=toy_global, local_model=toy_local, cfg=cfg)
        assert planner.name == name
    with pytest.raises(ValueError):
        build_planner("does_not_exist", global_model=toy_global, local_model=toy_local, cfg=cfg)
    with pytest.raises(ValueError):
        build_planner("global_cem", cfg=cfg)
    with pytest.raises(ValueError):
        build_planner("local_adam", global_model=toy_global, cfg=cfg)
