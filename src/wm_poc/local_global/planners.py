"""Planners for the local/global track.

All planners optimize an open-loop action sequence ``(horizon, step_action_dim)``
toward a latent goal:

- ``GlobalCEMPlanner`` — zero-order CEM scored by the global model (no_grad).
- ``LocalGradientPlanner`` — first-order GD/Adam through the differentiable
  local surrogate, with tanh-squashed (always in-bounds) actions.
- ``LocalCEMPlanner`` — CEM scored by the local surrogate (separates the
  "small model" effect from the "gradient" effect).
- ``HybridCEMLocalRefinePlanner`` — global CEM proposes, local gradients
  refine, and (optionally) a global re-score rejects refinements that worsen
  the trusted global cost.

Cost bookkeeping: planners optimize ``goal + action_smoothness * smoothness``
but report the components separately — ``goal_cost`` is always the pure latent
goal distance of the returned actions, ``total_cost`` the optimized objective.
Planners also count model usage (global forward rollouts, local forwards,
backward steps) so the results notebook can compare compute fairly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to use local_global.planners.") from exc

from wm_poc.local_global.configs import PlannerConfig
from wm_poc.local_global.global_models import latent_goal_cost

# Warm starts often sit exactly on the action bounds (CEM clamps its samples
# there). atanh(1 - 1e-5) lands where the tanh derivative is ~2e-5, freezing
# gradient refinement; a 1e-2 margin keeps the squash gradient healthy (~0.02).
WARM_START_EPS = 1e-2


@dataclass
class PlanContext:
    """Everything a planner may need for one planning call.

    ``z_context``/``actions_context`` follow the dataset convention: frames are
    frameskip-spaced and ``actions_context[t]`` is the folded block between
    frames ``t`` and ``t+1``. ``horizon`` (when set) overrides the configured
    planning horizon — the MPC loop shrinks it to the steps remaining to the
    goal. ``local_context_len`` is the context length the local surrogate was
    trained with; local planners truncate their inputs to it.
    """

    global_state: dict[str, Any] | None
    z_context: torch.Tensor  # (C, P, D)
    z_goal: torch.Tensor  # (P, D)
    actions_context: torch.Tensor | None = None  # (C-1, step_action_dim)
    proprio_context: torch.Tensor | None = None
    seed: int = 0
    horizon: int | None = None
    local_context_len: int | None = None


@dataclass
class PlanResult:
    actions: torch.Tensor
    costs: dict[str, float]
    trace: list[dict[str, float]]
    planner_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePlanner(Protocol):
    name: str

    def plan(self, context: PlanContext) -> PlanResult: ...


def squash_to_bounds(
    raw: torch.Tensor, low: torch.Tensor, high: torch.Tensor
) -> torch.Tensor:
    """Map unconstrained values into [low, high] differentiably."""
    return low + (high - low) * 0.5 * (torch.tanh(raw) + 1.0)


def unsquash_from_bounds(
    actions: torch.Tensor, low: torch.Tensor, high: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    unit = (actions - low) / (high - low) * 2.0 - 1.0
    return torch.atanh(unit.clamp(-1.0 + eps, 1.0 - eps))


def smoothness_batch(actions: torch.Tensor) -> torch.Tensor:
    """Mean squared consecutive-action difference per batch element: (B, K, A) -> (B,)."""
    if actions.shape[-2] < 2:
        return actions.new_zeros(actions.shape[0])
    return torch.mean((actions[:, 1:] - actions[:, :-1]) ** 2, dim=(1, 2))


def action_bound_violations(
    actions: torch.Tensor, low: torch.Tensor, high: torch.Tensor, tol: float = 1e-6
) -> int:
    return int(((actions < low - tol) | (actions > high + tol)).sum().item())


def _bounds(cfg: PlannerConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    low = torch.tensor(cfg.step_action_low, dtype=torch.float32, device=device)
    high = torch.tensor(cfg.step_action_high, dtype=torch.float32, device=device)
    return low, high


def refinement_decision(
    cem_total: float, refined_total: float, tolerance: float, *, global_rescore: bool
) -> tuple[bool, bool]:
    """Global re-score gate decision for a local refinement.

    Returns ``(accepted, improved)`` where the two are deliberately *different*
    tests under the global model's optimized objective (goal + smoothness):

    - ``improved`` — the strict test: the refinement is no worse than CEM
      (``refined_total <= cem_total``).
    - ``accepted`` — the *bounded-worsening* gate: with ``global_rescore`` the
      refinement is rejected only if it worsens the global total by **more than**
      ``tolerance`` (so a refinement up to ``tolerance`` worse is still accepted);
      with ``global_rescore=False`` the refinement is always accepted.

    The gate is therefore not "accept only if it improves" — it is "reject only
    if it worsens by more than ``tolerance``". An accepted refinement may be
    slightly worse than CEM (``accepted and not improved``); a rejected one is
    always worse by more than the tolerance.
    """
    improved = refined_total <= cem_total
    if global_rescore and refined_total > cem_total + tolerance:
        return False, improved
    return True, improved


def cem_optimize(
    cost_fn: Callable[[torch.Tensor], torch.Tensor],
    cfg: PlannerConfig,
    *,
    device: torch.device,
    horizon: int | None = None,
    init_mean: torch.Tensor | None = None,
    seed: int = 0,
    iters: int | None = None,
) -> tuple[torch.Tensor, float, list[dict[str, float]]]:
    """Cross-entropy method over bounded action sequences.

    ``cost_fn`` maps a candidate batch ``(B, K, A)`` to costs ``(B,)`` and is
    expected to run under no_grad internally.
    """
    low, high = _bounds(cfg, device)
    steps = horizon if horizon is not None else cfg.horizon
    dim = cfg.step_action_dim
    mean = (
        init_mean.detach().clone().to(device)
        if init_mean is not None
        else torch.zeros(steps, dim, device=device)
    )
    std = torch.full((steps, dim), cfg.cem_init_std, device=device) * (high - low) * 0.5
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    best_actions = mean.clone()
    best_cost = float("inf")
    trace: list[dict[str, float]] = []
    for iteration in range(iters if iters is not None else cfg.cem_iters):
        noise = torch.randn(
            cfg.cem_population, steps, dim, generator=generator, device="cpu"
        ).to(device)
        candidates = (mean.unsqueeze(0) + noise * std.unsqueeze(0)).clamp(low, high)
        candidates[0] = mean.clamp(low, high)  # keep the current mean in the pool
        costs = cost_fn(candidates)
        elite_idx = torch.argsort(costs)[: cfg.cem_elites]
        elites = candidates[elite_idx]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0, unbiased=False).clamp_min(1e-4)
        iter_best = float(costs[elite_idx[0]].item())
        if iter_best < best_cost:
            best_cost = iter_best
            best_actions = candidates[elite_idx[0]].detach().clone()
        trace.append(
            {
                "iter": iteration,
                "best_cost": iter_best,
                "mean_cost": float(costs.mean().item()),
                "elite_mean_cost": float(costs[elite_idx].mean().item()),
                "std_mean": float(std.mean().item()),
            }
        )
    return best_actions, best_cost, trace


def gradient_optimize(
    cost_fn: Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, float]]],
    cfg: PlannerConfig,
    *,
    device: torch.device,
    horizon: int | None = None,
    init_actions: torch.Tensor | None = None,
    optimizer: str = "adam",
) -> tuple[torch.Tensor, float, dict[str, float], list[dict[str, float]], int]:
    """First-order optimization of a bounded action sequence.

    ``cost_fn`` maps squashed actions ``(K, A)`` to ``(scalar_cost, components)``
    and must be differentiable with respect to the actions.
    Returns ``(best_actions, best_cost, best_components, trace, backward_steps)``.
    """
    low, high = _bounds(cfg, device)
    steps = horizon if horizon is not None else cfg.horizon
    if init_actions is not None:
        raw = unsquash_from_bounds(
            init_actions.detach().to(device), low, high, eps=WARM_START_EPS
        )
    else:
        raw = torch.zeros(steps, cfg.step_action_dim, device=device)
    raw = raw.requires_grad_(True)
    if optimizer == "adam":
        opt: torch.optim.Optimizer = torch.optim.Adam([raw], lr=cfg.gd_lr)
    elif optimizer == "gd":
        opt = torch.optim.SGD([raw], lr=cfg.gd_lr)
    else:
        raise ValueError(f"Unknown gradient optimizer: {optimizer!r}")
    best_actions = squash_to_bounds(raw.detach(), low, high)
    best_cost = float("inf")
    best_components: dict[str, float] = {}
    trace: list[dict[str, float]] = []
    backward_steps = 0
    for iteration in range(cfg.gd_iters):
        actions = squash_to_bounds(raw, low, high)
        cost, components = cost_fn(actions)
        opt.zero_grad(set_to_none=True)
        cost.backward()
        if cfg.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_([raw], cfg.gradient_clip)
        opt.step()
        backward_steps += 1
        value = float(cost.detach().item())
        if value < best_cost:
            best_cost = value
            best_components = dict(components)
            best_actions = actions.detach().clone()
        trace.append({"iter": iteration, "cost": value, **components})
    return best_actions, best_cost, best_components, trace, backward_steps


def _local_inputs(
    local_model: Any, context: PlanContext, cfg: PlannerConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the planning context for the local surrogate.

    Truncates to the surrogate's trained context length so a GRU surrogate is
    never conditioned on more transitions than it saw during training.
    """
    c = context.local_context_len or context.z_context.shape[0]
    z_context = context.z_context[-c:]
    with torch.no_grad():
        x_context = local_model.encode_global_latent(z_context.to(device).unsqueeze(0))
        x_goal = local_model.encode_global_latent(context.z_goal.to(device).unsqueeze(0))[0]
    if context.actions_context is not None and context.actions_context.shape[0] > 0:
        blocks = context.actions_context[-(c - 1) :] if c > 1 else context.actions_context[:0]
        actions_context = blocks.float().to(device).unsqueeze(0)
    else:
        actions_context = torch.zeros(1, 0, cfg.step_action_dim, device=device)
    return x_context, actions_context, x_goal


class GlobalCEMPlanner:
    name = "global_cem"

    def __init__(self, global_model: Any, cfg: PlannerConfig, *, device: str = "cpu") -> None:
        self.global_model = global_model
        self.cfg = cfg
        self.device = torch.device(device)

    def _ensure_state(self, context: PlanContext) -> dict[str, Any]:
        if context.global_state is not None:
            return context.global_state
        return self.global_model.init_state(
            context.z_context, context.proprio_context, context.actions_context
        )

    def _goal_and_total(
        self, state: dict[str, Any], z_goal: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float, float]:
        """Pure goal cost, smoothness, and optimized total for one sequence."""
        z_final = self.global_model.rollout_final(state, actions.unsqueeze(0))
        goal = float(latent_goal_cost(z_final, z_goal)[0].item())
        smooth = float(smoothness_batch(actions.unsqueeze(0))[0].item())
        return goal, smooth, goal + self.cfg.action_smoothness * smooth

    def plan(self, context: PlanContext) -> PlanResult:
        start = time.perf_counter()
        counter = {"global_forward": 0}
        state = self._ensure_state(context)
        z_goal = context.z_goal

        def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
            counter["global_forward"] += candidates.shape[0]
            z_final = self.global_model.rollout_final(state, candidates)
            goal = latent_goal_cost(z_final, z_goal)
            return goal + self.cfg.action_smoothness * smoothness_batch(candidates)

        actions, best_total, trace = cem_optimize(
            cost_fn, self.cfg, device=self.device, horizon=context.horizon, seed=context.seed
        )
        goal, smooth, _ = self._goal_and_total(state, z_goal, actions)
        counter["global_forward"] += 1
        return PlanResult(
            actions=actions,
            costs={
                "goal_cost": goal,
                "smoothness_cost": smooth,
                "total_cost": best_total,
                "first_iter_cost": trace[0]["best_cost"],
            },
            trace=trace,
            planner_name=self.name,
            metadata={
                "num_global_forward_calls": counter["global_forward"],
                "num_local_forward_calls": 0,
                "num_backward_steps": 0,
                "plan_wall_time_sec": time.perf_counter() - start,
            },
        )


class LocalCEMPlanner:
    name = "local_cem"

    def __init__(self, local_model: Any, cfg: PlannerConfig, *, device: str = "cpu") -> None:
        self.local_model = local_model
        self.cfg = cfg
        self.device = torch.device(device)

    def plan(self, context: PlanContext) -> PlanResult:
        start = time.perf_counter()
        counter = {"local_forward": 0}
        x_context, actions_context, x_goal = _local_inputs(
            self.local_model, context, self.cfg, self.device
        )

        def goal_costs(candidates: torch.Tensor) -> torch.Tensor:
            counter["local_forward"] += candidates.shape[0]
            batch = candidates.shape[0]
            states = self.local_model.rollout_from_context(
                x_context.expand(batch, -1, -1),
                actions_context.expand(batch, -1, -1),
                candidates,
            )
            return torch.mean((states[:, -1] - x_goal) ** 2, dim=-1)

        def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return goal_costs(candidates) + self.cfg.action_smoothness * smoothness_batch(
                    candidates
                )

        actions, best_total, trace = cem_optimize(
            cost_fn, self.cfg, device=self.device, horizon=context.horizon, seed=context.seed
        )
        with torch.no_grad():
            goal = float(goal_costs(actions.unsqueeze(0))[0].item())
            smooth = float(smoothness_batch(actions.unsqueeze(0))[0].item())
        return PlanResult(
            actions=actions,
            costs={
                "goal_cost": goal,
                "smoothness_cost": smooth,
                "total_cost": best_total,
                "first_iter_cost": trace[0]["best_cost"],
            },
            trace=trace,
            planner_name=self.name,
            metadata={
                "num_global_forward_calls": 0,
                "num_local_forward_calls": counter["local_forward"],
                "num_backward_steps": 0,
                "plan_wall_time_sec": time.perf_counter() - start,
            },
        )


class LocalGradientPlanner:
    """First-order planner through the local surrogate (GD or Adam)."""

    def __init__(
        self,
        local_model: Any,
        cfg: PlannerConfig,
        *,
        optimizer: str = "adam",
        device: str = "cpu",
    ) -> None:
        self.local_model = local_model
        self.cfg = cfg
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.name = f"local_{optimizer}"

    def plan(
        self, context: PlanContext, init_actions: torch.Tensor | None = None
    ) -> PlanResult:
        start = time.perf_counter()
        counter = {"local_forward": 0}
        x_context, actions_context, x_goal = _local_inputs(
            self.local_model, context, self.cfg, self.device
        )

        def cost_fn(actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
            counter["local_forward"] += 1
            states = self.local_model.rollout_from_context(
                x_context, actions_context, actions.unsqueeze(0)
            )
            goal = torch.mean((states[0, -1] - x_goal) ** 2)
            smooth = smoothness_batch(actions.unsqueeze(0))[0]
            total = goal + self.cfg.action_smoothness * smooth
            return total, {
                "goal_cost": float(goal.detach().item()),
                "smoothness_cost": float(smooth.detach().item()),
            }

        actions, best_total, best_components, trace, backward_steps = gradient_optimize(
            cost_fn,
            self.cfg,
            device=self.device,
            horizon=context.horizon,
            init_actions=init_actions,
            optimizer=self.optimizer,
        )
        return PlanResult(
            actions=actions,
            costs={
                "goal_cost": best_components.get("goal_cost", best_total),
                "smoothness_cost": best_components.get("smoothness_cost", 0.0),
                "total_cost": best_total,
                "first_iter_cost": trace[0]["cost"],
            },
            trace=trace,
            planner_name=self.name,
            metadata={
                "num_global_forward_calls": 0,
                "num_local_forward_calls": counter["local_forward"],
                "num_backward_steps": backward_steps,
                "plan_wall_time_sec": time.perf_counter() - start,
            },
        )


class HybridCEMLocalRefinePlanner:
    """Global CEM proposes; local gradients refine; global re-score guards.

    With ``global_rescore=True`` the gate is *bounded-worsening*: a refinement
    is rejected only if it worsens the global cost by **more than**
    ``reject_refine_if_worse_by`` (relative); a refinement up to that tolerance
    worse is still accepted, and the CEM sequence is returned unchanged on
    rejection. With ``global_rescore=False`` the refined sequence is always
    returned, but the re-scored cost is still logged. The accept/reject
    comparison uses the optimized total (goal + smoothness) so both sides are
    scored on the same objective. See :func:`refinement_decision` for the exact
    rule and the ``accepted`` vs ``refinement_improved_global_cost`` distinction
    surfaced in the result metadata.
    """

    def __init__(
        self,
        global_model: Any,
        local_model: Any,
        cfg: PlannerConfig,
        *,
        optimizer: str = "adam",
        global_rescore: bool = True,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.global_rescore = global_rescore
        self.device = torch.device(device)
        self.cem = GlobalCEMPlanner(global_model, cfg, device=device)
        self.refiner = LocalGradientPlanner(local_model, cfg, optimizer=optimizer, device=device)
        self.global_model = global_model
        self.name = (
            "hybrid_cem_local_refine_global_rescore"
            if global_rescore
            else "hybrid_cem_local_refine"
        )

    def plan(self, context: PlanContext) -> PlanResult:
        start = time.perf_counter()
        state = self.cem._ensure_state(context)
        context = PlanContext(
            global_state=state,
            z_context=context.z_context,
            z_goal=context.z_goal,
            actions_context=context.actions_context,
            proprio_context=context.proprio_context,
            seed=context.seed,
            horizon=context.horizon,
            local_context_len=context.local_context_len,
        )
        cem_result = self.cem.plan(context)
        refine_result = self.refiner.plan(context, init_actions=cem_result.actions)

        cem_total = cem_result.costs["total_cost"]
        refined_goal, refined_smooth, refined_total = self.cem._goal_and_total(
            state, context.z_goal, refine_result.actions
        )
        rescore_calls = 1
        tolerance = self.cfg.reject_refine_if_worse_by * max(abs(cem_total), 1e-12)
        accepted, refinement_improved = refinement_decision(
            cem_total, refined_total, tolerance, global_rescore=self.global_rescore
        )
        if accepted:
            actions = refine_result.actions
            goal, smooth, total = refined_goal, refined_smooth, refined_total
        else:
            actions = cem_result.actions
            goal = cem_result.costs["goal_cost"]
            smooth = cem_result.costs["smoothness_cost"]
            total = cem_total

        trace = [
            {**entry, "stage": "global_cem"} for entry in cem_result.trace
        ] + [{**entry, "stage": "local_refine"} for entry in refine_result.trace]
        return PlanResult(
            actions=actions,
            costs={
                "goal_cost": goal,
                "smoothness_cost": smooth,
                "total_cost": total,
                "cem_global_cost": cem_total,
                "refined_local_cost": refine_result.costs["total_cost"],
                "global_rescore_cost": refined_total,
                "first_iter_cost": cem_result.costs["first_iter_cost"],
            },
            trace=trace,
            planner_name=self.name,
            metadata={
                "accepted_refinement": accepted,
                "refinement_improved_global_cost": refinement_improved,
                "num_global_forward_calls": (
                    cem_result.metadata["num_global_forward_calls"] + rescore_calls
                ),
                "num_local_forward_calls": refine_result.metadata["num_local_forward_calls"],
                "num_backward_steps": refine_result.metadata["num_backward_steps"],
                "plan_wall_time_sec": time.perf_counter() - start,
            },
        )


def build_planner(
    name: str,
    *,
    global_model: Any = None,
    local_model: Any = None,
    cfg: PlannerConfig,
    device: str = "cpu",
) -> BasePlanner:
    if name == "global_cem":
        if global_model is None:
            raise ValueError("global_cem requires a global model.")
        return GlobalCEMPlanner(global_model, cfg, device=device)
    if name == "local_cem":
        if local_model is None:
            raise ValueError("local_cem requires a local model.")
        return LocalCEMPlanner(local_model, cfg, device=device)
    if name in {"local_gd", "local_adam"}:
        if local_model is None:
            raise ValueError(f"{name} requires a local model.")
        return LocalGradientPlanner(
            local_model, cfg, optimizer=name.removeprefix("local_"), device=device
        )
    if name in {"hybrid_cem_local_refine", "hybrid_cem_local_refine_global_rescore"}:
        if global_model is None or local_model is None:
            raise ValueError(f"{name} requires both global and local models.")
        return HybridCEMLocalRefinePlanner(
            global_model,
            local_model,
            cfg,
            global_rescore=name.endswith("global_rescore"),
            device=device,
        )
    raise ValueError(f"Unknown planner: {name!r}")
