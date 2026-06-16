"""MPC-style planning evaluation for the local/global track.

First-version evaluation is *offline visual goal reaching on cached latents*:
start and goal latents come from held-out validation episodes, and the global
model serves both as the MPC simulator (executed actions advance its imagined
state) and as the scorer of final goal distance. For the synthetic task the
global model is exact, so results there reflect true dynamics; for DINO-WM the
comparison is "judged by the trusted global model" rather than by the real
environment — the real-environment loop remains the DINO-WM track's plan.py.

Each evaluated episode also reports:

- a *reference* cost obtained by replaying the dataset's true action sequence
  through the same simulator, which calibrates the success threshold, and
- the *local-vs-global rollout disagreement*: the executed action sequence is
  replayed open-loop through the local surrogate and compared, step by step in
  projected space, against the global model's imagined trajectory (spec 8.1).

The MPC horizon shrinks to the steps remaining to the goal, so the final
planning rounds optimize arrival at the goal rather than a point ``horizon``
steps beyond it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to use local_global.eval.") from exc

from wm_poc.dino_wm.configs import get_config_value
from wm_poc.local_global.configs import (
    PlannerConfig,
    action_data_dir,
    latent_cache_dir,
    typed_config,
)
from wm_poc.local_global.datasets import (
    LatentTrajectoryStore,
    compute_action_state_stats,
    fold_actions,
    split_store_episodes,
)
from wm_poc.local_global.global_models import build_global_model, latent_goal_cost
from wm_poc.local_global.models import load_local_checkpoint
from wm_poc.local_global.planners import (
    PlanContext,
    action_bound_violations,
    build_planner,
)


def _model_device(model: Any) -> torch.device:
    """Device of a model's first parameter (cpu fallback for paramless stubs)."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def sample_episode_tasks(
    store: LatentTrajectoryStore,
    episodes: list[int],
    *,
    context_len: int,
    goal_steps: int,
    frameskip: int,
    num_tasks: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample (episode, start) goal-reaching tasks from held-out episodes."""
    rng = np.random.default_rng(int(seed))
    span = (context_len - 1 + goal_steps) * frameskip
    candidates = [
        ep
        for ep in episodes
        if min(store.episode_length(ep) - 1, store.action_length(ep)) >= span
    ]
    if not candidates:
        raise ValueError(
            f"No validation episode is long enough for context_len={context_len}, "
            f"goal_steps={goal_steps}, frameskip={frameskip}."
        )
    tasks = []
    for _ in range(num_tasks):
        ep = int(rng.choice(candidates))
        limit = min(store.episode_length(ep) - 1, store.action_length(ep)) - span
        t0 = int(rng.integers(0, limit + 1))
        tasks.append({"episode": ep, "start_t": t0})
    return tasks


def _episode_tensors(
    store: LatentTrajectoryStore,
    task: dict[str, Any],
    *,
    context_len: int,
    goal_steps: int,
    frameskip: int,
    proprio_dim: int,
) -> dict[str, Any]:
    ep, t0 = task["episode"], task["start_t"]
    latents = store.latents(ep)
    frame_idx = [t0 + j * frameskip for j in range(context_len)]
    t_cur = frame_idx[-1]
    t_goal = t_cur + goal_steps * frameskip
    z_context = torch.from_numpy(np.asarray(latents[frame_idx], dtype=np.float32))
    z_goal = torch.from_numpy(np.asarray(latents[t_goal], dtype=np.float32))
    raw_context = np.asarray(store.actions(ep)[t0:t_cur], dtype=np.float32)
    actions_context = (
        torch.from_numpy(fold_actions(raw_context, frameskip))
        if raw_context.shape[0]
        else torch.zeros(0, store.action_dim * frameskip)
    )
    raw_reference = np.asarray(store.actions(ep)[t_cur:t_goal], dtype=np.float32)
    reference_actions = torch.from_numpy(fold_actions(raw_reference, frameskip))
    proprio_context = None
    states = store.states(ep)
    if states is not None and proprio_dim > 0:
        proprio = np.asarray(states[frame_idx], dtype=np.float32)
        proprio_context = torch.from_numpy(proprio[..., :proprio_dim])
    return {
        "z_context": z_context,
        "z_goal": z_goal,
        "actions_context": actions_context,
        "reference_actions": reference_actions,
        "proprio_context": proprio_context,
        "episode": ep,
        "start_t": t0,
    }


def _local_disagreement(
    local_model: Any,
    sample: dict[str, Any],
    executed: torch.Tensor,
    global_step_latents: torch.Tensor,
    local_context_len: int,
) -> float:
    """Open-loop local rollout over the executed actions vs the global trajectory."""
    with torch.no_grad():
        device = _model_device(local_model)
        c = min(local_context_len, sample["z_context"].shape[0])
        z_ctx = sample["z_context"][-c:].to(device)
        a_ctx = (
            sample["actions_context"][-(c - 1) :] if c > 1 else executed[:0]
        ).to(device)
        x_context = local_model.encode_global_latent(z_ctx.unsqueeze(0))
        x_local = local_model.rollout_from_context(
            x_context, a_ctx.unsqueeze(0), executed.to(device).unsqueeze(0)
        )[0]
        x_global = local_model.encode_global_latent(global_step_latents.to(device))
        return float(torch.mean((x_local - x_global) ** 2).item())


def run_mpc_episode(
    global_model: Any,
    planner: Any,
    sample: dict[str, Any],
    *,
    goal_steps: int,
    exec_steps: int,
    local_model: Any = None,
    local_context_len: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Run one MPC episode against the global model as simulator."""
    start_time = time.perf_counter()
    z_goal = sample["z_goal"]
    state = global_model.init_state(
        sample["z_context"], sample["proprio_context"], sample["actions_context"]
    )
    z_context = sample["z_context"].clone()
    actions_context = sample["actions_context"].clone()
    baseline = float(latent_goal_cost(z_context[-1:].clone(), z_goal)[0].item())

    remaining = goal_steps
    rounds: list[dict[str, Any]] = []
    executed: list[torch.Tensor] = []
    step_latents: list[torch.Tensor] = []
    totals = {"num_global_forward_calls": 0, "num_local_forward_calls": 0, "num_backward_steps": 0}
    accepted_flags: list[bool] = []
    while remaining > 0:
        context = PlanContext(
            global_state=state,
            z_context=z_context,
            z_goal=z_goal,
            actions_context=actions_context,
            proprio_context=sample["proprio_context"],
            seed=seed + len(rounds),
            horizon=remaining,  # shrink to the goal: planner optimizes arrival
            local_context_len=local_context_len,
        )
        result = planner.plan(context)
        for key in totals:
            totals[key] += int(result.metadata.get(key, 0))
        if "accepted_refinement" in result.metadata:
            accepted_flags.append(bool(result.metadata["accepted_refinement"]))
        step_count = min(exec_steps, remaining, result.actions.shape[0])
        chunk = result.actions[:step_count].detach().cpu()
        state = global_model.advance(state, chunk)
        new_latents = state["step_latents"].detach().cpu()  # (step_count, P, D)
        step_latents.append(new_latents)
        z_context = torch.cat([z_context, new_latents], dim=0)[
            -sample["z_context"].shape[0] :
        ]
        keep = sample["actions_context"].shape[0]
        joined = torch.cat([actions_context, chunk], dim=0)
        actions_context = joined[joined.shape[0] - keep :]
        executed.append(chunk)
        remaining -= step_count
        rounds.append(
            {
                "round": len(rounds),
                "planned_cost": result.costs.get("goal_cost"),
                "first_iter_cost": result.costs.get("first_iter_cost"),
                "trace": result.trace,
                "costs": result.costs,
                "metadata": {k: v for k, v in result.metadata.items() if not torch.is_tensor(v)},
            }
        )

    final_latent = global_model.current_latent(state).detach().cpu()
    final_dist = float(latent_goal_cost(final_latent.unsqueeze(0), z_goal)[0].item())

    # Reference: replay the dataset's true actions through the same simulator.
    ref_state = global_model.init_state(
        sample["z_context"], sample["proprio_context"], sample["actions_context"]
    )
    ref_state = global_model.advance(ref_state, sample["reference_actions"])
    ref_latent = global_model.current_latent(ref_state).detach().cpu()
    reference_dist = float(latent_goal_cost(ref_latent.unsqueeze(0), z_goal)[0].item())

    all_actions = torch.cat(executed, dim=0)
    all_step_latents = torch.cat(step_latents, dim=0)
    local_dist = None
    disagreement = None
    if local_model is not None:
        # The surrogate lives on the planner device (cuda); the simulator's
        # latents and the dataset goal are on CPU here, so move them first.
        local_device = _model_device(local_model)
        with torch.no_grad():
            x_final = local_model.encode_global_latent(final_latent.to(local_device).unsqueeze(0))[0]
            x_goal = local_model.encode_global_latent(z_goal.to(local_device).unsqueeze(0))[0]
            local_dist = float(torch.mean((x_final - x_goal) ** 2).item())
        disagreement = _local_disagreement(
            local_model, sample, all_actions, all_step_latents, local_context_len
        )

    low = torch.tensor(planner.cfg.step_action_low)
    high = torch.tensor(planner.cfg.step_action_high)
    return {
        "episode": sample["episode"],
        "start_t": sample["start_t"],
        "baseline_latent_distance": baseline,
        "final_latent_distance_global": final_dist,
        "final_latent_distance_local": local_dist,
        "reference_final_distance_global": reference_dist,
        "local_global_disagreement": disagreement,
        "normalized_final_distance": final_dist / max(baseline, 1e-12),
        "episode_steps": int(all_actions.shape[0]),
        "planning_wall_time_sec": time.perf_counter() - start_time,
        "mean_plan_cost_first_iter": float(
            np.mean([r["first_iter_cost"] for r in rounds if r["first_iter_cost"] is not None])
        ),
        "mean_plan_cost_final_iter": float(
            np.mean([r["planned_cost"] for r in rounds if r["planned_cost"] is not None])
        ),
        "accepted_refinement_rate": (
            float(np.mean(accepted_flags)) if accepted_flags else None
        ),
        "action_bound_violation_count": action_bound_violations(all_actions, low, high),
        **totals,
        "rounds": rounds,
    }


def compute_episode_metrics(record: dict[str, Any], success_threshold: float) -> dict[str, Any]:
    metrics = {k: v for k, v in record.items() if k != "rounds"}
    metrics["success"] = bool(record["normalized_final_distance"] < success_threshold)
    return metrics


def save_episode_artifacts(
    planner_dir: Path, index: int, record: dict[str, Any], metrics: dict[str, Any]
) -> None:
    traces_dir = planner_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    with (planner_dir / "episodes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
    with (traces_dir / f"episode_{index:03d}.jsonl").open("w", encoding="utf-8") as f:
        for round_record in record["rounds"]:
            f.write(json.dumps(round_record) + "\n")


def summarize_episodes(episodes: list[dict[str, Any]], planner_name: str) -> dict[str, Any]:
    def _mean(key: str) -> float | None:
        values = [e[key] for e in episodes if e.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "planner": planner_name,
        "num_episodes": len(episodes),
        "success_rate": _mean("success"),
        "mean_final_latent_distance_global": _mean("final_latent_distance_global"),
        "mean_final_latent_distance_local": _mean("final_latent_distance_local"),
        "mean_normalized_final_distance": _mean("normalized_final_distance"),
        "mean_reference_final_distance_global": _mean("reference_final_distance_global"),
        "mean_local_global_disagreement": _mean("local_global_disagreement"),
        "mean_planning_wall_time_sec": _mean("planning_wall_time_sec"),
        "mean_plan_cost_first_iter": _mean("mean_plan_cost_first_iter"),
        "mean_plan_cost_final_iter": _mean("mean_plan_cost_final_iter"),
        "accepted_refinement_rate": _mean("accepted_refinement_rate"),
        "total_global_forward_calls": int(sum(e["num_global_forward_calls"] for e in episodes)),
        "total_local_forward_calls": int(sum(e["num_local_forward_calls"] for e in episodes)),
        "total_backward_steps": int(sum(e["num_backward_steps"] for e in episodes)),
        "action_bound_violation_count": int(
            sum(e["action_bound_violation_count"] for e in episodes)
        ),
    }


def evaluate_planner(
    config: dict[str, Any],
    planner_name: str,
    run_dir: str | Path,
    *,
    num_episodes: int | None = None,
    device: str = "cpu",
    checkpoint: str | Path | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Evaluate one planner; writes planning/<planner>/{episodes.jsonl,summary.json,traces/}.

    ``deadline`` is a ``time.perf_counter()`` timestamp: past it no new episode
    starts and the partial results are written to ``summary_partial.json``
    instead of ``summary.json`` (the notebook's self-gating treats only
    ``summary.json`` at >= the configured episode count as done). Episodes are
    appended to ``episodes.jsonl`` one at a time, and a re-run **resumes** from
    the episodes already logged (validated against ``eval_state.json``) rather
    than repeating the planner -- so an interruption costs at most the single
    in-flight episode.
    """
    run_dir = Path(run_dir).expanduser()
    typed = typed_config(config)
    cfg: PlannerConfig = typed.planner
    store = LatentTrajectoryStore(
        latent_cache_dir(config),
        action_data_dir(config),
        max_episodes=int(get_config_value(config, "training.max_episodes", 0)),
    )
    _, val_eps = split_store_episodes(
        store,
        float(get_config_value(config, "training.val_fraction", 0.1)),
        int(get_config_value(config, "training.split_seed", 42)),
    )
    # Checkpoints trained with upstream normalize_action=true consume normalized
    # actions/proprio; the adapter applies these dataset statistics internally.
    action_stats = None
    if str(get_config_value(config, "global_model.source", "dino_wm")) == "dino_wm":
        action_stats = compute_action_state_stats(store)
    global_model = build_global_model(config, device=device, action_stats=action_stats)

    local_model = None
    needs_local = planner_name != "global_cem"
    if needs_local:
        ckpt = Path(checkpoint) if checkpoint else _default_checkpoint(run_dir)
        if ckpt is None or not ckpt.is_file():
            raise FileNotFoundError(
                f"Planner {planner_name} needs a trained local surrogate; no checkpoint "
                f"found under {run_dir / 'checkpoints'}. Run train_local_surrogate.py first."
            )
        local_model, _ = load_local_checkpoint(ckpt, device=device)

    planner = build_planner(
        planner_name, global_model=global_model, local_model=local_model, cfg=cfg, device=device
    )
    episodes_n = int(
        num_episodes
        if num_episodes is not None
        else get_config_value(config, "evaluation.num_episodes", 10)
    )
    context_len = max(typed.local_model.context_len, getattr(global_model, "context_len", 1))
    tasks = sample_episode_tasks(
        store,
        val_eps,
        context_len=context_len,
        goal_steps=cfg.goal_steps,
        frameskip=cfg.frameskip,
        num_tasks=episodes_n,
        seed=int(get_config_value(config, "evaluation.episode_seed", 0)),
    )
    threshold = float(get_config_value(config, "evaluation.success_threshold", 0.5))

    planner_dir = run_dir / "planning" / planner.name
    planner_dir.mkdir(parents=True, exist_ok=True)
    episodes_file = planner_dir / "episodes.jsonl"
    traces_dir = planner_dir / "traces"
    episode_seed = int(get_config_value(config, "evaluation.episode_seed", 0))

    # Per-episode resume across wall-clock-capped sessions. The i-th task is
    # deterministic in episode_seed (sample_episode_tasks draws sequentially),
    # so a re-run continues from the episodes already logged -- including after
    # raising num_episodes (the first N tasks are unchanged). Resume is only
    # valid when the task-defining parameters match; otherwise start fresh.
    state = {
        "planner": planner.name,
        "episode_seed": episode_seed,
        "success_threshold": threshold,
        "goal_steps": cfg.goal_steps,
        "frameskip": cfg.frameskip,
        "context_len": context_len,
        "split_seed": int(get_config_value(config, "training.split_seed", 42)),
        "val_fraction": float(get_config_value(config, "training.val_fraction", 0.1)),
        "max_episodes": int(get_config_value(config, "training.max_episodes", 0)),
    }
    state_path = planner_dir / "eval_state.json"
    prior: list[dict[str, Any]] = []
    if state_path.is_file() and episodes_file.is_file():
        try:
            if json.loads(state_path.read_text(encoding="utf-8")) == state:
                prior = [
                    json.loads(ln)
                    for ln in episodes_file.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
        except (OSError, json.JSONDecodeError):
            prior = []
    if not prior:  # fresh start or incompatible prior: clear partial artifacts
        episodes_file.unlink(missing_ok=True)
        if traces_dir.is_dir():
            for stale in traces_dir.glob("episode_*.jsonl"):
                stale.unlink()
    (planner_dir / "summary_partial.json").unlink(missing_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    all_metrics: list[dict[str, Any]] = prior[:episodes_n]
    start_index = len(all_metrics)
    if start_index:
        print(f"Resuming {planner.name} from episode {start_index}/{episodes_n}.")
    capped = False
    for index in range(start_index, episodes_n):
        if deadline is not None and time.perf_counter() > deadline:
            capped = True
            print(
                f"Wall-clock limit reached after {len(all_metrics)}/{episodes_n} episodes "
                f"of {planner.name}; partial results saved (re-run to continue)."
            )
            break
        sample = _episode_tensors(
            store,
            tasks[index],
            context_len=context_len,
            goal_steps=cfg.goal_steps,
            frameskip=cfg.frameskip,
            proprio_dim=typed.global_model.proprio_dim,
        )
        record = run_mpc_episode(
            global_model,
            planner,
            sample,
            goal_steps=cfg.goal_steps,
            exec_steps=cfg.mpc_exec_steps,
            local_model=local_model,
            local_context_len=typed.local_model.context_len,
            seed=episode_seed * 1000 + index,
        )
        metrics = compute_episode_metrics(record, threshold)
        save_episode_artifacts(planner_dir, index, record, metrics)
        all_metrics.append(metrics)
        print(
            f"{planner.name}: episode {index + 1}/{episodes_n} "
            f"success={metrics['success']} "
            f"norm_dist={metrics['normalized_final_distance']:.3f}",
            flush=True,
        )

    summary = summarize_episodes(all_metrics, planner.name)
    summary["success_threshold"] = threshold
    summary["config_task"] = typed.task
    summary["wall_time_capped"] = capped
    summary["episodes_requested"] = episodes_n
    summary["episodes_completed"] = len(all_metrics)
    if capped:
        with (planner_dir / "summary_partial.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    else:
        with (planner_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def _default_checkpoint(run_dir: Path) -> Path | None:
    for name in ("local_best.pt", "local_latest.pt"):
        candidate = run_dir / "checkpoints" / name
        if candidate.is_file():
            return candidate
    return None
