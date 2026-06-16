from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from wm_poc.local_global.configs import (  # noqa: E402
    action_data_dir,
    latent_cache_dir,
    load_local_global_config,
    resolve_run_dir,
    typed_config,
)
from wm_poc.local_global.datasets import ensure_synthetic_task_data  # noqa: E402
from wm_poc.local_global.eval import evaluate_planner, sample_episode_tasks  # noqa: E402
from wm_poc.local_global.global_models import SyntheticPointGlobalModel, build_global_model  # noqa: E402
from wm_poc.local_global.models import build_local_model, save_local_checkpoint  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = REPO_ROOT / "configs" / "local_global" / "smoke_synthetic.yaml"


@pytest.fixture()
def smoke_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("LG_SMOKE_ROOT", str(tmp_path))
    config = load_local_global_config(SMOKE_CONFIG)
    ensure_synthetic_task_data(config)
    run_dir = resolve_run_dir(config, tmp_path / "smoke_run")
    typed = typed_config(config)
    build_kwargs = dict(
        patches=typed.global_model.latent_patches,
        embed_dim=typed.global_model.latent_dim,
        step_action_dim=typed.planner.step_action_dim,
        model_type=typed.local_model.model_type,
        projection=typed.local_model.projection,
        projection_grid=typed.local_model.projection_grid,
        projection_trainable=typed.local_model.projection_trainable,
        local_dim=typed.local_model.local_dim,
        hidden_dim=typed.local_model.hidden_dim,
        num_layers=typed.local_model.num_layers,
        layer_norm=typed.local_model.layer_norm,
        seed=0,
    )
    model = build_local_model(**build_kwargs)
    save_local_checkpoint(
        run_dir / "checkpoints" / "local_latest.pt", model, build_kwargs, step=0
    )
    return config, run_dir


def test_synthetic_global_model_is_exact(smoke_setup):
    config, _ = smoke_setup
    model = build_global_model(config)
    assert isinstance(model, SyntheticPointGlobalModel)
    state = torch.tensor([[0.1, -0.2, 0.05, 0.0]])
    z = model.encode_state(state)
    torch.testing.assert_close(model.decode_state(z), state, atol=1e-4, rtol=1e-4)


def test_synthetic_global_model_is_device_consistent(smoke_setup):
    # Regression: build_global_model must propagate device to the synthetic
    # model, and the model must co-locate its buffers and outputs there, even
    # when fed CPU inputs. (On a GPU box device=cuda; here we can only assert
    # the invariant on cpu, but the wiring it guards is the cross-device fix.)
    config, _ = smoke_setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_global_model(config, device=device)
    assert model.device == torch.device(device)
    assert model.weight.device == model.device and model.pinv.device == model.device
    z_context = torch.zeros(1, model.patches, model.embed_dim)  # deliberately on cpu
    state = model.init_state(z_context)
    assert state["state"].device == model.device
    # Candidates handed in on cpu must not raise and must land on model.device.
    candidates = torch.zeros(3, 2, model.action_dim * model.frameskip)
    z_final = model.rollout_final(state, candidates)
    assert z_final.device == model.device
    advanced = model.advance(state, candidates[0])
    assert advanced["step_latents"].device == model.device


def test_sample_episode_tasks_deterministic(smoke_setup):
    config, _ = smoke_setup
    from wm_poc.dino_wm.configs import get_config_value
    from wm_poc.local_global.datasets import LatentTrajectoryStore, split_store_episodes

    store = LatentTrajectoryStore(latent_cache_dir(config), action_data_dir(config))
    _, val_eps = split_store_episodes(store, 0.1, 42)
    kwargs = dict(context_len=2, goal_steps=4, frameskip=2, num_tasks=3, seed=1)
    tasks_a = sample_episode_tasks(store, val_eps, **kwargs)
    tasks_b = sample_episode_tasks(store, val_eps, **kwargs)
    assert tasks_a == tasks_b
    assert all(t["episode"] in val_eps for t in tasks_a)
    assert get_config_value(config, "planning.goal_steps") >= 1


def test_evaluate_global_cem_writes_artifacts(smoke_setup):
    config, run_dir = smoke_setup
    summary = evaluate_planner(config, "global_cem", run_dir, num_episodes=2)
    planner_dir = run_dir / "planning" / "global_cem"
    assert (planner_dir / "summary.json").is_file()
    episodes = [
        json.loads(line)
        for line in (planner_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert len(episodes) == 2
    assert summary["num_episodes"] == 2
    assert summary["success_rate"] is not None
    assert summary["mean_reference_final_distance_global"] is not None
    assert summary["total_global_forward_calls"] > 0
    assert (planner_dir / "traces" / "episode_000.jsonl").is_file()
    for key in ("final_latent_distance_global", "normalized_final_distance", "success"):
        assert key in episodes[0]
    # With an exact global model and a reachable goal, the CEM planner should
    # at least not end farther from the goal than it started, on average.
    assert summary["mean_normalized_final_distance"] < 1.5


def test_evaluate_hybrid_with_untrained_surrogate(smoke_setup):
    config, run_dir = smoke_setup
    summary = evaluate_planner(
        config, "hybrid_cem_local_refine_global_rescore", run_dir, num_episodes=1
    )
    assert summary["accepted_refinement_rate"] is not None
    assert summary["total_backward_steps"] > 0
    assert (run_dir / "planning" / summary["planner"] / "summary.json").is_file()


def _episode_lines(run_dir, planner):
    f = run_dir / "planning" / planner / "episodes.jsonl"
    return f.read_text().splitlines() if f.is_file() else []


def test_evaluate_resumes_and_extends(smoke_setup):
    config, run_dir = smoke_setup
    s2 = evaluate_planner(config, "global_cem", run_dir, num_episodes=2)
    assert s2["episodes_completed"] == 2
    first_two = _episode_lines(run_dir, "global_cem")
    assert len(first_two) == 2

    # Raising the episode count resumes: the first 2 lines are byte-identical
    # (deterministic tasks, appended not rewritten) and 2 more are added.
    s4 = evaluate_planner(config, "global_cem", run_dir, num_episodes=4)
    assert s4["episodes_completed"] == 4 and s4["episodes_requested"] == 4
    all_four = _episode_lines(run_dir, "global_cem")
    assert len(all_four) == 4
    assert all_four[:2] == first_two


def test_rerun_same_n_does_not_duplicate(smoke_setup):
    config, run_dir = smoke_setup
    evaluate_planner(config, "global_cem", run_dir, num_episodes=2)
    evaluate_planner(config, "global_cem", run_dir, num_episodes=2)
    assert len(_episode_lines(run_dir, "global_cem")) == 2


def test_deadline_writes_partial_then_completes(smoke_setup):
    import time

    config, run_dir = smoke_setup
    planner_dir = run_dir / "planning" / "global_cem"
    # Deadline already past -> no episode runs, partial summary only.
    s = evaluate_planner(
        config, "global_cem", run_dir, num_episodes=3, deadline=time.perf_counter()
    )
    assert s["wall_time_capped"] is True
    assert (planner_dir / "summary_partial.json").is_file()
    assert not (planner_dir / "summary.json").is_file()
    # Re-run without a deadline completes and clears the partial marker.
    s2 = evaluate_planner(config, "global_cem", run_dir, num_episodes=3)
    assert s2["wall_time_capped"] is False and s2["episodes_completed"] == 3
    assert (planner_dir / "summary.json").is_file()
    assert not (planner_dir / "summary_partial.json").is_file()


def test_evaluate_local_planner_requires_checkpoint(smoke_setup, tmp_path):
    config, _ = smoke_setup
    bare_run = tmp_path / "bare_run"
    (bare_run / "checkpoints").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="local surrogate"):
        evaluate_planner(config, "local_adam", bare_run, num_episodes=1)
