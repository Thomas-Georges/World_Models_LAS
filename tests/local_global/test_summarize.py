from __future__ import annotations

import json

import pytest

from wm_poc.local_global.visualization import (
    aggregate_summary,
    discover_runs,
    find_videos,
    load_planning_logs,
    load_train_metrics,
    load_trace,
    refinement_outcomes,
    write_summary_csv,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture()
def fake_run_root(tmp_path):
    root = tmp_path / "runs"
    run_a = root / "run_a"
    _write_jsonl(
        run_a / "metrics" / "train_metrics.jsonl",
        [
            {"step": 10, "split": "train", "loss_total": 1.0, "loss_rollout": 0.8},
            {"step": 20, "split": "val", "loss_total": 0.5, "loss_rollout": 0.4},
        ],
    )
    planner = run_a / "planning" / "global_cem"
    (planner / "videos").mkdir(parents=True)
    (planner / "summary.json").write_text(
        json.dumps(
            {
                "planner": "global_cem",
                "num_episodes": 2,
                "success_rate": 0.5,
                "mean_final_latent_distance_global": 0.1,
                "mean_normalized_final_distance": 0.4,
                "mean_planning_wall_time_sec": 1.5,
                "total_global_forward_calls": 100,
                "total_local_forward_calls": 0,
                "total_backward_steps": 0,
                "action_bound_violation_count": 0,
            }
        )
    )
    _write_jsonl(planner / "episodes.jsonl", [{"episode": 0, "success": True}])
    hybrid = run_a / "planning" / "hybrid_cem_local_refine_global_rescore"
    hybrid.mkdir(parents=True)
    _write_jsonl(
        hybrid / "traces" / "episode_000.jsonl",
        [
            {
                "round": 0,
                "trace": [
                    {"iter": 0, "best_cost": 1.0, "stage": "global_cem"},
                    {"iter": 0, "cost": 0.8, "stage": "local_refine"},
                ],
                "costs": {
                    "cem_global_cost": 0.5,
                    "global_rescore_cost": 0.7,
                },
                "metadata": {
                    "accepted_refinement": False,
                    "refinement_improved_global_cost": False,
                },
            }
        ],
    )
    # run_b: empty run dir (no metrics, no planning)
    (root / "run_b").mkdir(parents=True)
    # run_c: planner dir without summary.json
    (root / "run_c" / "planning" / "local_adam").mkdir(parents=True)
    # _summary must be excluded from discovery
    (root / "_summary").mkdir(parents=True)
    return root


def test_discover_runs_excludes_underscore(fake_run_root):
    names = [p.name for p in discover_runs(fake_run_root)]
    assert names == ["run_a", "run_b", "run_c"]


def test_discover_runs_missing_root(tmp_path):
    assert discover_runs(tmp_path / "nope") == []


def test_load_train_metrics_missing(tmp_path):
    assert load_train_metrics(tmp_path) == []


def test_load_planning_logs(fake_run_root):
    logs = load_planning_logs(fake_run_root / "run_a")
    assert set(logs) == {"global_cem", "hybrid_cem_local_refine_global_rescore"}
    assert logs["global_cem"]["summary"]["success_rate"] == 0.5
    assert len(logs["global_cem"]["episodes"]) == 1
    # Planner dir without summary.json must not crash
    logs_c = load_planning_logs(fake_run_root / "run_c")
    assert logs_c["local_adam"]["summary"] is None


def test_aggregate_summary_handles_all_run_shapes(fake_run_root):
    rows = aggregate_summary(fake_run_root)
    by_run = {}
    for row in rows:
        by_run.setdefault(row["run_name"], []).append(row)
    assert by_run["run_b"][0]["planner"] is None
    assert by_run["run_c"][0]["planner"] == "local_adam"
    cem_rows = [r for r in by_run["run_a"] if r["planner"] == "global_cem"]
    assert cem_rows[0]["success_rate"] == 0.5
    assert cem_rows[0]["final_val_loss"] == 0.5
    assert cem_rows[0]["complete"] is True  # has summary.json


def test_aggregate_summary_surfaces_wall_capped_partial(tmp_path):
    # Wall-capped: summary_partial.json (full metrics, no summary.json yet).
    root = tmp_path / "runs"
    planner = root / "run_x" / "planning" / "global_cem"
    planner.mkdir(parents=True)
    (planner / "summary_partial.json").write_text(
        json.dumps(
            {
                "planner": "global_cem",
                "success_rate": 0.6,
                "mean_normalized_final_distance": 0.4,
                "total_global_forward_calls": 6000,
                "episodes_requested": 100,
                "episodes_completed": 60,
                "wall_time_capped": True,
            }
        )
    )
    row = next(r for r in aggregate_summary(root) if r.get("planner") == "global_cem")
    assert row["complete"] is False
    assert row["episodes_completed"] == 60
    assert row["episodes_requested"] == 100
    assert row["success_rate"] == 0.6


def test_aggregate_summary_computes_from_episodes_after_hard_kill(tmp_path):
    # Hard kill leaves episodes.jsonl but no summary of any kind.
    root = tmp_path / "runs"
    planner = root / "run_y" / "planning" / "global_cem"
    _write_jsonl(
        planner / "episodes.jsonl",
        [
            {"success": True, "normalized_final_distance": 0.2, "num_global_forward_calls": 100},
            {"success": False, "normalized_final_distance": 0.8, "num_global_forward_calls": 100},
            {"success": True, "normalized_final_distance": 0.1, "num_global_forward_calls": 100},
        ],
    )
    row = next(r for r in aggregate_summary(root) if r.get("planner") == "global_cem")
    assert row["complete"] is False
    assert row["episodes_completed"] == 3
    assert row["episodes_requested"] is None
    assert row["success_rate"] == pytest.approx(2 / 3)
    assert row["total_global_forward_calls"] == 300


def test_write_summary_csv(fake_run_root, tmp_path):
    rows = aggregate_summary(fake_run_root)
    out = write_summary_csv(rows, tmp_path / "_summary" / "summary.csv")
    text = out.read_text()
    assert "run_a" in text and "success_rate" in text


def test_write_summary_csv_empty(tmp_path):
    out = write_summary_csv([], tmp_path / "summary.csv")
    assert out.is_file()


def test_find_videos_empty(fake_run_root):
    assert find_videos(fake_run_root / "run_a") == []
    assert find_videos(fake_run_root / "missing") == []


def test_load_trace_and_refinement_outcomes(fake_run_root):
    hybrid_dir = fake_run_root / "run_a" / "planning" / "hybrid_cem_local_refine_global_rescore"
    rows = load_trace(hybrid_dir / "traces" / "episode_000.jsonl")
    assert len(rows) == 2 and rows[0]["stage"] == "global_cem"
    outcomes = refinement_outcomes(fake_run_root / "run_a")
    assert len(outcomes) == 1
    assert outcomes[0]["accepted_refinement"] is False
    assert outcomes[0]["global_rescore_cost"] == 0.7


def test_plots_with_and_without_data(fake_run_root, tmp_path):
    pytest.importorskip("matplotlib")
    from wm_poc.local_global.visualization import (
        plot_optimization_trace,
        plot_planner_bars,
        plot_rollout_errors,
        plot_training_curves,
    )

    train_rows = load_train_metrics(fake_run_root / "run_a")
    out = plot_training_curves(train_rows, tmp_path / "figs" / "train.png", label="run_a")
    assert out is not None and out.is_file()
    assert plot_training_curves([], tmp_path / "figs" / "none.png") is None

    rows = aggregate_summary(fake_run_root)
    bars = plot_planner_bars(rows, tmp_path / "figs" / "bars.png", metric="success_rate")
    assert bars is not None and bars.is_file()
    assert plot_planner_bars([], tmp_path / "figs" / "nobars.png") is None

    hybrid_dir = fake_run_root / "run_a" / "planning" / "hybrid_cem_local_refine_global_rescore"
    trace_rows = load_trace(hybrid_dir / "traces" / "episode_000.jsonl")
    trace = plot_optimization_trace(trace_rows, tmp_path / "figs" / "trace.png")
    assert trace is not None and trace.is_file()
    assert plot_rollout_errors([], tmp_path / "figs" / "rollout.png") is None
