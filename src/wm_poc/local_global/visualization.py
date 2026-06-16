"""Post-hoc loading and plotting helpers for the local/global results notebook.

Torch-free and tolerant of missing artifacts: every loader returns an empty
list/dict when files are absent, and every plotter is a no-op (returning None)
when given nothing to plot. Matplotlib is imported lazily.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

TRAIN_METRICS_FILE = "metrics/train_metrics.jsonl"
VAL_ROLLOUTS_FILE = "metrics/val_rollouts.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_train_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(run_dir) / TRAIN_METRICS_FILE)


def load_val_rollouts(run_dir: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(run_dir) / VAL_ROLLOUTS_FILE)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_planning_logs(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Per-planner episodes/summary/traces for one run directory.

    ``summary`` is the *completed* summary (``summary.json``, written only when
    a planner finishes all episodes); ``summary_partial`` is the wall-capped
    partial. ``episodes`` are the raw per-episode rows, present mid-run, from
    which a partial summary can be computed even after a hard kill.
    """
    planning = Path(run_dir) / "planning"
    out: dict[str, dict[str, Any]] = {}
    if not planning.is_dir():
        return out
    for planner_dir in sorted(p for p in planning.iterdir() if p.is_dir()):
        out[planner_dir.name] = {
            "summary": _read_json(planner_dir / "summary.json"),
            "summary_partial": _read_json(planner_dir / "summary_partial.json"),
            "episodes": _read_jsonl(planner_dir / "episodes.jsonl"),
            "trace_files": sorted((planner_dir / "traces").glob("*.jsonl"))
            if (planner_dir / "traces").is_dir()
            else [],
        }
    return out


def summarize_episode_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode metric rows the way ``eval.summarize_episodes`` does,
    but torch-free, so the results notebook can summarize a planner mid-run from
    ``episodes.jsonl`` alone (e.g. after a hard kill, before any summary)."""
    if not rows:
        return {}

    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(sum(vals) / len(vals)) if vals else None

    def total(key: str) -> int:
        return int(sum(int(r.get(key) or 0) for r in rows))

    return {
        "success_rate": mean("success"),
        "mean_final_latent_distance_global": mean("final_latent_distance_global"),
        "mean_final_latent_distance_local": mean("final_latent_distance_local"),
        "mean_normalized_final_distance": mean("normalized_final_distance"),
        "mean_reference_final_distance_global": mean("reference_final_distance_global"),
        "mean_local_global_disagreement": mean("local_global_disagreement"),
        "mean_planning_wall_time_sec": mean("planning_wall_time_sec"),
        "accepted_refinement_rate": mean("accepted_refinement_rate"),
        "total_global_forward_calls": total("num_global_forward_calls"),
        "total_local_forward_calls": total("num_local_forward_calls"),
        "total_backward_steps": total("num_backward_steps"),
        "action_bound_violation_count": total("action_bound_violation_count"),
        "episodes_completed": len(rows),
    }


def discover_runs(run_root: str | Path) -> list[Path]:
    root = Path(run_root).expanduser()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))


def aggregate_summary(run_root: str | Path) -> list[dict[str, Any]]:
    """One row per (run, planner), merged with final training metrics."""
    rows: list[dict[str, Any]] = []
    for run_dir in discover_runs(run_root):
        train_rows = load_train_metrics(run_dir)
        val_rows = [r for r in train_rows if r.get("split") == "val"]
        last_val = val_rows[-1] if val_rows else {}
        planning = load_planning_logs(run_dir)
        base = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "train_steps": max((int(r.get("step", 0)) for r in train_rows), default=0),
            "final_val_loss": last_val.get("loss_total"),
            "final_val_rollout_mse": last_val.get("loss_rollout"),
        }
        if not planning:
            rows.append({**base, "planner": None})
            continue
        for planner_name, info in planning.items():
            # Prefer the completed summary; fall back to the wall-capped partial,
            # then to summarizing the raw episodes (so a planner stopped mid-run
            # still appears, flagged complete=False with its episode count).
            complete = info["summary"] is not None
            summary = info["summary"] or info.get("summary_partial")
            if summary is None and info["episodes"]:
                summary = summarize_episode_rows(info["episodes"])
            summary = summary or {}
            episodes_completed = summary.get("episodes_completed")
            if episodes_completed is None:
                episodes_completed = len(info["episodes"]) or None
            rows.append(
                {
                    **base,
                    "planner": planner_name,
                    "complete": complete,
                    "episodes_completed": episodes_completed,
                    "episodes_requested": summary.get("episodes_requested"),
                    "success_rate": summary.get("success_rate"),
                    "mean_final_latent_distance_global": summary.get(
                        "mean_final_latent_distance_global"
                    ),
                    "mean_normalized_final_distance": summary.get(
                        "mean_normalized_final_distance"
                    ),
                    "mean_reference_final_distance_global": summary.get(
                        "mean_reference_final_distance_global"
                    ),
                    "mean_local_global_disagreement": summary.get(
                        "mean_local_global_disagreement"
                    ),
                    "mean_planning_wall_time_sec": summary.get("mean_planning_wall_time_sec"),
                    "accepted_refinement_rate": summary.get("accepted_refinement_rate"),
                    "total_global_forward_calls": summary.get("total_global_forward_calls"),
                    "total_local_forward_calls": summary.get("total_local_forward_calls"),
                    "total_backward_steps": summary.get("total_backward_steps"),
                    "action_bound_violation_count": summary.get(
                        "action_bound_violation_count"
                    ),
                }
            )
    return rows


def write_summary_csv(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["run_name"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(
    train_rows: list[dict[str, Any]], output: str | Path, *, label: str = ""
) -> Path | None:
    if not train_rows:
        return None
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    for split, style in (("train", "--"), ("val", "-")):
        rows = [r for r in train_rows if r.get("split") == split and "loss_total" in r]
        if rows:
            ax.plot(
                [r["step"] for r in rows],
                [r["loss_total"] for r in rows],
                style,
                label=f"{label} {split}".strip(),
            )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Local surrogate training loss")
    fig.tight_layout()
    out = Path(output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_rollout_errors(
    val_rows: list[dict[str, Any]], output: str | Path
) -> Path | None:
    rows = [r for r in val_rows if r.get("rollout_mse_per_step")]
    if not rows:
        return None
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    last = rows[-1]
    steps = list(range(1, len(last["rollout_mse_per_step"]) + 1))
    ax.plot(steps, last["rollout_mse_per_step"], marker="o", label=f"step {last.get('step')}")
    if len(rows) > 1:
        first = rows[0]
        ax.plot(
            list(range(1, len(first["rollout_mse_per_step"]) + 1)),
            first["rollout_mse_per_step"],
            marker="o",
            alpha=0.5,
            label=f"step {first.get('step')}",
        )
    ax.set_xlabel("rollout horizon (model steps)")
    ax.set_ylabel("val MSE (local space)")
    ax.legend()
    ax.set_title("Multi-step rollout error")
    fig.tight_layout()
    out = Path(output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_planner_bars(
    summary_rows: list[dict[str, Any]],
    output: str | Path,
    *,
    metric: str = "mean_normalized_final_distance",
) -> Path | None:
    rows = [r for r in summary_rows if r.get("planner") and r.get(metric) is not None]
    if not rows:
        return None
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['run_name']}\n{r['planner']}" for r in rows]
    ax.bar(range(len(rows)), [float(r[metric]) for r in rows])
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title("Planner comparison")
    fig.tight_layout()
    out = Path(output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def load_trace(trace_file: str | Path) -> list[dict[str, Any]]:
    """Flatten a per-episode trace file into per-iteration rows."""
    rows: list[dict[str, Any]] = []
    for round_record in _read_jsonl(Path(trace_file)):
        for entry in round_record.get("trace", []):
            rows.append({"round": round_record.get("round", 0), **entry})
    return rows


def plot_optimization_trace(
    trace_rows: list[dict[str, Any]], output: str | Path, *, title: str = ""
) -> Path | None:
    if not trace_rows:
        return None
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    stages = sorted({r.get("stage", "plan") for r in trace_rows})
    offset = 0
    for stage in stages:
        rows = [r for r in trace_rows if r.get("stage", "plan") == stage and r.get("round", 0) == 0]
        costs = [r.get("best_cost", r.get("cost")) for r in rows]
        costs = [c for c in costs if c is not None]
        if not costs:
            continue
        ax.plot(range(offset, offset + len(costs)), costs, marker=".", label=stage)
        offset += len(costs)
    ax.set_xlabel("optimization iteration")
    ax.set_ylabel("cost")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(title or "Planner optimization trace (round 0)")
    fig.tight_layout()
    out = Path(output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def find_videos(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    return sorted(run_dir.glob("planning/*/videos/*.mp4"))


def refinement_outcomes(run_dir: str | Path) -> list[dict[str, Any]]:
    """Episodes where local refinement improved/worsened/was rejected (hybrid planners)."""
    outcomes: list[dict[str, Any]] = []
    planning = load_planning_logs(run_dir)
    for planner_name, info in planning.items():
        if not planner_name.startswith("hybrid"):
            continue
        for trace_file in info["trace_files"]:
            for round_record in _read_jsonl(trace_file):
                costs = round_record.get("costs", {})
                metadata = round_record.get("metadata", {})
                if "global_rescore_cost" not in costs:
                    continue
                outcomes.append(
                    {
                        "planner": planner_name,
                        "episode_file": trace_file.name,
                        "round": round_record.get("round"),
                        "cem_global_cost": costs.get("cem_global_cost"),
                        "global_rescore_cost": costs.get("global_rescore_cost"),
                        "accepted_refinement": metadata.get("accepted_refinement"),
                        "refinement_improved_global_cost": metadata.get(
                            "refinement_improved_global_cost"
                        ),
                    }
                )
    return outcomes
