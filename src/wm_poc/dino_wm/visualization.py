from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for DINO-WM plotting scripts.") from exc
    return plt


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def plot_training_loss_curves(run_dirs: list[Path], output: Path) -> None:
    from wm_poc.dino_wm.metrics import epoch_loss_series

    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for run_dir in run_dirs:
        records = _jsonl(run_dir / "metrics.jsonl")
        xs = [record.get("epoch", record.get("step")) for record in records]
        ys = [record.get("val/loss_pred_hstep") for record in records]
        points = [(x, y) for x, y in zip(xs, ys, strict=False) if x is not None and y is not None]
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], label=run_dir.name)
            plotted = True
            continue
        # The no-decoder runs write no metrics.jsonl; their per-epoch losses
        # only exist in the upstream log line captured in stdout.log.
        series = epoch_loss_series(run_dir)
        if series:
            epochs = [record["epoch"] for record in series]
            ax.plot(epochs, [record["val_loss"] for record in series], label=f"{run_dir.name} (val)")
            ax.plot(
                epochs,
                [record["train_loss"] for record in series],
                linestyle="--",
                alpha=0.6,
                label=f"{run_dir.name} (train)",
            )
            plotted = True
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if plotted:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def display_label(run_name: str) -> str:
    """Readable chart label for a run directory name."""

    name = re.sub(r"_seed\d+$", "", run_name)
    rules = [
        ("full_nodecoder_t4", "full data (T4, stride-2)"),
        ("full_nodecoder", "full data"),
        ("lowdata_scratch", "low-data scratch"),
        ("lowdata_finetune", "low-data fine-tune"),
        ("oom_safe", "OOM diagnostic"),
        ("smoke_pointmaze_latent", "smoke (latent)"),
        ("smoke", "smoke"),
    ]
    for needle, label in rules:
        if needle in name:
            return label
    return name.replace("pointmaze_", "").replace("_", " ")


def _is_diagnostic(row: dict[str, str]) -> bool:
    name = row.get("run_name", "")
    return row.get("mode") == "smoke" or "smoke" in name or "oom_safe" in name


def prepare_planning_rows(
    rows: list[dict[str, str]], include_diagnostics: bool = False
) -> list[tuple[str, float]]:
    kept = [
        (display_label(row["run_name"]), float(row["best_success_rate"]))
        for row in rows
        if row.get("best_success_rate") and (include_diagnostics or not _is_diagnostic(row))
    ]
    return sorted(kept, key=lambda item: item[1])


def plot_planning_success(
    summary_csv: Path, output: Path, include_diagnostics: bool = False
) -> None:
    plt = _require_matplotlib()
    pairs = prepare_planning_rows(_read_summary_csv(summary_csv), include_diagnostics)
    labels = [pair[0] for pair in pairs]
    values = [pair[1] for pair in pairs]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.6 * len(labels) + 1)))
    bars = ax.barh(labels, values, color="#2f6f73")
    for bar, value in zip(bars, values, strict=False):
        ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    ax.set_xlabel("best planning success rate")
    ax.set_xlim(0, 1.05)
    ax.set_title("CEM planning success (smoke/diagnostic runs excluded)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def prepare_scratch_finetune_rows(
    rows: list[dict[str, str]],
) -> list[tuple[str, float, str]]:
    kept = []
    for row in rows:
        if row.get("mode") not in {"scratch", "finetune"} or _is_diagnostic(row):
            continue
        if not row.get("final_val_loss_pred_hstep"):
            continue
        kept.append(
            (
                display_label(row["run_name"]),
                float(row["final_val_loss_pred_hstep"]),
                str(row.get("mode")),
            )
        )
    return sorted(kept, key=lambda item: item[1])


def plot_scratch_vs_finetune(summary_csv: Path, output: Path) -> None:
    plt = _require_matplotlib()
    triples = prepare_scratch_finetune_rows(_read_summary_csv(summary_csv))
    labels = [item[0] for item in triples]
    values = [item[1] for item in triples]
    colors = ["#3b5b92" if item[2] == "scratch" else "#b85c38" for item in triples]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.6 * len(labels) + 1)))
    bars = ax.barh(labels, values, color=colors)
    span = max(values) if values else 1.0
    for bar, value in zip(bars, values, strict=False):
        ax.text(value + span * 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center")
    ax.set_xlabel("final validation h-step loss (lower is better)")
    ax.set_xlim(0, span * 1.18 if values else 1.0)
    ax.set_title("Scratch (blue) vs fine-tune (orange), diagnostics excluded")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_latent_distance_curves(run_dirs: list[Path], output: Path) -> None:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_dir in run_dirs:
        records = _jsonl(run_dir / "metrics.jsonl")
        ys = [record.get("final_goal_latent_distance") for record in records]
        points = [(index, y) for index, y in enumerate(ys) if y is not None]
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=run_dir.name)
    ax.set_xlabel("planning record")
    ax.set_ylabel("final goal latent distance")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
