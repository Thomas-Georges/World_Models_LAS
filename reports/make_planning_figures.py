#!/usr/bin/env python3
"""Regenerate the Track II planning figures used by reports/world_models_report.tex.

These four figures (cem_success.pdf, cem_state_dist.pdf, cem_embed_div.pdf,
success_rates.pdf) are derived purely from the recorded planner telemetry
(`planning/*/logs.json`, exported as a flat CSV by the results notebook). The
committed copy of that telemetry is reports/pointmaze_planning_logs_200evals.csv.

    python reports/make_planning_figures.py reports/pointmaze_planning_logs_200evals.csv

Figures are written to reports/figures/. The two training-curve figures the
report also references (pointmaze_training_curves.png,
pointmaze_scratch_vs_finetune.png) are exported separately by results
notebook 07 and committed under reports/figures/.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"

# The science runs, in a fixed label/colour scheme (smoke/diagnostic runs are
# intentionally excluded from the report figures).
LABELS = {
    "pointmaze_full_nodecoder_t4_fp16_b32_stride2_seed0": "full data (2000 rollouts)",
    "pointmaze_lowdata_finetune_a100_seed0": "low-data fine-tune (300 rollouts)",
    "pointmaze_lowdata_scratch_a100_seed0": "low-data scratch (300 rollouts)",
}
COLORS = {
    "pointmaze_full_nodecoder_t4_fp16_b32_stride2_seed0": "#2f6f73",
    "pointmaze_lowdata_finetune_a100_seed0": "#b85c38",
    "pointmaze_lowdata_scratch_a100_seed0": "#3b5b92",
}
N_EVALS = 200  # final sample size (50 was the interim, noisier pass)


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def load(csv_path: Path):
    rows = list(csv.DictReader(csv_path.open()))
    iters: dict[tuple[str, float], dict] = {}
    final: dict[str, float] = {}
    for row in rows:
        run = row.get("run")
        if run not in LABELS:
            continue
        if row.get("step"):
            iters[(run, float(row["step"]))] = row  # latest attempt wins
        elif row.get("success_rate"):
            final[run] = float(row["success_rate"])
    series = defaultdict(list)
    for (run, step), row in sorted(iters.items()):
        series[run].append((step, row))
    return series, final


def line_figure(series, key, ylabel, outname, size=(5.0, 3.4)):
    fig, ax = plt.subplots(figsize=size)
    for run, pts in series.items():
        xs = [s for s, _ in pts]
        ys = [float(r[key]) for _, r in pts]
        ax.plot(xs, ys, marker="o", ms=3.5, color=COLORS[run], label=LABELS[run])
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / outname, dpi=200)
    plt.close(fig)
    print("wrote", outname)


def success_figure(final):
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    runs = sorted(final, key=lambda r: final[r])
    ys = list(range(len(runs)))
    vals = [final[r] for r in runs]
    lo = [final[r] - wilson(final[r], N_EVALS)[0] for r in runs]
    hi = [wilson(final[r], N_EVALS)[1] - final[r] for r in runs]
    ax.barh(ys, vals, color=[COLORS[r] for r in runs], xerr=[lo, hi], capsize=4, ecolor="#333")
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS[r] for r in runs], fontsize=8)
    for y, v, h in zip(ys, vals, hi):
        ax.text(v + h + 0.025, y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel(f"CEM planning success rate ({N_EVALS} evals)")
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "success_rates.pdf", dpi=200)
    plt.close(fig)
    print("wrote success_rates.pdf")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    series, final = load(Path(sys.argv[1]).expanduser())
    if not series or not final:
        print("No recognized science-run rows in the CSV.", file=sys.stderr)
        return 1
    line_figure(series, "plan_0/success_rate", f"success rate ({N_EVALS} evals)", "cem_success.pdf")
    line_figure(series, "plan_0/mean_state_dist", "mean state distance to goal", "cem_state_dist.pdf")
    line_figure(
        series, "plan_0/mean_div_visual_emb",
        "imagined vs executed embedding divergence", "cem_embed_div.pdf",
        size=(5.6, 3.4),
    )
    success_figure(final)
    print("final success rates:", {LABELS[k]: v for k, v in final.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
