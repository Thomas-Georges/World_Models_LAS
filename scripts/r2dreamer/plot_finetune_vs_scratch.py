#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.r2dreamer.metrics import numeric_series, read_jsonl, score_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fine-tune vs scratch eval score.")
    parser.add_argument("--finetune", type=Path, required=True, help="Fine-tune metrics.jsonl.")
    parser.add_argument("--scratch", type=Path, required=True, help="Scratch metrics.jsonl.")
    parser.add_argument("--out", type=Path, required=True, help="Output image path.")
    parser.add_argument("--pdf", type=Path, help="Optional PDF output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print("ERROR: matplotlib is required for plotting.", file=sys.stderr)
        print(repr(exc), file=sys.stderr)
        return 1

    ft_records = read_jsonl(args.finetune)
    scratch_records = read_jsonl(args.scratch)
    ft_key = score_key(ft_records)
    scratch_key = score_key(scratch_records)
    if ft_key != "episode/eval_score" or scratch_key != "episode/eval_score":
        print("WARNING: falling back to episode/score for at least one run.", file=sys.stderr)

    ft_steps, ft_scores = numeric_series(ft_records, ft_key)
    scratch_steps, scratch_scores = numeric_series(scratch_records, scratch_key)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ft_steps, ft_scores, label="target_finetune")
    ax.plot(scratch_steps, scratch_scores, label="target_scratch")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation return")
    ax.set_title("Fine-tune vs scratch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.expanduser(), dpi=160)
    if args.pdf:
        args.pdf.expanduser().parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf.expanduser())
    print(f"Wrote {args.out.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
