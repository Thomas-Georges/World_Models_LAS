# Reports

## Authoritative final report

- **`world_models_report.tex`** / **`world_models_report.pdf`** — the single
  combined report covering all three tracks (R2-Dreamer transfer, DINO-WM
  reproduction + low-data adaptation, and local/global planning). This is the
  only report that should be cited or handed off.

Build it with `make report` (uses `latexmk` when available, otherwise repeated
`pdflatex` passes so cross-references settle). Every figure it embeds lives under
`figures/`, and every number is read from the committed outputs of results
notebooks 06/07/08 (see `../docs/notebook_policy.md`).

## Provenance artifacts

- `artifact_manifest.json` — maps every headline number in the report to its
  source (notebook, committed log/CSV, run name, seed). Start here to trace a
  claim.
- `r2dreamer_summary.csv`, `dino_wm_pointmaze_summary.csv`,
  `local_global_planning_summary.csv` — flat, one-row-per-metric summaries.
- `pointmaze_planning_logs_200evals.csv` (+ the 50-eval interim) — raw Track II
  CEM planner telemetry; `make_planning_figures.py` builds the Track II planning
  figures from it.
- `make_planning_table.py` — regenerates the Track III planner table from
  `local_global_planning_summary.csv`.
- `figures/` — final figures embedded in the report (see `figures/README.md` for
  per-figure provenance).

## Superseded drafts (`drafts/`)

The standalone per-track drafts that the combined report replaced are kept under
`drafts/` for history only. They are **not** maintained and may contain `TBD`
placeholders:

- `drafts/r2dreamer_report.tex` — folded into Track I.
- `drafts/dino_wm_report.tex`, `drafts/dino_wm_pointmaze_report.tex` — folded
  into Tracks II–III.
- `drafts/local_global_report.tex` — folded into Track III.
- `drafts/r2dreamer_results_template.md` — early results template.

Do not cite anything under `drafts/`.
