# Reproducibility checklist

One documented sequence to verify repository health and rebuild the report. All
commands run from the repository root. The test suite pins BLAS/torch threads via
`tests/conftest.py`, so no hidden environment variables are needed.

## Quick health check

```bash
make all      # compile + test + notebook-error-check
```

or run the steps individually:

```bash
# 1. Source compiles
python -m compileall -q src scripts tests reports

# 2. Tests pass (torch-backed modules skip automatically without torch installed;
#    install a track extra to run them -- see pyproject.toml).
python -m pytest -q

# 3. No notebook carries a stale error output
python scripts/check_notebook_errors.py
```

## Rebuild the report and its figures

```bash
# Figures (Track II planning + Track III table) from committed telemetry
python reports/make_planning_figures.py reports/pointmaze_planning_logs_200evals.csv
python reports/make_planning_table.py   reports/local_global_planning_summary.csv

# PDF -- canonical build target (uses latexmk when available, else repeated
# pdflatex passes, so cross-references settle correctly)
make report
```

Expected: no undefined references, no serious overfull boxes.

## Tracing a headline number

Every headline value in the report is traceable via
`reports/artifact_manifest.json` (claim -> report location, value, source
notebook, committed log/CSV, run name, seed) and the flat summaries
`reports/r2dreamer_summary.csv`, `reports/dino_wm_pointmaze_summary.csv`,
`reports/local_global_planning_summary.csv`.

## Pinned upstream code

`external_revisions.lock` records the upstream remotes and the commit each track
should use. Pin the exact report-run SHAs there (recorded in the Drive
run-metadata archives) before reproducing numbers; see the lockfile header.

## Notebook + artifact policy

- Notebook output conventions: `docs/notebook_policy.md`
  (training notebooks 01/02/03 cleared; results notebooks 06/07/08 keep final
  outputs; no error cells anywhere).
- Datasets, checkpoints, videos, and TensorBoard logs are intentionally **not**
  in git (see `.gitignore` and the README "Artifact policy").

## Release archive

Build a clean archive from tracked files only (no `.git`, caches, or OS cruft):

```bash
git archive --format=zip --output ../wm-prediction-release.zip HEAD
```
