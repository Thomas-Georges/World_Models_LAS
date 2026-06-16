# Release checklist

Run from a clean checkout before a final handoff. See `REPRODUCIBILITY.md` for
the full command details.

- [ ] `git status --short` is clean.
- [ ] External revisions are pinned to exact SHAs in `external_revisions.lock`
      (`pinned: true`); report-relevant configs carry the same SHA, no `commit: main`.
- [ ] `make verify` passes (compile + tests + notebook-error-check + release markers).
- [ ] `make report` builds the final PDF from a clean aux state (no undefined
      references, no "Rerun to get cross-references" warnings, no overfull boxes).
- [ ] All report CSVs parse with consistent columns (`tests/test_report_csvs.py`,
      run as part of `make test`).
- [ ] `python scripts/check_notebook_errors.py` reports no error outputs.
- [ ] `reports/artifact_manifest.json` points every claim at a committed file or a
      clearly labelled external/Drive log, consistent with `docs/notebook_policy.md`
      (foundation notebooks 01/02/03 are cleared and are NOT the committed source).
- [ ] Final archive built with `git archive` (not a raw directory zip).
- [ ] Archive cleanliness check returns no cache/metadata files.

## Commands

```bash
git status --short                                   # expect empty

make verify                                          # compile + test + nb-check + markers
make report                                          # final PDF (latexmk / 3 pdflatex passes)

# Report CSVs parse (also covered by make test)
python - <<'PY'
import csv
from pathlib import Path
for p in sorted(Path("reports").glob("*.csv")):
    with p.open(newline="") as f:
        rows = list(csv.reader(f))
    n = len(rows[0])
    for i, r in enumerate(rows, 1):
        assert len(r) == n, f"{p}:{i}: {len(r)} != {n}"
    print("OK", p)
PY

# Build + check the release archive (from the committed tree only)
git archive --format=zip --prefix=wm-prediction/ --output ../wm-prediction-release.zip HEAD
zipinfo -1 ../wm-prediction-release.zip | \
  grep -E '(^|/)\.git/|__MACOSX|\.DS_Store|__pycache__|\.pyc$|\.pytest_cache|\.ruff_cache|\.review_pkgs' \
  && echo "ARCHIVE NOT CLEAN" || echo "archive clean"
```
