# Repository health + reproducibility targets. See REPRODUCIBILITY.md.
# Thread limits are baked into tests/conftest.py, so `make test` needs no env vars.

.PHONY: all compile test notebook-error-check report planning-figures planning-table verify clean-latex

all: compile test notebook-error-check

compile:
	python -m compileall -q src scripts tests reports

test:
	python -m pytest -q

notebook-error-check:
	python scripts/check_notebook_errors.py

# Deterministic build: latexmk runs as many passes as needed for cross-refs;
# falls back to three pdflatex passes (enough to settle labels) if latexmk is
# unavailable.
report:
	cd reports && ( command -v latexmk >/dev/null 2>&1 \
		&& latexmk -pdf -interaction=nonstopmode world_models_report.tex \
		|| ( pdflatex -interaction=nonstopmode world_models_report.tex \
			&& pdflatex -interaction=nonstopmode world_models_report.tex \
			&& pdflatex -interaction=nonstopmode world_models_report.tex ) )

planning-figures:
	python reports/make_planning_figures.py reports/pointmaze_planning_logs_200evals.csv

planning-table:
	python reports/make_planning_table.py reports/local_global_planning_summary.csv

# Full verification sweep used before a handoff (mirrors REPRODUCIBILITY.md).
# `test` already parses the report CSVs (tests/test_report_csvs.py); the marker
# check flags only genuine unfinished markers, not prose about placeholders.
verify: compile test notebook-error-check
	python scripts/check_release_markers.py

clean-latex:
	rm -f reports/*.aux reports/*.log reports/*.out reports/*.toc \
		reports/*.fls reports/*.fdb_latexmk reports/*.synctex.gz
