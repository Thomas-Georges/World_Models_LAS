# Notebook output policy

This repository commits notebooks under two different conventions depending on
their role. The split exists so a reviewer can tell, at a glance, which notebook
outputs are *authoritative report artifacts* and which are just a runnable
recipe.

| Notebook | Role | Committed with outputs? |
|---|---|---|
| `00_colab_setup.ipynb` | environment/folder setup | cleared |
| `01_r2dreamer_foundation.ipynb` | Track I training | **cleared** |
| `02_dino_wm_foundation.ipynb` | Track II training/eval | **cleared** |
| `03_local_global_foundation.ipynb` | Track III training/eval | **cleared** |
| `06_r2dreamer_results.ipynb` | Track I figures/results | **clean final outputs** |
| `07_dino_wm_results.ipynb` | Track II figures/results | **clean final outputs** |
| `08_local_global_results.ipynb` | Track III figures/results | **clean final outputs** |

## Rules

1. **Training notebooks (01/02/03): clear outputs before commit.** They drive
   long, GPU-bound runs whose real artifacts (checkpoints, logs) live on Drive,
   not in the cell outputs. Committing them as cleared recipes keeps the repo
   small and avoids stale/interrupted outputs. Clear with either
   *Kernel → Restart & Clear Output* or:

   ```bash
   jupyter nbconvert --clear-output --inplace notebooks/0{1,2,3}_*.ipynb
   ```

2. **Results notebooks (06/07/08): commit clean, final outputs.** The report
   reads every figure and number from these committed outputs, so they must be
   run to completion with no error cells. Re-run end to end before committing.

3. **No error outputs anywhere.** Enforced by:

   ```bash
   python scripts/check_notebook_errors.py
   ```

   which fails if any committed notebook has a cell with an `error` output.

4. **Gate long-running or environment-specific cells.** Cells that may not run
   in every environment (e.g. the optional Push-T plot in `07`, which is future
   work) must guard rather than fail — e.g. `if 'name' in globals():` or an
   explicit `RUN_*` env flag — and print a clear message when skipped.

See also `README.md` ("Reproducing the report") and the per-track docs in this
directory.
