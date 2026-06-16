# World Models for Control across the Representation Spectrum

A PyTorch study of how to **obtain** and how to **use** a world model for control
under a strict commodity-GPU (free-tier Colab) budget, at two opposite choices of
latent representation. The project is complete: it contains trained-model results,
figures, and a single combined report.

**Final report:** [`reports/world_models_report.tex`](reports/world_models_report.tex)
/ `reports/world_models_report.pdf` (see [`reports/README.md`](reports/README.md)).

## The three tracks

| Track | What it studies | Representation | Task | Upstream |
|---|---|---|---|---|
| **I — R2-Dreamer** | task-transfer fine-tuning vs scratch | learned end-to-end (reconstruction-free Barlow latent) | DMC vision `walker_walk → walker_run` | [NM512/r2dreamer](https://github.com/NM512/r2dreamer) |
| **II — DINO-WM** | same-domain low-data adaptation; latent-cache training | frozen, pretrained (DINOv2 patch features) | PointMaze goal reaching | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) |
| **III — local/global** | cheap first-order planning against the trusted DINO-WM model via a differentiable surrogate | (uses Track II's model) | PointMaze latent goal reaching | (this repo) |

Two findings recur across the spectrum: (i) fine-tuning beats scratch at equal
**target-task** budget; (ii) the scalar metric a model is trained on understates a
structural latent difference that governs control. See the report for the full
argument and caveats.

## Reproducing the report

Every figure and number in the report is read from the committed outputs of the
**results** notebooks (06/07/08). The **foundation** notebooks (01/02/03) drive
the (GPU-bound) training/eval runs and are committed as cleared recipes — see
[`docs/notebook_policy.md`](docs/notebook_policy.md).

| Notebook / file | Maps to |
|---|---|
| `notebooks/01_r2dreamer_foundation.ipynb` | Track I training |
| `notebooks/06_r2dreamer_results.ipynb` | Track I figures/results |
| `notebooks/02_dino_wm_foundation.ipynb` | Track II training/evaluation |
| `notebooks/07_dino_wm_results.ipynb` | Track II figures/results |
| `notebooks/03_local_global_foundation.ipynb` | Track III training/evaluation |
| `notebooks/08_local_global_results.ipynb` | Track III figures/results |
| `reports/world_models_report.tex` / `.pdf` | final combined report |

Provenance for headline numbers is in
[`reports/artifact_manifest.json`](reports/artifact_manifest.json) and the summary
CSVs alongside it. Regenerate the Track II planning figures and the Track III table:

```bash
python reports/make_planning_figures.py reports/pointmaze_planning_logs_200evals.csv
python reports/make_planning_table.py reports/local_global_planning_summary.csv
```

## Environments per track

The experiment *definition* (model sizes, data budgets, episode counts, seeds) is
identical across GPU tiers; only throughput knobs differ. External code is used as
**pristine upstream checkouts with reversible runtime patches** (never vendored or
edited in place) — see [`docs/`](docs) and the implementation specs.

- **Track I (R2-Dreamer):** Colab **Python 3.11** (upstream requires `>=3.11,<3.12`).
  A100 for the `dmc_vision`/`size25M` pillar; T4 for the `dmc_proprio`/`size12M` pillar.
- **Tracks II–III (DINO-WM, local/global):** free-tier **T4** (16 GB, fp16) by
  default; A100-40 GB / L4 (bf16) opportunistically. The enabling optimization is
  the latent cache (≈21× faster, `docs/dino_wm_latent_cache_training.md`).
- **Tests + tooling:** any **Python ≥ 3.10**. The report-relevant upstreams
  (R2-Dreamer, DINO-WM) are pinned to exact commits in
  [`external_revisions.lock`](external_revisions.lock) and the setup scripts check
  those out by default (`jepa-wms` is pinned too but is an optional reference, not
  used for any report number). Export `DINO_WM_COMMIT` / edit a config's
  `r2dreamer.commit` to override for exploratory work.

Install the dependencies for what you want to run (heavy GPU stacks are *not*
installed by default):

```bash
pip install -e '.[dev]'           # CPU test suite, linter, report/figure scripts
pip install -e '.[r2dreamer]'     # Track I
pip install -e '.[dino]'          # Track II
pip install -e '.[local-global]'  # Track III
```

External **datasets and checkpoints are never stored in git** (see "Artifact
policy"). Place them on Google Drive and point the setup scripts at them:
`scripts/dino_wm/setup_dino_wm.sh` (clones the upstream repo to `DINO_WM_REPO`),
`scripts/r2dreamer/` (R2-Dreamer), and the per-track `configs/`.

## Tests and checks

The suite pins BLAS/torch threads to 1 via `tests/conftest.py`, so no hidden
environment variables are needed on a many-core machine. A `Makefile` wraps the
full set:

```bash
make test                  # python -m pytest -q
make compile               # python -m compileall -q src scripts tests
make notebook-error-check  # fail if any notebook has a stale error output
make report                # rebuild the PDF (needs pdflatex/tectonic)
```

The torch-backed tests skip automatically where torch is not installed (the CPU
subset stays green); install a track extra to run them. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full verification sequence.

## Colab entry point

Open [`notebooks/00_colab_setup.ipynb`](notebooks/00_colab_setup.ipynb) first; it
only prepares the environment and folder structure and trains nothing. Then open
the foundation notebook for the track you want. The foundation notebooks are
self-gating and resumable; long-running cells are separated and guarded (e.g.
training scripts print their command and exit without `RUN_TRAINING=1`):

```bash
python scripts/r2dreamer/build_commands.py --dry-run
bash scripts/r2dreamer/run_smoke.sh
```

If the GitHub repository is private, the notebooks prompt for a fine-grained,
read-only personal access token when cloning into the Colab runtime.

## Run presets (Track I)

| Config | GPU | Obs | Model | Rep loss | Source/Target steps |
|---|---|---|---|---|---|
| `configs/r2dreamer/debug_walker_walk_to_run.yaml` | any | `dmc_proprio` | `size12M` | `dreamer` | 100K / 50K |
| `configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml` | T4 | `dmc_proprio` | `size12M` | `r2dreamer` | 510K / 250K |
| `configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml` | A100 | `dmc_vision` | `size25M` | `r2dreamer` | 800K / 400K |

The report uses the A100 `dmc_vision`/`size25M` preset (Track I). Presets use
parallel env workers by default; a serial fallback is available with
`R2_SERIAL_ENVS=true`.

## Artifact policy

Stored in git: code, configuration, documentation, the final report `.tex`/`.pdf`,
small figures, short telemetry CSVs, and reproducibility scripts.

Stored on Google Drive (intentionally **not** in git): datasets, checkpoints,
model weights, videos, TensorBoard runs, and large generated artifacts. The
`.gitignore` enforces this.

## Documentation

- Notebook policy: [`docs/notebook_policy.md`](docs/notebook_policy.md)
- Reproducibility checklist: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- Track I: [`docs/r2dreamer_finetune.md`](docs/r2dreamer_finetune.md), `DREAMER_R2_FINETUNE_SPEC.md`
- Tracks II–III: [`docs/dino_wm_latent_cache_training.md`](docs/dino_wm_latent_cache_training.md),
  [`docs/local_global_methodology.md`](docs/local_global_methodology.md),
  `LOCAL_GLOBAL_DINO_WM_IMPLEMENTATION_SPEC.md`
