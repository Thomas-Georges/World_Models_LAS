# Repository Bootstrap Specification — World-Model Predictive-Control Proof of Concept

This Markdown file is intended to be placed in the root of an empty GitHub repository and given to Codex as the implementation specification.

The goal is **not** to train or run any model yet. The goal is to create a clean, reproducible repository foundation so that future Colab runs can train and fine-tune PyTorch world models with minimal ambiguity.

Working project name:

```text
World_Models_LAS
```

Core project idea:

```text
Part 1:
DreamerV3 / R2-Dreamer fine-tuning experiment.
Train a model, save and reload checkpoints, adapt it to a shifted task or dataset, and compare fine-tuning against training from scratch.

Part 2:
Small local/global world-model planning experiment.
Use a larger latent world model for prediction/planning, train a smaller local surrogate for differentiable dynamics, and compare CEM-style zero-order planning with a first-order method through the local model.
```

The repository should be ready for future work on Google Colab with T4/A100 GPUs, but this bootstrap stage must not run experiments.

---

## 0. Codex task prompt

Use this exact prompt for the first Codex task on the empty GitHub repository.

```text
Read REPO_BOOTSTRAP_SPEC.md and create the initial repository scaffold for the world-model predictive-control proof-of-concept project.

Important constraints:
- Do not run model training.
- Do not run long GPU jobs.
- Do not download large datasets.
- Do not clone large external repositories during this Codex task unless explicitly marked as safe and optional.
- Do not commit checkpoints, datasets, videos, TensorBoard logs, model weights, or large binary artifacts.
- Do not add secrets, tokens, credentials, API keys, or local machine paths.
- Keep the repository PyTorch-oriented.
- Make all paths configurable through environment variables.
- Create scripts and notebooks that will be run later in Google Colab by the user.
- Use clear TODO comments where manual user actions are required, especially Google Drive authorization and dataset downloads.
- Add lightweight CPU-only smoke tests where useful.
- Do not silently delete or overwrite existing files.

Implement the required repository structure, files, scripts, and documentation described in REPO_BOOTSTRAP_SPEC.md.

After creating the scaffold:
- Run only lightweight syntax/format/smoke checks.
- Summarize exactly what was created.
- List anything that still requires manual action in Colab.
```

Recommended filename for this spec in the repo:

```text
REPO_BOOTSTRAP_SPEC.md
```

---

## 1. Non-goals for the bootstrap stage

The bootstrap stage must **not** do the following:

```text
No model training.
No Dreamer/R2-Dreamer execution.
No DINO-WM execution.
No CEM planning execution.
No GPU benchmark.
No dataset download unless explicitly run later by the user.
No checkpoint download unless explicitly run later by the user.
No Google Drive access from Codex Web.
No hard-coded personal paths.
No committed .pt, .pth, .ckpt, .hdf5, .npz, .mp4, TensorBoard event files, or datasets.
```

The bootstrap stage should create the scaffolding that makes those future steps easy.

---

## 2. Required repository tree

Codex should create this repository structure:

```text
World_Models_LAS/
  README.md
  AGENTS.md
  REPO_BOOTSTRAP_SPEC.md
  commands.md
  data_manifest.md
  .gitignore
  pyproject.toml
  requirements-dev.txt

  notebooks/
    00_colab_setup.ipynb
    01_r2dreamer_foundation.ipynb
    02_dino_wm_foundation.ipynb
    03_local_global_foundation.ipynb

  scripts/
    create_drive_tree.py
    verify_environment.py
    verify_drive_layout.py
    clone_external_repos.sh
    download_dino_wm_data_placeholder.sh
    collect_system_info.py
    make_empty_run_manifest.py

  src/
    wm_poc/
      __init__.py
      paths.py
      system_info.py
      manifests.py
      plotting.py
      local_global/
        __init__.py
        datasets.py
        models.py
        planners.py
        losses.py
        eval.py

  configs/
    paths.env.example
    r2dreamer/
      README.md
      base_dmc_proprio_example.yaml
      base_dmc_vision_example.yaml
    dino_wm/
      README.md
      pointmaze_example.yaml
      pusht_example.yaml
    local_global/
      README.md
      pointmaze_local_surrogate_example.yaml

  tests/
    test_paths.py
    test_manifests.py

  figures/
    .gitkeep

  reports/
    .gitkeep

  logs/
    .gitkeep
```

The `logs/`, `figures/`, and `reports/` directories are only for small human-readable files committed to GitHub. Heavy logs and experiment artifacts must go to Google Drive.

---

## 3. Project path convention

Use these environment variables everywhere.

```bash
# Local GitHub repo checked out in Colab.
export WM_POC_REPO=/content/World_Models_LAS

# Persistent Google Drive root.
export WM_POC_DRIVE_ROOT=/content/drive/MyDrive/wm_poc

# Persistent artifact locations.
export WM_POC_DATA_DIR=$WM_POC_DRIVE_ROOT/data
export WM_POC_CKPT_DIR=$WM_POC_DRIVE_ROOT/checkpoints
export WM_POC_LOG_DIR=$WM_POC_DRIVE_ROOT/logs
export WM_POC_FIG_DIR=$WM_POC_DRIVE_ROOT/figures
export WM_POC_TB_DIR=$WM_POC_DRIVE_ROOT/tensorboard
export WM_POC_VIDEO_DIR=$WM_POC_DRIVE_ROOT/videos
export WM_POC_EXTERNAL_REPOS=$WM_POC_DRIVE_ROOT/external_repos
export WM_POC_REPORT_DIR=$WM_POC_DRIVE_ROOT/reports
```

The repository should include `configs/paths.env.example` with these variables.

The active Git checkout should usually live on the Colab local disk:

```text
/content/World_Models_LAS
```

Large persistent files should live on Drive:

```text
/content/drive/MyDrive/wm_poc
```

Rationale: Colab local disk is faster for code and small file operations. Drive is better for persistence across runtime disconnects.

---

## 4. Google Drive folder tree

The setup notebook and `scripts/create_drive_tree.py` should create this Drive tree:

```text
/content/drive/MyDrive/wm_poc/
  data/
    dino_wm/
    dmc/
    metaworld/
    robomimic_optional/
    libero_optional/

  checkpoints/
    r2dreamer/
    dino_wm/
    local_global/

  logs/
    r2dreamer/
    dino_wm/
    local_global/
    system/

  figures/
    r2dreamer/
    dino_wm/
    local_global/

  tensorboard/
    r2dreamer/
    dino_wm/
    local_global/

  videos/
    r2dreamer/
    dino_wm/
    local_global/

  external_repos/
    r2dreamer/
    dino_wm/
    jepa_wms_optional/

  reports/
```

The script should be idempotent: running it multiple times must not delete data.

---

## 5. Required `.gitignore`

Create a `.gitignore` that excludes:

```gitignore
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/

# Environments
.venv/
venv/
env/
.env
.env.*
!.env.example
*.local

# Large artifacts
*.pt
*.pth
*.ckpt
*.safetensors
*.h5
*.hdf5
*.npz
*.npy
*.pkl
*.joblib
*.mp4
*.mov
*.avi
*.webm
*.gif
*.wandb
events.out.tfevents*
wandb/
tensorboard/
lightning_logs/
runs/

# Data/checkpoints/logs
data/
checkpoints/
external_repos/
large_artifacts/
outputs/
results/
tmp/

# Colab/OS
.DS_Store
.Trashes
desktop.ini

# Keep directory placeholders
!figures/.gitkeep
!reports/.gitkeep
!logs/.gitkeep
```

Small curated plots may be committed under `figures/`. Raw experiment artifacts should not be committed.

---

## 6. `AGENTS.md` requirements

Create `AGENTS.md` with the following guidance for future Codex sessions:

```markdown
# AGENTS.md

## Project goal

This repository supports a PyTorch proof of concept for world models, predictive control, and reinforcement learning.

The intended experiments are:

1. DreamerV3 / R2-Dreamer fine-tuning:
   - train a model,
   - save/reload checkpoints,
   - adapt to a shifted task or dataset,
   - compare fine-tuning against training from scratch.

2. Local/global world-model planning:
   - use a larger latent world model for prediction/planning,
   - train a smaller local surrogate for differentiable dynamics,
   - compare CEM-style zero-order planning with first-order planning through the local model.

## Hard constraints

- Do not run long training jobs unless explicitly asked.
- Do not download large datasets unless explicitly asked.
- Do not commit datasets, checkpoints, TensorBoard logs, videos, or model weights.
- Keep code PyTorch-based.
- Keep paths configurable through environment variables.
- Save large artifacts to Google Drive, not GitHub.
- Prefer small smoke tests before long runs.
- Never include secrets, tokens, API keys, or personal credentials.

## Code style

- Use typed Python where practical.
- Use `pathlib.Path` for filesystem paths.
- Make scripts runnable from the repository root.
- Prefer explicit command-line arguments.
- Keep functions small and testable.
- Use clear error messages when expected folders or environment variables are missing.

## Validation

Before proposing changes, run lightweight checks only:

```bash
python scripts/verify_environment.py --cpu-only
python scripts/verify_drive_layout.py --dry-run
pytest -q
```

Do not run GPU training as part of ordinary Codex tasks.
```

This file is important because future Codex sessions should read it as repository-specific instructions.

---

## 7. `README.md` requirements

Create a README with this structure:

```markdown
# World-Model Predictive-Control Proof of Concept

This repository is a PyTorch-oriented project scaffold for demonstrating hands-on experience with world models, GPU training, fine-tuning, and predictive control.

## Project tracks

### Track 1 — DreamerV3 / R2-Dreamer fine-tuning

Goal:
Train a DreamerV3/R2-Dreamer-style world model, save and reload checkpoints, adapt it to a shifted task or dataset, and compare fine-tuning against training from scratch.

Candidate starting tasks:
- DMC Proprio: `walker_walk -> walker_run`
- DMC Vision: `walker_walk -> walker_run`
- Optional later: Meta-World task shift

Primary external repo:
- https://github.com/NM512/r2dreamer

### Track 2 — Local/global world-model planning

Goal:
Use a larger latent world model for prediction/planning, train a smaller local surrogate for differentiable dynamics, and compare CEM-style zero-order planning with first-order planning through the local model.

Candidate starting tasks:
- PointMaze first
- PushT second

Primary external repo:
- https://github.com/gaoyuezhou/dino_wm

Optional reference repo:
- https://github.com/facebookresearch/jepa-wms

## What this repository contains

- Colab setup notebooks.
- Drive folder setup scripts.
- External repo clone scripts.
- Dataset preparation placeholders.
- Lightweight utility modules.
- Local/global planning scaffolding.
- Experiment command templates.

## What this repository does not contain

- Datasets.
- Checkpoints.
- Videos.
- TensorBoard logs.
- Model weights.
- Full training outputs.

## Setup overview

1. Clone this repository into Colab.
2. Mount Google Drive.
3. Create the Drive artifact tree.
4. Clone external repositories into Drive or `/content`.
5. Install dependencies for the selected external repo.
6. Verify GPU/PyTorch/CUDA availability.
7. Only then run model-specific experiments.

## Colab entry point

Open:

```text
notebooks/00_colab_setup.ipynb
```

The notebook should only prepare the environment and folder structure. It should not train any model.

## Artifact policy

Use GitHub for:
- code,
- configuration,
- documentation,
- small plots,
- short logs,
- reproducibility scripts.

Use Google Drive for:
- datasets,
- checkpoints,
- long logs,
- videos,
- TensorBoard runs,
- large generated artifacts.

## Current status

Bootstrap stage only. No model has been trained yet.
```

---

## 8. `commands.md` requirements

Create a `commands.md` file containing future command templates, but mark them as **not to be run during bootstrap**.

It should include:

````markdown
# Command Templates

These commands are templates for future Colab runs. They should not be run during repository bootstrap.

## 1. Clone this repository in Colab

```bash
cd /content
git clone https://github.com/Thomas-Georges/World_Models_LAS.git
cd World_Models_LAS
```

## 2. Mount Drive in Colab

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 3. Create Drive tree

```bash
python scripts/create_drive_tree.py \
  --drive-root /content/drive/MyDrive/wm_poc
```

## 4. Verify environment

```bash
python scripts/verify_environment.py
```

## 5. Clone external repos

```bash
bash scripts/clone_external_repos.sh \
  --external-root /content/drive/MyDrive/wm_poc/external_repos
```

## 6. Future R2-Dreamer smoke test

Do not run until dependencies are installed.

```bash
cd /content/drive/MyDrive/wm_poc/external_repos/r2dreamer

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

python3 train.py \
  logdir=/content/drive/MyDrive/wm_poc/logs/r2dreamer/debug_dmc_proprio \
  env=dmc_proprio \
  env.task=dmc_walker_walk \
  model.rep_loss=dreamer \
  trainer.steps=2000
```

## 7. Future DINO-WM setup

Do not run until datasets are downloaded and dependencies are installed.

```bash
cd /content/drive/MyDrive/wm_poc/external_repos/dino_wm
python train.py --help
python plan.py --help
```
````

Use the repository URL shown above, or replace it with a fork URL if needed.

---

## 9. `data_manifest.md` requirements

Create a `data_manifest.md` file with this content structure:

````markdown
# Data Manifest

This repository does not store datasets.

## Track 1 — DreamerV3 / R2-Dreamer

R2-Dreamer/DreamerV3 online RL experiments collect data through environment interaction.

Initial environments:
- DMC Proprio
- DMC Vision
- Optional: Meta-World

No static dataset is required for the first DMC runs.

## Track 2 — DINO-WM / local-global planning

DINO-WM-style experiments use offline trajectory datasets.

Candidate datasets:
- PointMaze
- PushT
- Wall
- Optional later: Rope
- Optional later: Granular

Expected Drive location:

```text
/content/drive/MyDrive/wm_poc/data/dino_wm/
```

Expected subfolders, once downloaded:

```text
point_maze/
pusht_noise/
wall_single/
rope/
granular/
```

## Dataset policy

- Do not commit datasets to GitHub.
- Store datasets under Google Drive or another persistent artifact store.
- Record dataset source URL, download date, checksum if available, and local path.
- Prefer official dataset links from the upstream repositories.
````

---

## 10. `notebooks/00_colab_setup.ipynb` requirements

Create a valid Jupyter notebook with these sections.

### Section 1 — Project overview

Markdown cell:

```markdown
# Colab Setup — World-Model Predictive-Control Proof

This notebook prepares the Colab runtime and Google Drive folders for future experiments. It does not train models.
```

### Section 2 — Check GPU

Code cell:

```python
import os
import sys
import subprocess
from pathlib import Path

print("Python:", sys.version)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
except Exception as exc:
    print("PyTorch check failed:", repr(exc))

subprocess.run(["nvidia-smi"], check=False)
```

### Section 3 — Mount Google Drive

Code cell:

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Section 4 — Create Drive folder tree

Code cell:

```python
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/wm_poc")

subdirs = [
    "data/dino_wm",
    "data/dmc",
    "data/metaworld",
    "data/robomimic_optional",
    "data/libero_optional",
    "checkpoints/r2dreamer",
    "checkpoints/dino_wm",
    "checkpoints/local_global",
    "logs/r2dreamer",
    "logs/dino_wm",
    "logs/local_global",
    "logs/system",
    "figures/r2dreamer",
    "figures/dino_wm",
    "figures/local_global",
    "tensorboard/r2dreamer",
    "tensorboard/dino_wm",
    "tensorboard/local_global",
    "videos/r2dreamer",
    "videos/dino_wm",
    "videos/local_global",
    "external_repos/r2dreamer",
    "external_repos/dino_wm",
    "external_repos/jepa_wms_optional",
    "reports",
]

for subdir in subdirs:
    (DRIVE_ROOT / subdir).mkdir(parents=True, exist_ok=True)

print(f"Drive tree ready at: {DRIVE_ROOT}")
```

### Section 5 — Define environment variables

Code cell:

```python
import os

os.environ["WM_POC_REPO"] = "/content/World_Models_LAS"
os.environ["WM_POC_DRIVE_ROOT"] = "/content/drive/MyDrive/wm_poc"
os.environ["WM_POC_DATA_DIR"] = "/content/drive/MyDrive/wm_poc/data"
os.environ["WM_POC_CKPT_DIR"] = "/content/drive/MyDrive/wm_poc/checkpoints"
os.environ["WM_POC_LOG_DIR"] = "/content/drive/MyDrive/wm_poc/logs"
os.environ["WM_POC_FIG_DIR"] = "/content/drive/MyDrive/wm_poc/figures"
os.environ["WM_POC_TB_DIR"] = "/content/drive/MyDrive/wm_poc/tensorboard"
os.environ["WM_POC_VIDEO_DIR"] = "/content/drive/MyDrive/wm_poc/videos"
os.environ["WM_POC_EXTERNAL_REPOS"] = "/content/drive/MyDrive/wm_poc/external_repos"

for key in [
    "WM_POC_REPO",
    "WM_POC_DRIVE_ROOT",
    "WM_POC_DATA_DIR",
    "WM_POC_CKPT_DIR",
    "WM_POC_LOG_DIR",
    "WM_POC_EXTERNAL_REPOS",
]:
    print(key, "=", os.environ[key])
```

### Section 6 — Clone this repo

Markdown cell:

```markdown
If this notebook is opened directly from GitHub, the repository may not be cloned into `/content`. Use the cell below, or replace the URL if using a fork.
```

Code cell:

```bash
%%bash
set -e

cd /content

if [ ! -d World_Models_LAS ]; then
  git clone https://github.com/Thomas-Georges/World_Models_LAS.git
else
  echo "Repository already exists at /content/World_Models_LAS"
fi

cd /content/World_Models_LAS
git status --short
```

### Section 7 — Verify repository scripts

Code cell:

```bash
%%bash
set -e
cd /content/World_Models_LAS

python scripts/verify_environment.py --cpu-only
python scripts/verify_drive_layout.py --drive-root /content/drive/MyDrive/wm_poc
```

### Section 8 — External repo clone commands

Markdown cell explaining that this is optional and should only be run when ready.

Code cell:

```bash
%%bash
set -e

cd /content/World_Models_LAS

bash scripts/clone_external_repos.sh \
  --external-root /content/drive/MyDrive/wm_poc/external_repos
```

### Section 9 — Next steps

Markdown cell:

```markdown
At this point the foundation is ready. The next phase is to install dependencies for the selected external repo and run a small smoke test. Do not start long training until the smoke test passes.
```

---

## 11. Other notebooks

Create valid notebooks with markdown-only skeletons for now.

### `notebooks/01_r2dreamer_foundation.ipynb`

Sections:

```text
# R2-Dreamer / DreamerV3 Foundation

Purpose:
Prepare future DreamerV3/R2-Dreamer training and fine-tuning runs.

This notebook will eventually:
1. Install r2dreamer dependencies.
2. Run a tiny DMC smoke test.
3. Train source task.
4. Save checkpoint.
5. Reload checkpoint.
6. Fine-tune target task.
7. Train target task from scratch.
8. Plot fine-tune vs scratch.
```

Do not include active training cells yet except clearly commented command templates.

### `notebooks/02_dino_wm_foundation.ipynb`

Sections:

```text
# DINO-WM Foundation

Purpose:
Prepare future DINO-WM latent predictive-control experiments.

This notebook will eventually:
1. Verify DINO-WM data location.
2. Install DINO-WM dependencies.
3. Run train.py --help and plan.py --help.
4. Train or load a PointMaze model.
5. Run CEM planning.
6. Export latent trajectories for local surrogate training.
```

### `notebooks/03_local_global_foundation.ipynb`

Sections:

```text
# Local/Global World-Model Planning Foundation

Purpose:
Prepare future local/global planning experiments.

This notebook will eventually:
1. Load latent trajectory tuples.
2. Train local surrogate dynamics.
3. Compare CEM global planning, gradient local planning, and hybrid planning.
4. Produce planning metrics and figures.
```

---

## 12. Script requirements

### 12.1 `scripts/create_drive_tree.py`

Requirements:

- Python script.
- Uses `argparse`.
- Accepts `--drive-root`.
- Creates the Drive folder tree.
- Does not delete anything.
- Prints created/existing directories.
- Exits successfully if directories already exist.

Expected usage:

```bash
python scripts/create_drive_tree.py \
  --drive-root /content/drive/MyDrive/wm_poc
```

### 12.2 `scripts/verify_environment.py`

Requirements:

- Python script.
- Accepts `--cpu-only`.
- Prints:
  - Python version,
  - platform,
  - current working directory,
  - PyTorch version if installed,
  - CUDA availability if PyTorch is installed,
  - GPU name if CUDA is available,
  - `nvidia-smi` output if available.
- Does not fail if PyTorch is missing unless `--require-torch` is added.
- Does not require a GPU when `--cpu-only` is used.

Expected usage:

```bash
python scripts/verify_environment.py --cpu-only
python scripts/verify_environment.py
```

### 12.3 `scripts/verify_drive_layout.py`

Requirements:

- Python script.
- Uses `argparse`.
- Accepts:
  - `--drive-root`,
  - `--dry-run`.
- Checks whether the expected Drive tree exists.
- In dry-run mode, prints what would be checked/created.
- Should not delete anything.

Expected usage:

```bash
python scripts/verify_drive_layout.py \
  --drive-root /content/drive/MyDrive/wm_poc
```

### 12.4 `scripts/clone_external_repos.sh`

Requirements:

- Bash script.
- Accepts `--external-root`.
- Creates the external repo root if needed.
- Clones these repositories if missing:
  - `https://github.com/NM512/r2dreamer.git`
  - `https://github.com/gaoyuezhou/dino_wm.git`
- Optionally includes a commented-out line for:
  - `https://github.com/facebookresearch/jepa-wms.git`
- If a repo already exists, print its current commit and skip cloning.
- Do not install dependencies.
- Do not run training.

Expected usage:

```bash
bash scripts/clone_external_repos.sh \
  --external-root /content/drive/MyDrive/wm_poc/external_repos
```

### 12.5 `scripts/download_dino_wm_data_placeholder.sh`

Requirements:

- Bash script.
- Contains comments and TODOs.
- Does not download anything by default.
- Explains where DINO-WM datasets should go.
- Prints the expected folder structure.
- Points the user to the upstream DINO-WM README for the official dataset link.

Expected usage:

```bash
bash scripts/download_dino_wm_data_placeholder.sh \
  --data-root /content/drive/MyDrive/wm_poc/data/dino_wm
```

### 12.6 `scripts/collect_system_info.py`

Requirements:

- Python script.
- Writes system metadata to JSON.
- Accepts:
  - `--output`.
- Captures:
  - timestamp,
  - platform,
  - Python version,
  - PyTorch version if installed,
  - CUDA availability,
  - CUDA version,
  - GPU name,
  - current git commit if inside a git repo.
- Useful later for reports.

Expected usage:

```bash
python scripts/collect_system_info.py \
  --output /content/drive/MyDrive/wm_poc/logs/system/system_info.json
```

### 12.7 `scripts/make_empty_run_manifest.py`

Requirements:

- Python script.
- Creates a JSON or YAML run manifest template for future experiments.
- Accepts:
  - `--output`,
  - `--track`,
  - `--run-name`.
- Does not require actual model outputs.

Fields:

```json
{
  "run_name": "",
  "track": "",
  "date": "",
  "git_commit": "",
  "external_repo": "",
  "external_repo_commit": "",
  "gpu": "",
  "environment": "",
  "task_source": "",
  "task_target": "",
  "model_config": "",
  "checkpoint_input": "",
  "checkpoint_output": "",
  "logdir": "",
  "notes": ""
}
```

---

## 13. Python package scaffolding

The `src/wm_poc/` package should be lightweight.

### 13.1 `src/wm_poc/paths.py`

Should provide:

```python
from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: str | None = None) -> Path:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return Path(value).expanduser().resolve()


def repo_root() -> Path:
    return env_path("WM_POC_REPO", default=".")


def drive_root() -> Path:
    return env_path("WM_POC_DRIVE_ROOT", default="/content/drive/MyDrive/wm_poc")


def data_dir() -> Path:
    return env_path("WM_POC_DATA_DIR", default=str(drive_root() / "data"))


def log_dir() -> Path:
    return env_path("WM_POC_LOG_DIR", default=str(drive_root() / "logs"))


def checkpoint_dir() -> Path:
    return env_path("WM_POC_CKPT_DIR", default=str(drive_root() / "checkpoints"))
```

### 13.2 `src/wm_poc/system_info.py`

Should include helper functions for collecting Python/PyTorch/GPU/git metadata.

### 13.3 `src/wm_poc/manifests.py`

Should include helper functions to create JSON manifests.

### 13.4 `src/wm_poc/local_global/models.py`

Create placeholder PyTorch modules only, no training code yet.

Suggested minimal classes:

```python
class LocalDynamics(nn.Module):
    """Small differentiable latent dynamics surrogate.

    Intended future use:
    z_{t+1} = z_t + f_phi(z_t, a_t)
    """
```

### 13.5 `src/wm_poc/local_global/planners.py`

Create placeholder planner interfaces only.

Suggested functions:

```python
def cem_plan_placeholder(*args, **kwargs):
    raise NotImplementedError("CEM planning will be implemented after the foundation setup.")

def gradient_plan_placeholder(*args, **kwargs):
    raise NotImplementedError("Gradient-based local planning will be implemented after the foundation setup.")
```

Do not implement full planners during bootstrap unless asked.

---

## 14. Config files

Create simple placeholder YAML files.

### 14.1 `configs/r2dreamer/base_dmc_proprio_example.yaml`

```yaml
track: r2dreamer
external_repo: r2dreamer
env: dmc_proprio
source_task: dmc_walker_walk
target_task: dmc_walker_run
model_rep_loss: dreamer
bootstrap_only: true
notes: "Template only. Do not run training during repository bootstrap."
```

### 14.2 `configs/r2dreamer/base_dmc_vision_example.yaml`

```yaml
track: r2dreamer
external_repo: r2dreamer
env: dmc_vision
source_task: dmc_walker_walk
target_task: dmc_walker_run
model_rep_loss: dreamer
bootstrap_only: true
notes: "Template only. Do not run training during repository bootstrap."
```

### 14.3 `configs/dino_wm/pointmaze_example.yaml`

```yaml
track: dino_wm
external_repo: dino_wm
env: point_maze
planner: cem
bootstrap_only: true
notes: "Template only. Dataset must be downloaded manually before training/planning."
```

### 14.4 `configs/local_global/pointmaze_local_surrogate_example.yaml`

```yaml
track: local_global
dataset: point_maze_latents
global_model_source: dino_wm
local_model_type: residual_mlp
planner_baselines:
  - cem_global
  - gradient_local
  - hybrid_cem_gradient
bootstrap_only: true
notes: "Template only. No model training during bootstrap."
```

---

## 15. `pyproject.toml` and dev requirements

Create a minimal `pyproject.toml` for local utility code, not for external training repos.

Suggested content:

```toml
[project]
name = "wm-poc"
version = "0.0.1"
description = "Scaffolding for a PyTorch world-model predictive-control proof of concept."
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
dev = [
  "pytest",
  "ruff",
  "nbformat",
]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

Create `requirements-dev.txt`:

```text
pytest
ruff
nbformat
```

Do not add heavyweight model dependencies here. External repos manage their own dependencies.

---

## 16. Tests

Create lightweight tests only.

### `tests/test_paths.py`

Should test that path helpers return `Path` objects and respect environment variables.

### `tests/test_manifests.py`

Should test manifest template creation.

No GPU required.

No external repos required.

No Google Drive required.

Run:

```bash
pytest -q
```

---

## 17. Validation checklist for Codex

After Codex creates the scaffold, it should run only lightweight commands:

```bash
python scripts/verify_environment.py --cpu-only
python scripts/verify_drive_layout.py --drive-root /tmp/wm_poc_test --dry-run
python scripts/create_drive_tree.py --drive-root /tmp/wm_poc_test
python scripts/verify_drive_layout.py --drive-root /tmp/wm_poc_test
pytest -q
```

If `ruff` is installed:

```bash
ruff check .
```

Codex should not run:

```bash
python train.py
python plan.py
pip install -e ".[dmc]"
pip install -r requirements.txt
wget large_dataset
gdown large_dataset
```

unless explicitly instructed in a later task.

---

## 18. Manual user checklist after Codex scaffold is merged

After Codex creates the repo scaffold and you merge or accept the changes:

### Step 1 — Open the setup notebook in Colab

Use the GitHub/Colab URL pattern:

```text
https://colab.research.google.com/github/Thomas-Georges/World_Models_LAS/blob/main/notebooks/00_colab_setup.ipynb
```

Replace the GitHub path if using a fork.

### Step 2 — Select a GPU runtime

In Colab:

```text
Runtime -> Change runtime type -> GPU
```

T4 is enough for foundation checks. A100 is preferable for later model runs.

### Step 3 — Run only foundation cells

Run cells that:

```text
check GPU
mount Drive
create Drive folder tree
define environment variables
clone this repo into /content
verify repository scripts
```

Do not run training.

### Step 4 — Confirm Drive tree exists

Check:

```text
/content/drive/MyDrive/wm_poc/
```

Expected subfolders:

```text
data/
checkpoints/
logs/
figures/
tensorboard/
videos/
external_repos/
reports/
```

### Step 5 — Clone external repos only when ready

When ready, run:

```bash
bash scripts/clone_external_repos.sh \
  --external-root /content/drive/MyDrive/wm_poc/external_repos
```

### Step 6 — Dataset preparation later

For DMC/R2-Dreamer, no static dataset is required at the start because online RL collects data through environment interaction.

For DINO-WM/local-global planning, download official DINO-WM datasets later into:

```text
/content/drive/MyDrive/wm_poc/data/dino_wm/
```

Expected subfolders:

```text
point_maze/
pusht_noise/
wall_single/
rope/
granular/
```

---

## 19. Future milestones after foundation setup

These are future tasks, not part of bootstrap.

### Milestone A — R2-Dreamer smoke test

Goal:
Run a tiny DMC Proprio test to verify dependencies and GPU.

Output:
A short log only.

### Milestone B — R2-Dreamer fine-tuning baseline

Goal:
Run:

```text
source task training
target task fine-tuning from source checkpoint
target task scratch baseline
```

Output:
Fine-tune-vs-scratch curve.

### Milestone C — DINO-WM PointMaze baseline

Goal:
Verify DINO-WM dataset, train/load model, run CEM planning.

Output:
Planning success/final-distance table.

### Milestone D — Local surrogate

Goal:
Train a small differentiable latent dynamics model on latent tuples.

Output:
One-step/multi-step prediction loss.

### Milestone E — CEM vs first-order planning

Goal:
Compare:

```text
CEM through global model
gradient planning through local surrogate
hybrid CEM + gradient refinement
```

Output:
Planning-time and final-distance comparison.

---

## 20. External references for maintainers

Primary external repositories:

```text
R2-Dreamer:
https://github.com/NM512/r2dreamer

DINO-WM:
https://github.com/gaoyuezhou/dino_wm

Optional JEPA-WM reference:
https://github.com/facebookresearch/jepa-wms
```

Operational references:

```text
Codex AGENTS.md guidance:
https://developers.openai.com/codex/guides/agents-md

Codex cloud environments:
https://developers.openai.com/codex/cloud/environments

Colab GitHub integration:
https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb

Colab Drive IO:
https://colab.research.google.com/notebooks/io.ipynb
```

---

## 21. Acceptance criteria

The repository foundation is complete when:

```text
README.md explains the project clearly.
AGENTS.md gives future Codex sessions strict project rules.
.gitignore prevents large artifacts from being committed.
notebooks/00_colab_setup.ipynb is valid and opens in Colab.
scripts/create_drive_tree.py can create the Drive folder tree locally under /tmp.
scripts/verify_environment.py runs without GPU.
scripts/verify_drive_layout.py checks the folder tree.
scripts/clone_external_repos.sh is present but does not install dependencies or train.
scripts/download_dino_wm_data_placeholder.sh documents future dataset setup but does not download by default.
configs/ contains placeholder experiment templates.
src/wm_poc contains lightweight path/system/manifest utilities.
tests pass with pytest.
No large artifacts are committed.
No training has been run.
No dataset has been downloaded by Codex.
```

Final Codex response should include:

```text
Files created.
Checks run.
Checks passed/failed.
Manual steps left for the user in Colab.
Risks or assumptions.
```

---

## 22. Suggested next Codex task after bootstrap

After the bootstrap scaffold is complete, use a second Codex task:

```text
Read AGENTS.md, README.md, and commands.md.

Prepare the repository for the first future R2-Dreamer smoke test, but do not run training.

Add:
- a script that patches or wraps r2dreamer checkpoint loading if needed,
- a command template for DMC Proprio smoke testing,
- a run manifest template for source/fine-tune/scratch,
- a plotting placeholder for future fine-tune-vs-scratch curves.

Do not clone external repos, install dependencies, or train models.
```

This keeps the work incremental and reviewable.
