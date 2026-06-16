# STEP 2 — DreamerV3 / R2-Dreamer fine-tuning track

This document is a Codex-ready implementation specification for the next project stage.

The repository foundations should already exist from Step 1. This Step 2 should add the infrastructure needed to run, later in Colab, a PyTorch DreamerV3/R2-Dreamer training and fine-tuning experiment.

The aim is not to train inside Codex. The aim is to make the repository capable of launching and documenting the experiment cleanly.

---

## 0. Codex task prompt

Copy this prompt into Codex while the working directory is the GitHub repository root.

```text
Read DREAMER_R2_FINETUNE_SPEC.md and implement the repository changes it describes.

Goal:
Prepare the DreamerV3/R2-Dreamer fine-tuning track. This is the first real experiment track of the project.

Hard constraints:
- Do not train a model.
- Do not launch long GPU jobs.
- Do not download large datasets.
- Do not commit checkpoints, replay buffers, videos, TensorBoard event files, or model weights.
- Do not vendor the external r2dreamer repository into this repo.
- Keep this repo as an orchestration layer: scripts, configs, notebooks, patches, result parsers, plots, and documentation.
- Make all paths configurable through environment variables.
- Every training command must be printed or written into scripts, not executed by Codex.
- Add lightweight tests that run on CPU only.

Main external dependency:
- NM512/r2dreamer, a PyTorch R2-Dreamer implementation that also includes an efficient PyTorch DreamerV3 reproduction.

Implementation tasks:
1. Add a dedicated docs/spec file for the R2-Dreamer experiment track.
2. Add Colab notebook skeleton: notebooks/01_r2dreamer_foundation.ipynb.
3. Add scripts under scripts/r2dreamer/ for cloning, installing, patching, smoke testing, command generation, launching runs, parsing metrics, plotting results, and verifying checkpoints.
4. Add patching support for r2dreamer so that its train.py can load a pretrained checkpoint before continuing training.
5. Add run configs under configs/r2dreamer/.
6. Add result templates under reports/.
7. Update README.md and commands.md with the new experiment track.
8. Add tests for command generation, metrics parsing, checkpoint verification, and path handling.
9. Run only lightweight checks:
   - python scripts/verify_environment.py --cpu-only
   - python scripts/r2dreamer/build_commands.py --dry-run
   - python scripts/r2dreamer/verify_checkpoint.py --help
   - python scripts/r2dreamer/parse_metrics.py --help
   - pytest -q

Stop when the repository is ready for me to open Colab and manually run the DreamerV3/R2-Dreamer setup notebook.
```

Optional `/goal` version:

```text
/goal Implement the Step 2 DreamerV3/R2-Dreamer fine-tuning track described in DREAMER_R2_FINETUNE_SPEC.md.

Stop only when:
- notebooks/01_r2dreamer_foundation.ipynb exists and is valid JSON.
- scripts/r2dreamer/ contains setup, patch, command generation, run launchers, checkpoint verification, metrics parsing, plotting, and metadata scripts.
- configs/r2dreamer/ contains smoke, proprio, vision, and fine-tune templates.
- reports/r2dreamer_results_template.md exists.
- README.md and commands.md mention the DreamerV3/R2-Dreamer track.
- CPU-only tests pass or failures are clearly explained.

Hard constraints:
- Do not train.
- Do not run GPU jobs.
- Do not download datasets.
- Do not commit large artifacts.
- Do not access Google Drive from Codex Web.
```

Use `/plan` first if the repo has evolved substantially:

```text
/plan Read DREAMER_R2_FINETUNE_SPEC.md and propose the file changes needed for Step 2. Do not edit yet.
```

---

## 1. Experiment purpose

This track should demonstrate that I can:

1. Use a real PyTorch world-model codebase.
2. Run a DreamerV3-style baseline through `r2dreamer`.
3. Train on a source task.
4. Save a checkpoint.
5. Reload the checkpoint.
6. Fine-tune on a shifted target task.
7. Train the target task from scratch under the same budget.
8. Compare fine-tuning versus scratch training with plots and run metadata.

The professor-facing claim should eventually be:

```text
I trained a PyTorch DreamerV3/R2-Dreamer world-model agent, saved and reloaded checkpoints, fine-tuned the agent on a shifted task, and compared transfer against training from scratch using the same model size and compute budget.
```

This is not a benchmark reproduction. It is a controlled GPU/PyTorch world-model manipulation experiment.

---

## 2. External reference facts to encode in docs

Use the following current assumptions about `NM512/r2dreamer`.

Source repository:

```text
https://github.com/NM512/r2dreamer
```

Important details:

- The repository provides a PyTorch implementation of R2-Dreamer.
- It also includes an efficient PyTorch DreamerV3 reproduction.
- The README says the repo is tested with Ubuntu 24.04 and Python 3.11.
- Environment extras include `dmc`, `atari`, `crafter`, `metaworld`, `memorymaze`, `isaaclab`, and `all`.
- Example install command:

```bash
pip install -e ".[dmc,metaworld]"
```

- Example default training command:

```bash
python3 train.py logdir=./logdir/test
```

- Algorithm selection uses:

```bash
model.rep_loss=r2dreamer
model.rep_loss=dreamer
model.rep_loss=infonce
model.rep_loss=dreamerpro
```

- Benchmark selection uses Hydra-style overrides such as:

```bash
env=dmc_vision env.task=dmc_walker_walk
```

- MuJoCo/DMC/Meta-World headless rendering may need:

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

- Current `train.py` constructs a `Dreamer` agent and saves `latest.pt` containing at least:

```python
{
    "agent_state_dict": agent.state_dict(),
    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
}
```

- The repository includes `tools.recursively_load_optim_state_dict`, so optimizer resume support can be added carefully.
- Available model config files currently include sizes such as `size12M`, `size25M`, `size50M`, `size100M`, `size200M`, and `size400M`.

Do not assume these details are frozen forever. The setup script should print the checked-out commit hash.

---

## 3. Recommended experimental design

Start with a small and reliable DMC Proprio experiment before trying image observations.

### 3.1 Primary source and target tasks

Use this first:

```text
Source task: dmc_walker_walk
Target task: dmc_walker_run
Observation mode: dmc_proprio
Model size: size12M
Algorithm setting: model.rep_loss=dreamer
```

Reasoning:

- Same domain and compatible action/observation spaces.
- Shift is meaningful but not too large.
- Low-dimensional observation mode is easier to debug.
- It tests checkpoint loading and transfer before expensive image rendering.

### 3.2 Secondary image-based version

After the low-dimensional version works:

```text
Source task: dmc_walker_walk
Target task: dmc_walker_run
Observation mode: dmc_vision
Model size: size12M first, size25M/size50M later if A100 permits
Algorithm setting: model.rep_loss=dreamer
```

This version is more aligned with visual world models.

### 3.3 Optional R2-Dreamer comparison

After the DreamerV3-style baseline works:

```text
model.rep_loss=r2dreamer
```

This lets the report say that the same code path can run the Dreamer-style baseline and the newer R2-Dreamer representation objective.

### 3.4 Minimum run set

The key comparison requires three runs:

```text
A. source_base:
   Train source task from scratch.
   Example: walker_walk from scratch.

B. target_finetune:
   Initialize from source_base/latest.pt.
   Train target task.
   Example: walker_run initialized from walker_walk.

C. target_scratch:
   Train target task from random initialization.
   Same target task, same model size, same environment, same steps as target_finetune.
```

The key figure is:

```text
target_finetune vs target_scratch
evaluation return versus environment steps
```

---

## 4. Repository additions required

Codex should create or update the following files.

```text
notebooks/
  01_r2dreamer_foundation.ipynb

scripts/
  r2dreamer/
    README.md
    setup_r2dreamer.sh
    clone_r2dreamer.sh
    install_r2dreamer.sh
    patch_checkpoint_loading.py
    verify_r2dreamer_patch.py
    verify_checkpoint.py
    build_commands.py
    run_smoke.sh
    run_source_base.sh
    run_target_finetune.sh
    run_target_scratch.sh
    run_three_way_experiment.sh
    parse_metrics.py
    plot_finetune_vs_scratch.py
    summarize_runs.py
    collect_gpu_info.py
    archive_run_metadata.py

configs/
  r2dreamer/
    smoke_dmc_proprio.yaml
    dreamer_dmc_proprio_12m.yaml
    dreamer_dmc_vision_12m.yaml
    r2dreamer_dmc_proprio_12m.yaml
    r2dreamer_dmc_vision_12m.yaml
    three_way_walker_walk_to_run.yaml

patches/
  r2dreamer/
    README.md
    checkpoint_loading_notes.md

reports/
  r2dreamer_results_template.md

tests/
  test_r2dreamer_command_builder.py
  test_r2dreamer_metrics_parser.py
  test_r2dreamer_checkpoint_verifier.py
```

If some equivalent files already exist from Step 1, extend them rather than duplicating functionality.

---

## 5. Environment variables

All scripts must respect these environment variables and have safe defaults.

```bash
export WM_POC_REPO=${WM_POC_REPO:-/content/wm-prediction}
export WM_POC_DRIVE_ROOT=${WM_POC_DRIVE_ROOT:-/content/drive/MyDrive/wm_poc}

export WM_POC_DATA_DIR=${WM_POC_DATA_DIR:-$WM_POC_DRIVE_ROOT/data}
export WM_POC_LOG_DIR=${WM_POC_LOG_DIR:-$WM_POC_DRIVE_ROOT/logs}
export WM_POC_CKPT_DIR=${WM_POC_CKPT_DIR:-$WM_POC_DRIVE_ROOT/checkpoints}
export WM_POC_FIGURE_DIR=${WM_POC_FIGURE_DIR:-$WM_POC_DRIVE_ROOT/figures}
export WM_POC_EXTERNAL_REPOS=${WM_POC_EXTERNAL_REPOS:-/content/external_repos}

export R2DREAMER_REPO=${R2DREAMER_REPO:-$WM_POC_EXTERNAL_REPOS/r2dreamer}
export R2DREAMER_REMOTE=${R2DREAMER_REMOTE:-https://github.com/NM512/r2dreamer.git}
export R2DREAMER_COMMIT=${R2DREAMER_COMMIT:-main}

export R2_LOG_ROOT=${R2_LOG_ROOT:-$WM_POC_LOG_DIR/r2dreamer}
export R2_FIGURE_DIR=${R2_FIGURE_DIR:-$WM_POC_FIGURE_DIR/r2dreamer}
```

Experiment-specific variables:

```bash
export R2_ENV=${R2_ENV:-dmc_proprio}
export R2_SOURCE_TASK=${R2_SOURCE_TASK:-dmc_walker_walk}
export R2_TARGET_TASK=${R2_TARGET_TASK:-dmc_walker_run}
export R2_MODEL=${R2_MODEL:-size12M}
export R2_REP_LOSS=${R2_REP_LOSS:-dreamer}
export R2_SEED=${R2_SEED:-0}

export R2_SOURCE_STEPS=${R2_SOURCE_STEPS:-100000}
export R2_TARGET_STEPS=${R2_TARGET_STEPS:-50000}
export R2_SMOKE_STEPS=${R2_SMOKE_STEPS:-2000}

export R2_BATCH_SIZE=${R2_BATCH_SIZE:-16}
export R2_BATCH_LENGTH=${R2_BATCH_LENGTH:-64}
export R2_TRAIN_RATIO=${R2_TRAIN_RATIO:-16}
export R2_EVAL_EVERY=${R2_EVAL_EVERY:-10000}
export R2_EVAL_EPISODES=${R2_EVAL_EPISODES:-2}
```

The scripts should print these variables before launching anything.

---

## 6. Colab notebook structure

Create `notebooks/01_r2dreamer_foundation.ipynb`.

It should contain cells for the user to run manually.

### Cell 1 — title and warning

Markdown:

```markdown
# DreamerV3 / R2-Dreamer fine-tuning track

This notebook prepares and launches the DreamerV3/R2-Dreamer fine-tuning experiment.

It will:
1. Mount Google Drive.
2. Clone the project repo and the external `r2dreamer` repo.
3. Install dependencies.
4. Patch `r2dreamer` for checkpoint loading.
5. Run a tiny smoke test.
6. Provide commands for the full source/fine-tune/scratch runs.

Long training cells are intentionally separated and should be run manually.
```

### Cell 2 — mount Drive and set paths

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/wm_poc")
for subdir in [
    "data",
    "checkpoints",
    "logs",
    "figures",
    "external_repos",
    "tensorboard",
    "videos",
    "reports",
]:
    (DRIVE_ROOT / subdir).mkdir(parents=True, exist_ok=True)

os.environ["WM_POC_DRIVE_ROOT"] = str(DRIVE_ROOT)
os.environ["WM_POC_LOG_DIR"] = str(DRIVE_ROOT / "logs")
os.environ["WM_POC_CKPT_DIR"] = str(DRIVE_ROOT / "checkpoints")
os.environ["WM_POC_FIGURE_DIR"] = str(DRIVE_ROOT / "figures")
os.environ["WM_POC_EXTERNAL_REPOS"] = "/content/external_repos"
```

### Cell 3 — GPU check

```bash
nvidia-smi
python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("cuda capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
PY
```

### Cell 4 — clone/update project repo

Use placeholders for the GitHub URL.

```bash
cd /content

if [ ! -d wm-prediction ]; then
  git clone https://github.com/Thomas-Georges/wm-prediction.git
fi

cd /content/wm-prediction
git pull

export WM_POC_REPO=/content/wm-prediction
```

### Cell 5 — clone and install r2dreamer

```bash
cd /content/wm-prediction

bash scripts/r2dreamer/setup_r2dreamer.sh \
  --extras dmc \
  --target-dir /content/external_repos/r2dreamer
```

### Cell 6 — patch r2dreamer

```bash
cd /content/wm-prediction

python scripts/r2dreamer/patch_checkpoint_loading.py \
  --r2-repo /content/external_repos/r2dreamer

python scripts/r2dreamer/verify_r2dreamer_patch.py \
  --r2-repo /content/external_repos/r2dreamer
```

### Cell 7 — smoke command

This should run only a tiny test, not a meaningful experiment.

```bash
cd /content/wm-prediction

bash scripts/r2dreamer/run_smoke.sh
```

### Cell 8 — print the three full commands

```bash
cd /content/wm-prediction

python scripts/r2dreamer/build_commands.py \
  --config configs/r2dreamer/three_way_walker_walk_to_run.yaml \
  --print-only
```

### Cell 9 — run source base

Manual long-running cell:

```bash
cd /content/wm-prediction
bash scripts/r2dreamer/run_source_base.sh
```

### Cell 10 — run target fine-tune

Manual long-running cell:

```bash
cd /content/wm-prediction
bash scripts/r2dreamer/run_target_finetune.sh
```

### Cell 11 — run target scratch

Manual long-running cell:

```bash
cd /content/wm-prediction
bash scripts/r2dreamer/run_target_scratch.sh
```

### Cell 12 — parse and plot

```bash
cd /content/wm-prediction

python scripts/r2dreamer/summarize_runs.py \
  --run-root "$WM_POC_LOG_DIR/r2dreamer" \
  --out "$WM_POC_LOG_DIR/r2dreamer/summary.csv"

python scripts/r2dreamer/plot_finetune_vs_scratch.py \
  --finetune "$WM_POC_LOG_DIR/r2dreamer/target_finetune/metrics.jsonl" \
  --scratch "$WM_POC_LOG_DIR/r2dreamer/target_scratch/metrics.jsonl" \
  --out "$WM_POC_FIGURE_DIR/r2dreamer/finetune_vs_scratch.png"
```

---

## 7. Scripts to implement

### 7.1 `scripts/r2dreamer/setup_r2dreamer.sh`

Responsibilities:

- Create external repo root.
- Clone `NM512/r2dreamer` if missing.
- Checkout requested commit or branch.
- Print commit hash.
- Install dependencies with selected extras.
- Set EGL env vars in shell output.
- Never run training.

Suggested CLI:

```bash
bash scripts/r2dreamer/setup_r2dreamer.sh \
  --target-dir /content/external_repos/r2dreamer \
  --extras dmc \
  --commit main
```

Behavior:

```bash
pip install -e ".[dmc]"
```

For optional Meta-World later:

```bash
pip install -e ".[dmc,metaworld]"
```

Do not make Meta-World the default.

### 7.2 `scripts/r2dreamer/clone_r2dreamer.sh`

Can be a lower-level helper used by setup.

Required behavior:

- If repo exists, fetch and checkout.
- If repo does not exist, clone.
- Print:

```text
remote URL
current branch
current commit
status
```

### 7.3 `scripts/r2dreamer/install_r2dreamer.sh`

Can be a lower-level helper used by setup.

Required behavior:

- Check Python version.
- Check CUDA availability using Python.
- Install selected extras.
- Print installed package versions relevant to PyTorch, CUDA, Hydra, MuJoCo if available.

### 7.4 `scripts/r2dreamer/patch_checkpoint_loading.py`

This is the most important script.

It should modify the external `r2dreamer/train.py` to support loading pretrained weights.

Do this carefully and idempotently.

Requirements:

- Accept:

```bash
python scripts/r2dreamer/patch_checkpoint_loading.py --r2-repo /path/to/r2dreamer
```

- Refuse to run if `train.py` is missing.
- Backup original file once:

```text
train.py.before_wm_poc_checkpoint_patch
```

- Insert a clearly delimited block after the `agent = Dreamer(...).to(config.device)` creation.

- The patch should support these Hydra keys:

```text
+pretrained=/path/to/latest.pt
+pretrained_strict=true
+load_optimizer=false
```

- Default behavior:
  - If `pretrained` is absent or null: do nothing.
  - If `pretrained` is present:
    - load checkpoint using `torch.load(..., map_location=config.device)` or CPU then load to device.
    - load `agent_state_dict`.
    - print missing and unexpected keys.
    - fail if strict mode is true and keys mismatch.
  - Do not load optimizer state by default.
  - If `load_optimizer=true` and `optims_state_dict` exists:
    - call `tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])`.
    - print that optimizer state was loaded.
  - If `load_optimizer=true` but missing optimizer state:
    - warn, do not crash unless `strict` is true.

- Wrap training with `try/finally` so that `latest.pt` is saved even if the run is interrupted after agent creation.

Suggested inserted logic:

```python
# BEGIN WM_POC_CHECKPOINT_LOADING
pretrained = config.get("pretrained", None)
pretrained_strict = bool(config.get("pretrained_strict", True))
load_optimizer = bool(config.get("load_optimizer", False))

if pretrained:
    pretrained_path = pathlib.Path(str(pretrained)).expanduser()
    print(f"[wm_poc] Loading pretrained checkpoint: {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location=config.device)

    if "agent_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint {pretrained_path} does not contain 'agent_state_dict'. "
            f"Available keys: {list(ckpt.keys())}"
        )

    missing, unexpected = agent.load_state_dict(
        ckpt["agent_state_dict"],
        strict=pretrained_strict,
    )
    print(f"[wm_poc] Loaded agent weights from {pretrained_path}")
    print(f"[wm_poc] Missing keys: {len(missing)}")
    print(f"[wm_poc] Unexpected keys: {len(unexpected)}")

    if load_optimizer:
        if "optims_state_dict" not in ckpt:
            msg = f"[wm_poc] No optimizer state found in {pretrained_path}"
            if pretrained_strict:
                raise KeyError(msg)
            print(msg)
        else:
            tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
            print("[wm_poc] Loaded optimizer states.")
# END WM_POC_CHECKPOINT_LOADING
```

Also change:

```python
policy_trainer.begin(agent)

items_to_save = {
    "agent_state_dict": agent.state_dict(),
    "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
}
torch.save(items_to_save, logdir / "latest.pt")
```

to:

```python
try:
    policy_trainer.begin(agent)
finally:
    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "wm_poc_meta": {
            "pretrained": str(config.get("pretrained", None)),
            "pretrained_strict": bool(config.get("pretrained_strict", True)),
            "load_optimizer": bool(config.get("load_optimizer", False)),
            "env": str(config.env),
            "model": str(config.model),
            "seed": int(config.seed),
        },
    }
    torch.save(items_to_save, logdir / "latest.pt")
    print(f"[wm_poc] Saved checkpoint to {logdir / 'latest.pt'}")
```

Codex may need to adjust this because `config.env` and `config.model` may be OmegaConf objects. If serialization is fragile, convert with `str(...)`.

### 7.5 `scripts/r2dreamer/verify_r2dreamer_patch.py`

Responsibilities:

- Confirm that `train.py` contains `BEGIN WM_POC_CHECKPOINT_LOADING`.
- Confirm that `train.py` contains `pretrained`.
- Confirm that `train.py` contains a `try/finally` save block.
- Run `python -m py_compile train.py`.
- Optionally print a short diff against backup if available.
- Do not import MuJoCo or launch training.

### 7.6 `scripts/r2dreamer/verify_checkpoint.py`

Responsibilities:

- Load a `.pt` checkpoint.
- Check top-level keys.
- Count number of tensors and parameters in `agent_state_dict`.
- Print optimizer state availability.
- Print `wm_poc_meta` if present.
- Return non-zero if required keys are missing.

Suggested usage:

```bash
python scripts/r2dreamer/verify_checkpoint.py \
  --checkpoint "$WM_POC_LOG_DIR/r2dreamer/source_base/latest.pt"
```

### 7.7 `scripts/r2dreamer/build_commands.py`

Responsibilities:

- Read a YAML config from `configs/r2dreamer/`.
- Build the exact shell commands for:
  - smoke run
  - source base
  - target fine-tune
  - target scratch
- Support `--print-only`.
- Support `--write-scripts` if useful.
- Never launch commands unless explicitly passed `--execute`.
- Default must be print-only.

Generated commands should include:

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

Generated command form:

```bash
cd "$R2DREAMER_REPO"

python3 train.py \
  logdir="$R2_LOG_ROOT/source_base" \
  env="$R2_ENV" \
  env.task="$R2_SOURCE_TASK" \
  model="$R2_MODEL" \
  model.rep_loss="$R2_REP_LOSS" \
  seed="$R2_SEED" \
  batch_size="$R2_BATCH_SIZE" \
  batch_length="$R2_BATCH_LENGTH" \
  trainer.steps="$R2_SOURCE_STEPS" \
  trainer.eval_every="$R2_EVAL_EVERY" \
  trainer.eval_episode_num="$R2_EVAL_EPISODES" \
  trainer.train_ratio="$R2_TRAIN_RATIO"
```

Fine-tune command must include:

```bash
+pretrained="$R2_LOG_ROOT/source_base/latest.pt" \
+pretrained_strict=true \
+load_optimizer=false
```

Use `load_optimizer=false` for the main fine-tuning comparison. The goal is to transfer weights but reset the optimizer.

### 7.8 `scripts/r2dreamer/run_smoke.sh`

This should run a very short smoke test only.

Suggested default:

```bash
R2_ENV=dmc_proprio
R2_SOURCE_TASK=dmc_walker_walk
R2_MODEL=size12M
R2_REP_LOSS=dreamer
R2_SMOKE_STEPS=2000
R2_BATCH_SIZE=4
R2_BATCH_LENGTH=16
R2_TRAIN_RATIO=4
R2_EVAL_EVERY=1000
R2_EVAL_EPISODES=1
```

Command:

```bash
python3 train.py \
  logdir="$R2_LOG_ROOT/smoke" \
  env=dmc_proprio \
  env.task=dmc_walker_walk \
  model=size12M \
  model.rep_loss=dreamer \
  trainer.steps=2000 \
  batch_size=4 \
  batch_length=16 \
  trainer.train_ratio=4 \
  trainer.eval_every=1000 \
  trainer.eval_episode_num=1
```

This may still take a few minutes in Colab. It should never be run by Codex Web.

### 7.9 `scripts/r2dreamer/run_source_base.sh`

Run the source task from scratch.

Should require explicit confirmation:

```bash
export RUN_TRAINING=1
bash scripts/r2dreamer/run_source_base.sh
```

If `RUN_TRAINING` is not `1`, print the command and exit.

Default command should write to:

```text
$R2_LOG_ROOT/source_base
```

### 7.10 `scripts/r2dreamer/run_target_finetune.sh`

Run the target task initialized from:

```text
$R2_LOG_ROOT/source_base/latest.pt
```

Should fail early if the checkpoint is missing.

Should require:

```bash
export RUN_TRAINING=1
```

Default command should write to:

```text
$R2_LOG_ROOT/target_finetune
```

### 7.11 `scripts/r2dreamer/run_target_scratch.sh`

Run target task from random initialization with the same compute budget as fine-tune.

Should require:

```bash
export RUN_TRAINING=1
```

Default command should write to:

```text
$R2_LOG_ROOT/target_scratch
```

### 7.12 `scripts/r2dreamer/run_three_way_experiment.sh`

This script should orchestrate the three runs but still require `RUN_TRAINING=1`.

Flow:

1. Print configuration.
2. Verify r2dreamer repo exists.
3. Verify patch.
4. Run source base unless checkpoint already exists and `--skip-existing` is set.
5. Verify source checkpoint.
6. Run target fine-tune.
7. Run target scratch.
8. Summarize metrics.
9. Plot comparison.

It should support:

```bash
bash scripts/r2dreamer/run_three_way_experiment.sh --skip-existing
```

### 7.13 `scripts/r2dreamer/parse_metrics.py`

Responsibilities:

- Parse `metrics.jsonl`.
- Extract at minimum:
  - `step`
  - `episode/eval_score`
  - `episode/eval_length`
  - `episode/score`
  - `episode/length`
  - `fps/fps`
  - any `train/*` metrics available.
- Write CSV.
- Handle missing fields gracefully.
- Return useful errors if file is missing or invalid.

Usage:

```bash
python scripts/r2dreamer/parse_metrics.py \
  --metrics "$R2_LOG_ROOT/target_finetune/metrics.jsonl" \
  --out "$R2_LOG_ROOT/target_finetune/metrics.csv"
```

### 7.14 `scripts/r2dreamer/plot_finetune_vs_scratch.py`

Responsibilities:

- Read two or more parsed metrics files or raw `metrics.jsonl`.
- Plot eval return versus environment steps.
- Prefer `episode/eval_score`.
- Fall back to `episode/score` with a warning.
- Save PNG and optionally PDF.

Usage:

```bash
python scripts/r2dreamer/plot_finetune_vs_scratch.py \
  --finetune "$R2_LOG_ROOT/target_finetune/metrics.jsonl" \
  --scratch "$R2_LOG_ROOT/target_scratch/metrics.jsonl" \
  --out "$R2_FIGURE_DIR/finetune_vs_scratch.png"
```

Matplotlib rules:

- Use plain matplotlib.
- Do not require seaborn.
- Do not set custom colors unless necessary.
- One chart per figure.

### 7.15 `scripts/r2dreamer/summarize_runs.py`

Responsibilities:

- Walk a run root.
- Detect runs with:
  - `metrics.jsonl`
  - `console.log`
  - `latest.pt`
  - Hydra config files if present.
- Extract:
  - run name
  - final eval score
  - best eval score
  - final train episode score
  - number of metric rows
  - checkpoint exists
  - checkpoint size
  - GPU info if metadata exists.
- Write `summary.csv` and `summary.md`.

### 7.16 `scripts/r2dreamer/collect_gpu_info.py`

Responsibilities:

- Print and optionally save:
  - `nvidia-smi`
  - Python version
  - PyTorch version
  - CUDA availability
  - CUDA device name
  - GPU memory if available
  - current Git commit of this repo
  - current Git commit of r2dreamer
- Should not require GPU to pass; handle CPU-only gracefully.

### 7.17 `scripts/r2dreamer/archive_run_metadata.py`

Responsibilities:

- Copy or write small metadata files into each logdir:
  - command used
  - environment variables
  - Git commits
  - GPU info
  - timestamp
- Do not copy checkpoints into GitHub.
- Do not commit artifacts automatically.

---

## 8. Config templates

Create YAML configs that are simple and explicit. They are for our wrapper scripts, not necessarily Hydra configs.

### 8.1 `configs/r2dreamer/three_way_walker_walk_to_run.yaml`

```yaml
experiment_name: walker_walk_to_run_dmc_proprio_dreamer12m

r2dreamer:
  repo_env_var: R2DREAMER_REPO
  remote: https://github.com/NM512/r2dreamer.git
  commit: main
  extras: dmc

paths:
  log_root_env_var: R2_LOG_ROOT
  figure_dir_env_var: R2_FIGURE_DIR

algorithm:
  rep_loss: dreamer
  model: size12M

environment:
  env: dmc_proprio
  source_task: dmc_walker_walk
  target_task: dmc_walker_run

training:
  seed: 0
  source_steps: 100000
  target_steps: 50000
  batch_size: 16
  batch_length: 64
  train_ratio: 16
  eval_every: 10000
  eval_episodes: 2

finetuning:
  pretrained_strict: true
  load_optimizer: false

smoke:
  steps: 2000
  batch_size: 4
  batch_length: 16
  train_ratio: 4
  eval_every: 1000
  eval_episodes: 1
```

### 8.2 `configs/r2dreamer/smoke_dmc_proprio.yaml`

```yaml
experiment_name: smoke_dmc_proprio

algorithm:
  rep_loss: dreamer
  model: size12M

environment:
  env: dmc_proprio
  task: dmc_walker_walk

training:
  seed: 0
  steps: 2000
  batch_size: 4
  batch_length: 16
  train_ratio: 4
  eval_every: 1000
  eval_episodes: 1
```

### 8.3 `configs/r2dreamer/dreamer_dmc_vision_12m.yaml`

```yaml
experiment_name: walker_walk_to_run_dmc_vision_dreamer12m

algorithm:
  rep_loss: dreamer
  model: size12M

environment:
  env: dmc_vision
  source_task: dmc_walker_walk
  target_task: dmc_walker_run

training:
  seed: 0
  source_steps: 100000
  target_steps: 50000
  batch_size: 16
  batch_length: 64
  train_ratio: 16
  eval_every: 10000
  eval_episodes: 2

rendering:
  mujoco_gl: egl
  mujoco_egl_device_id: 0
```

The vision version should be treated as a later run, especially on T4.

---

## 9. Expected logs and outputs

Each run should write to a stable folder.

```text
$R2_LOG_ROOT/
  smoke/
    console.log
    metrics.jsonl
    latest.pt

  source_base/
    console.log
    metrics.jsonl
    latest.pt
    command.sh
    gpu_info.json
    run_metadata.json

  target_finetune/
    console.log
    metrics.jsonl
    latest.pt
    command.sh
    gpu_info.json
    run_metadata.json

  target_scratch/
    console.log
    metrics.jsonl
    latest.pt
    command.sh
    gpu_info.json
    run_metadata.json

  summary.csv
  summary.md
```

Figures:

```text
$R2_FIGURE_DIR/
  finetune_vs_scratch.png
  source_base_eval_score.png
  run_summary_table.md
```

Report template:

```text
reports/r2dreamer_results_template.md
```

---

## 10. Report template content

Create `reports/r2dreamer_results_template.md`.

It should contain:

```markdown
# DreamerV3/R2-Dreamer fine-tuning experiment

## Purpose

This experiment demonstrates checkpoint handling, GPU training, task adaptation, and fine-tuning versus scratch training for a PyTorch world-model agent.

## Hardware

- GPU:
- VRAM:
- PyTorch:
- CUDA:
- Runtime:
- Notes:

## External code

- r2dreamer remote:
- r2dreamer commit:
- Patch applied:
- Project repo commit:

## Experiment

Source task:
Target task:
Observation mode:
Model size:
Representation objective:
Source training steps:
Target fine-tuning steps:
Target scratch steps:
Seed:

## Runs

| Run | Initialization | Task | Steps | Final eval score | Best eval score | Wall-clock | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|
| source_base | random |  |  |  |  |  |  |
| target_finetune | source_base checkpoint |  |  |  |  |  |  |
| target_scratch | random |  |  |  |  |  |  |

## Main figure

Insert `figures/r2dreamer/finetune_vs_scratch.png`.

## Observations

- Did fine-tuning learn faster than scratch?
- Did the checkpoint load cleanly?
- Were observation/action spaces compatible?
- Any instability?
- Any GPU or Colab limitations?

## Limitations

This is not a full DreamerV3 benchmark reproduction. It is a controlled proof of competence in training, checkpointing, fine-tuning, and evaluating PyTorch world models.

## Next steps

- Repeat with `dmc_vision`.
- Try `model.rep_loss=r2dreamer`.
- Increase model size on A100.
- Move to local/global predictive-control experiment on PointMaze or PushT.
```

---

## 11. Testing requirements

Add lightweight tests. They must not require MuJoCo, GPU, or `r2dreamer`.

### 11.1 Command builder test

Test that `build_commands.py --dry-run` produces commands containing:

- `python3 train.py`
- `env=dmc_proprio`
- `env.task=dmc_walker_walk`
- `env.task=dmc_walker_run`
- `model.rep_loss=dreamer`
- `+pretrained=`
- `trainer.steps=`

### 11.2 Metrics parser test

Create a temporary `metrics.jsonl` file:

```jsonl
{"step": 0, "episode/eval_score": 10.0, "episode/eval_length": 100.0}
{"step": 1000, "episode/eval_score": 20.0, "fps/fps": 123.0}
{"step": 2000, "episode/score": 15.0}
```

Verify parser writes a CSV with expected columns.

### 11.3 Checkpoint verifier test

Create a tiny dummy checkpoint:

```python
torch.save({
    "agent_state_dict": {
        "dummy.weight": torch.zeros(2, 2),
    },
    "optims_state_dict": {},
    "wm_poc_meta": {
        "test": True,
    },
}, path)
```

Verify `verify_checkpoint.py` returns success.

Create a bad checkpoint without `agent_state_dict` and verify it fails.

### 11.4 Patch verifier test

Do not patch the real external repo in tests. Instead, use a temporary fake `train.py` containing the relevant pattern and verify patch insertion and idempotency.

---

## 12. Safety and artifact policy

The repo must not commit:

```text
*.pt
*.pth
*.ckpt
*.h5
*.hdf5
*.npz
*.npy
events.out.tfevents*
wandb/
tensorboard/
videos/
replay*/
buffer*/
data/
checkpoints/
logs/
external_repos/
```

Small generated figures may be committed if useful. Large figures/videos should stay in Drive.

The run scripts should write large outputs to:

```text
/content/drive/MyDrive/wm_poc/
```

not to the GitHub working tree.

---

## 13. Actual commands the user will run later

After Codex finishes this stage, the user should open Colab and run the notebook.

Manual Colab sequence:

```bash
cd /content/wm-prediction

bash scripts/r2dreamer/setup_r2dreamer.sh \
  --extras dmc \
  --target-dir /content/external_repos/r2dreamer

python scripts/r2dreamer/patch_checkpoint_loading.py \
  --r2-repo /content/external_repos/r2dreamer

python scripts/r2dreamer/verify_r2dreamer_patch.py \
  --r2-repo /content/external_repos/r2dreamer
```

Smoke test:

```bash
export RUN_TRAINING=1
bash scripts/r2dreamer/run_smoke.sh
```

Full three-way experiment:

```bash
export RUN_TRAINING=1
bash scripts/r2dreamer/run_source_base.sh

export RUN_TRAINING=1
bash scripts/r2dreamer/run_target_finetune.sh

export RUN_TRAINING=1
bash scripts/r2dreamer/run_target_scratch.sh
```

Parse and plot:

```bash
python scripts/r2dreamer/summarize_runs.py \
  --run-root "$R2_LOG_ROOT" \
  --out "$R2_LOG_ROOT/summary.csv"

python scripts/r2dreamer/plot_finetune_vs_scratch.py \
  --finetune "$R2_LOG_ROOT/target_finetune/metrics.jsonl" \
  --scratch "$R2_LOG_ROOT/target_scratch/metrics.jsonl" \
  --out "$R2_FIGURE_DIR/finetune_vs_scratch.png"
```

---

## 14. Acceptance criteria for Step 2

The step is complete when:

1. The repo contains a complete DreamerV3/R2-Dreamer experiment scaffold.
2. Codex did not run long training or download large data.
3. The Colab notebook can be opened and read top-to-bottom.
4. The setup script can clone and install `r2dreamer` when the user runs it.
5. The patch script can add checkpoint-loading support to `r2dreamer/train.py`.
6. The command builder prints source/fine-tune/scratch commands.
7. The launch scripts require `RUN_TRAINING=1` before executing.
8. The metrics parser and plotter can process fake test logs.
9. The checkpoint verifier can inspect a dummy checkpoint.
10. CPU-only tests pass.

Step 2 should end with the project ready for the first actual Colab run.

---

## 15. What not to do in this step

Do not:

- Run a real training job.
- Download any large datasets.
- Try Meta-World before DMC works.
- Try DMC Vision before DMC Proprio smoke tests pass.
- Increase model size beyond `size12M`.
- Commit external repositories.
- Commit checkpoints.
- Commit TensorBoard event files.
- Attempt to reproduce DreamerV4.
- Mix this stage with the later local/global planning experiment.

The only goal is to make the DreamerV3/R2-Dreamer fine-tuning track clean, reproducible, and ready to run.
