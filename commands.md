# Command Templates

These commands are templates for future Colab runs. They should not be run during repository bootstrap.

## 1. Clone this repository in Colab

```bash
cd /content
git clone https://github.com/Thomas-Georges/wm-prediction.git
cd wm-prediction
```

If you fork this repository, replace the GitHub URL with your fork.

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

Do not run until dependencies are installed and the checkpoint patch is verified.

The R2-Dreamer notebook is self-contained. It mounts Drive, defines paths, and ensures the runtime repo exists before running these commands.

If the repository is private, the notebook clone cell will ask for a GitHub username and personal access token. Do not paste tokens into committed files.

Setup:

```bash
cd /content/wm-prediction

bash scripts/r2dreamer/setup_r2dreamer.sh \
  --extras dmc \
  --target-dir /content/external_repos/r2dreamer \
  --allow-unsupported-python

python scripts/r2dreamer/patch_checkpoint_loading.py \
  --r2-repo /content/external_repos/r2dreamer

python scripts/r2dreamer/verify_r2dreamer_patch.py \
  --r2-repo /content/external_repos/r2dreamer
```

Print smoke/source/fine-tune/scratch commands without running:

```bash
cd /content/wm-prediction
python scripts/r2dreamer/build_commands.py --dry-run
```

T4 R2 Proprio preset:

```bash
export R2_CONFIG=configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml
export R2_LOG_ROOT=/content/drive/MyDrive/wm_poc/logs/r2dreamer/walker_walk_to_run_t4_r2_proprio_12m_seed0
export R2_FIGURE_DIR=/content/drive/MyDrive/wm_poc/figures/r2dreamer/walker_walk_to_run_t4_r2_proprio_12m_seed0
```

A100 R2 Vision preset:

```bash
export R2_CONFIG=configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml
export R2_LOG_ROOT=/content/drive/MyDrive/wm_poc/logs/r2dreamer/walker_walk_to_run_a100_r2_vision25m_seed0
export R2_FIGURE_DIR=/content/drive/MyDrive/wm_poc/figures/r2dreamer/walker_walk_to_run_a100_r2_vision25m_seed0
export R2_MUJOCO_GL=egl
export R2_MUJOCO_EGL_DEVICE_ID=0
```

Full runs save model-only interval checkpoints at evaluation boundaries by default. Override retention if needed:

```bash
export R2_CHECKPOINT_KEEP=6
```

Run the tiny smoke test manually:

The smoke command should print or execute with `env.env_num=1`, `env.eval_episode_num=0`, `trainer.eval_episode_num=0`, and `WM_POC_DMC_DISABLE_IMAGE_RENDER=true`.

```bash
cd /content/wm-prediction
export RUN_TRAINING=1
bash scripts/r2dreamer/run_smoke.sh
```

Run the three required comparison runs manually:

```bash
cd /content/wm-prediction
export RUN_TRAINING=1
bash scripts/r2dreamer/run_source_base.sh
bash scripts/r2dreamer/run_target_finetune.sh
bash scripts/r2dreamer/run_target_scratch.sh
```

The comparison design is:

```text
source_base: train dmc_walker_walk from scratch.
target_finetune: train dmc_walker_run initialized from source_base/latest.pt with optimizer reset.
target_scratch: train dmc_walker_run from random initialization with the same target budget.
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

## 7. Future DINO-WM setup

Do not run until datasets are downloaded and dependencies are installed.

```bash
cd /content/drive/MyDrive/wm_poc/external_repos/dino_wm
python train.py --help
python plan.py --help
```
