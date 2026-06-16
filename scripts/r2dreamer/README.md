# R2-Dreamer Scripts

These scripts orchestrate the external `NM512/r2dreamer` repository from this project.

They do not vendor external code into this repository. Training scripts are guarded by `RUN_TRAINING=1`; without that environment variable they print the command and exit.

Default Colab locations:

```bash
export WM_POC_REPO=/content/World_Models_LAS
export WM_POC_DRIVE_ROOT=/content/drive/MyDrive/wm_poc
export WM_POC_EXTERNAL_REPOS=/content/external_repos
export R2DREAMER_REPO=/content/external_repos/r2dreamer
export R2_LOG_ROOT=/content/drive/MyDrive/wm_poc/logs/r2dreamer
export R2_FIGURE_DIR=/content/drive/MyDrive/wm_poc/figures/r2dreamer
```

Use a Colab Python 3.11 runtime for training. Upstream `r2dreamer` requires Python `>=3.11,<3.12`; if Colab opens a 3.12 runtime, switch the runtime version before installing. The install script stops with a clear message on unsupported Python.

For Colab smoke runs, the generated command overrides upstream DMC env defaults to keep worker startup small:

```text
env.env_num=1
env.eval_episode_num=0
trainer.eval_episode_num=0
```

For `dmc_proprio`, generated commands also set:

```text
WM_POC_DMC_DISABLE_IMAGE_RENDER=true
```

The external patch keeps the upstream `image` observation key but returns a zero image placeholder when that variable is true. This avoids unnecessary MuJoCo offscreen rendering for proprio runs.

For `dmc_vision`, keep image rendering enabled and prefer OSMesa in Colab, even on GPU runtimes:

```text
MUJOCO_GL=osmesa
PYOPENGL_PLATFORM=osmesa
WM_POC_DMC_DISABLE_IMAGE_RENDER=false
```

Generated commands set `WM_POC_DMC_DISABLE_IMAGE_RENDER` explicitly from the
selected wrapper config. This prevents a stale internal debug value from
carrying into a vision run. Use `R2_DISABLE_DMC_IMAGE_RENDER=true` only for an
intentional blank-image debug run.

Before launching a long `dmc_vision` run, verify that the current runtime returns
nonblank rendered frames:

```bash
python scripts/r2dreamer/check_dmc_vision_render.py \
  --r2-repo /content/external_repos/r2dreamer \
  --out /content/drive/MyDrive/wm_poc/figures/r2dreamer/dmc_vision_render_check.png
```

The model still trains on CUDA; OSMesa only handles MuJoCo camera frames. Use EGL only after validating it in the current runtime.

The DMC vision presets default to parallel env workers on Python 3.11:

```text
WM_POC_R2_SERIAL_ENVS=false
```

If a Colab runtime still crashes in the upstream multiprocessing environment wrapper, use the serial fallback:

```text
WM_POC_R2_SERIAL_ENVS=true
```

This keeps real image observations but bypasses multiprocessing env stepping, so it may increase wall-clock time.

Full runs also patch `trainer.py` to print progress heartbeats independently of checkpointing:

```text
[wm_poc] progress [###---------------------] 000010000/000800000 (  1.2%) elapsed=...
```

Use `R2_PROGRESS_EVERY` to control the cadence. The default is 100 trainer steps; use `0` to disable.

Smoke commands set `model.compile=false` by default so they reach env stepping quickly. Full source/fine-tune/scratch commands keep `model.compile=true` unless overridden with `R2_COMPILE=false`.

Full runs also patch `trainer.py` to save model-only interval checkpoints at eval boundaries:

```text
<run>/checkpoints/step_000010000.pt
```

Use `R2_CHECKPOINT_KEEP` to control how many interval checkpoints are retained per run. The T4 scaled preset defaults to 8; the balanced A100 vision preset defaults to 6.

## Run presets

| Config | Intended GPU | Obs | Model | Rep loss | Source steps | Target steps | Train ratio | Env workers | Eval eps |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `debug_walker_walk_to_run.yaml` | any | `dmc_proprio` | `size12M` | `dreamer` | 100K | 50K | 16 | 4 | 2 |
| `three_way_walker_walk_to_run_t4_r2_proprio.yaml` | T4 | `dmc_proprio` | `size12M` | `r2dreamer` | 510K | 250K | 64 | 4 | 5 |
| `three_way_walker_walk_to_run_a100_r2_vision25m.yaml` | A100 | `dmc_vision` | `size25M` | `r2dreamer` | 800K | 400K | 224 | 8 | 5 |

Useful overrides:

```bash
export R2_TRAIN_RATIO=128
export R2_TARGET_STEPS=510000
export R2_ENV_NUM=8
export R2_PROGRESS_EVERY=100
export R2_COMPILE=true
export R2_SOURCE_CKPT=/path/to/source/latest.pt
export R2_PRETRAINED_STRICT=true
export R2_LOAD_OPTIMIZER=false
```

Leave `R2_MODEL` unset for the balanced A100 vision run so it keeps the configured `size25M` model. `finetuning.load_optimizer: false` is intentional for transfer. Loading optimizer state is for resuming the same run, not for a clean source-to-target fine-tune comparison.

## Live Colab Monitoring

Colab can buffer a long foreground `%%bash` cell and dump stdout only after the command exits. In notebooks, use the Python display monitor from `wm_poc.r2dreamer.notebook_monitor` so the running cell updates one live panel in place:

```python
from wm_poc.r2dreamer.notebook_monitor import run_r2_with_live_display

run_r2_with_live_display("source_base")
```

The run argument can be `smoke`, `source_base`, `target_finetune`, or `target_scratch`. The helper starts the training process in the background from the notebook kernel, clears stale `<run>/console.log`, polls the new log every `R2_MONITOR_INTERVAL` seconds, updates the notebook display, waits for training to finish, and exits with the training status. Interrupting the cell stops the monitored run.

```python
run_r2_with_live_display("source_base", monitor_interval=15)
```

In a terminal, the shell monitor is still available:

```bash
bash scripts/r2dreamer/run_with_live_monitor.sh source_base
```

In a terminal or a notebook environment that supports concurrent cells, the split form is also available:

```bash
bash scripts/r2dreamer/run_background.sh source_base
bash scripts/r2dreamer/tail_run_progress.sh source_base
```

The background launcher writes `<run>/launcher.pid` and `<run>/launcher.log`; the upstream trainer writes `<run>/console.log`.

## Post-hoc checkpoint visualizations

Use `notebooks/06_r2dreamer_results.ipynb` or the scripts below after checkpoints already exist. These scripts do not train or modify checkpoints.

```bash
export RUN_R2_VISUALIZATIONS=1
export WM_POC_DRIVE_ROOT=/content/drive/MyDrive/wm_poc
export R2DREAMER_REPO=/content/external_repos/r2dreamer
export R2_LOG_ROOT=/content/drive/MyDrive/wm_poc/logs/r2dreamer/walker_walk_to_run_t4_r2_proprio_12m_seed0
export R2_FIGURE_DIR=/content/drive/MyDrive/wm_poc/figures/r2dreamer/walker_walk_to_run_t4_r2_proprio_12m_seed0
```

`RUN_R2_VISUALIZATIONS=1` is used by the notebook command runner. Direct script commands execute immediately, but keeping the variable in the environment is harmless and makes the notebook behavior explicit.

If the notebook reports a missing `/content/World_Models_LAS/...` path, run the notebook section `Update repository in Colab` first. It clones or pulls the public repo inside Colab over HTTPS.

If it reports missing `/content/external_repos/r2dreamer`, run the next notebook section, `Update upstream R2-Dreamer checkout`. That cell only clones or updates the upstream repo; it does not install dependencies, train, or touch checkpoints.

If the first visualization command fails with `No module named 'hydra'`, run the notebook section `Install R2-Dreamer runtime`. This is the same setup step used by the training notebook:

```bash
bash scripts/r2dreamer/setup_r2dreamer.sh \
  --extras dmc \
  --target-dir /content/external_repos/r2dreamer
```

That installs the upstream R2-Dreamer package and DMC dependencies in the current Colab runtime. It does not run source training, fine-tuning, scratch training, or any checkpoint-producing job.

If MuJoCo reports `DISPLAY`, `gladLoadGL`, or the rollout process exits with `-11`, run the notebook section `Configure headless MuJoCo rendering`. In Colab, prefer OSMesa for rendering stability even on GPU runtimes; the policy can still run on CUDA while MuJoCo renders through CPU Mesa.

For the T4 `dmc_proprio` checkpoints, the safest Colab fallback is trace mode. It still runs the trained policy in DMC, but disables MuJoCo camera rendering and writes an MP4 diagnostic from latent PCA, return, and action norm:

```bash
python scripts/r2dreamer/render_policy_rollouts.py \
  --config configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml \
  --run all \
  --episodes 1 \
  --max-steps 1000 \
  --mujoco-gl osmesa \
  --render-mode trace
```

The rollout scripts also accept an explicit backend:

```bash
# Colab-safe default
python scripts/r2dreamer/render_policy_rollouts.py --mujoco-gl osmesa ...

# Try this only if EGL works in your runtime
python scripts/r2dreamer/render_policy_rollouts.py --mujoco-gl egl ...
```

For CPU Colab, OSMesa system libraries may be needed. The notebook cell can install them if you set:

```python
os.environ["INSTALL_OSMESA_PACKAGES"] = "1"
```

The equivalent manual Colab cell is:

```bash
cd /content/World_Models_LAS
git pull
export WM_POC_REPO=/content/World_Models_LAS
test -f scripts/r2dreamer/render_policy_rollouts.py
```

The notebook runner now performs this preflight check and prints stdout/stderr from failed commands.

Render environment videos from `latest.pt` checkpoints:

```bash
RUN_R2_VISUALIZATIONS=1 python scripts/r2dreamer/render_policy_rollouts.py \
  --config configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml \
  --run all \
  --episodes 1
```

Extract RSSM latent trajectories and plot shared-basis 2D/3D PCA comparisons:

```bash
RUN_R2_VISUALIZATIONS=1 python scripts/r2dreamer/extract_latent_trajectories.py \
  --config configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml \
  --run all \
  --episodes 3

RUN_R2_VISUALIZATIONS=1 python scripts/r2dreamer/plot_latent_trajectories.py \
  --config configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml \
  --color-by run
```

For the A100 vision preset, point `R2_LOG_ROOT` at the vision run folder and use the vision config:

```bash
export RUN_R2_VISUALIZATIONS=1
export R2_LOG_ROOT=/content/drive/MyDrive/wm_poc/logs/r2dreamer/<vision-run-folder>
export R2_FIGURE_DIR=/content/drive/MyDrive/wm_poc/figures/r2dreamer/<vision-run-folder>

RUN_R2_VISUALIZATIONS=1 python scripts/r2dreamer/render_policy_rollouts.py \
  --config configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml \
  --run all \
  --episodes 1

RUN_R2_VISUALIZATIONS=1 python scripts/r2dreamer/extract_latent_trajectories.py \
  --config configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml \
  --run all \
  --episodes 3

RUN_R2_VISUALIZATIONS=1 python scripts/r2dreamer/plot_latent_trajectories.py \
  --config configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml \
  --color-by run
```

Outputs are written to Drive by default:

```text
videos/r2dreamer/<run-name>/rollouts/
figures/r2dreamer/<run-name>/visualizations/
reports/r2dreamer/<run-name>/visualizations/
```

The visualization notebook includes preview cells that copy generated MP4s from Drive to `/content/r2dreamer_visualization_previews` and embed them inline as base64 HTML videos with playback controls. This local copy is only for smoother notebook playback; the durable artifact remains in Drive. If a video is large, set `R2_VIDEO_EMBED_LIMIT_MB` before rerunning the preview cell.

For R2-Dreamer vision checkpoints, the policy videos and latent plots use the same scripts with the vision config. Decoder reconstruction grids are only available when the upstream agent exposes `video_pred`, which currently requires `rep_loss == "dreamer"`.
