# DreamerV3 / R2-Dreamer Fine-Tuning Track

This track prepares a controlled PyTorch world-model experiment using the external `NM512/r2dreamer` repository.

The goal is to demonstrate:

- training a DreamerV3-style world model on a source task,
- saving and verifying `latest.pt`,
- saving model-only interval checkpoints at evaluation boundaries,
- reloading the checkpoint into a target task,
- fine-tuning with optimizer reset,
- training the same target task from scratch,
- comparing fine-tuning against scratch training.

## Run presets

| Config | Intended GPU | Obs | Model | Rep loss | Source steps | Target steps | Train ratio | Env workers | Eval eps |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `debug_walker_walk_to_run.yaml` | any | `dmc_proprio` | `size12M` | `dreamer` | 100K | 50K | 16 | 4 | 2 |
| `three_way_walker_walk_to_run_t4_r2_proprio.yaml` | T4 | `dmc_proprio` | `size12M` | `r2dreamer` | 510K | 250K | 64 | 4 | 5 |
| `three_way_walker_walk_to_run_a100_r2_vision25m.yaml` | A100 | `dmc_vision` | `size25M` | `r2dreamer` | 800K | 400K | 224 | 8 | 5 |

The previous short source run used the preserved debug values: `source_steps=100000`, `target_steps=50000`, and `train_ratio=16`. The default professor-facing config now mirrors the T4 R2 Proprio preset.

## Experiment design

Use the low-dimensional DMC Proprio path as the T4-safe first pillar:

```text
Source task: dmc_walker_walk
Target task: dmc_walker_run
Observation mode: dmc_proprio
Model size: size12M
Representation objective: model.rep_loss=r2dreamer
```

Use the image-based `dmc_vision` A100 preset as the vision pillar. The preset now uses a balanced compute budget while keeping the existing size25M config and run names.

## External code

External repository:

```text
https://github.com/NM512/r2dreamer
```

The external repository is cloned to `/content/external_repos/r2dreamer` in Colab. It is not vendored into this repository.

Use a Colab Python 3.11 runtime for R2-Dreamer training. The upstream package requires Python `>=3.11,<3.12`. If Colab opens a Python 3.12 runtime, switch the runtime version (Runtime -> Change runtime type) before installing R2-Dreamer; the install script stops with a clear message on unsupported Python.

Install the upstream package and DMC dependencies with:

```bash
bash scripts/r2dreamer/setup_r2dreamer.sh \
  --extras dmc \
  --target-dir /content/external_repos/r2dreamer
```

## Checkpoint patch

The wrapper patch adds Hydra keys to external `train.py`:

```text
+pretrained=/path/to/latest.pt
+pretrained_strict=true
+load_optimizer=false
```

The fine-tune run uses `load_optimizer=false` by default so the comparison transfers model weights but resets optimizer state.

The patch is idempotent and creates one backup:

```text
train.py.before_wm_poc_checkpoint_patch
```

## Manual run order

Open `notebooks/01_r2dreamer_foundation.ipynb` in Colab. Select the preset in the "Select run preset" cell: the default `a100_vision` preset runs the A100 Vision pillar, and `R2_PRESET=t4_proprio` runs the T4 Proprio pillar. The notebook is self-contained: it mounts Drive, defines paths, selects the preset and unique run directories, ensures the runtime repository exists under `/content/wm-prediction`, and then runs the R2-Dreamer setup cells.

If the repository is private, the clone cell prompts for a GitHub username and personal access token. Use a fine-grained token scoped to this repository with contents read access. The token is only passed to Git for the clone/pull operation and is not written into the notebook.

Then run:

```bash
export R2_CONFIG=configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml
export R2_LOG_ROOT=/content/drive/MyDrive/wm_poc/logs/r2dreamer/walker_walk_to_run_t4_r2_proprio_12m_seed0
export R2_FIGURE_DIR=/content/drive/MyDrive/wm_poc/figures/r2dreamer/walker_walk_to_run_t4_r2_proprio_12m_seed0
export RUN_TRAINING=1
bash scripts/r2dreamer/run_smoke.sh
bash scripts/r2dreamer/run_source_base.sh
bash scripts/r2dreamer/run_target_finetune.sh
bash scripts/r2dreamer/run_target_scratch.sh
```

Without `RUN_TRAINING=1`, these scripts print the commands and exit.

For A100 Vision, use:

```bash
export R2_CONFIG=configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml
export R2_LOG_ROOT=/content/drive/MyDrive/wm_poc/logs/r2dreamer/walker_walk_to_run_a100_r2_vision25m_seed0
export R2_FIGURE_DIR=/content/drive/MyDrive/wm_poc/figures/r2dreamer/walker_walk_to_run_a100_r2_vision25m_seed0
export R2_MUJOCO_GL=osmesa
export R2_MUJOCO_EGL_DEVICE_ID=0
```

The smoke run is intentionally constrained for Colab. It overrides both upstream `env.*` and trainer settings, including `env.env_num=1` and `env.eval_episode_num=0`. This avoids the upstream DMC defaults that otherwise start 16 train env workers and 10 eval env workers before the run even begins.

If Colab buffers a foreground `%%bash` cell until completion, launch and monitor from one cell:

```bash
bash scripts/r2dreamer/run_with_live_monitor.sh source_base
```

Use `smoke`, `target_finetune`, or `target_scratch` instead of `source_base` for those runs. This starts the training process in the background inside the same shell, tails `<run>/console.log`, waits for training to finish, and exits with the training status. Interrupting this cell stops the monitored run.

For `dmc_proprio`, commands also export `WM_POC_DMC_DISABLE_IMAGE_RENDER=true`. Upstream DMC always returns an `image` observation, even when the proprio model ignores CNN inputs. The WM POC patch keeps that key in the observation dictionary but returns a zero image placeholder, avoiding Colab MuJoCo offscreen-render crashes during proprio smoke and training runs.

For `dmc_vision`, keep `WM_POC_DMC_DISABLE_IMAGE_RENDER=false` so the model trains on real rendered pixels. The A100 vision presets default to parallel DMC env workers on Python 3.11. If a runtime still crashes in the upstream multiprocessing env wrapper, set `R2_SERIAL_ENVS=true` as a fallback; this preserves real image observations but may increase wall-clock time.

Generated run commands also set quiet headless-runtime defaults for Colab: `TF_CPP_MIN_LOG_LEVEL=2`, `SDL_AUDIODRIVER=dummy`, and `XDG_RUNTIME_DIR=/tmp/xdg-runtime`. These suppress most TensorFlow CPU banner, missing audio device, and missing runtime-dir noise without changing training behavior.

Generated commands also patch `trainer.py` to print a heartbeat before the first checkpoint boundary:

```text
[wm_poc] progress [###---------------------] 000010000/000800000 (  1.2%) elapsed=...
```

The default cadence is every 100 trainer steps. Override with `R2_PROGRESS_EVERY=1000` for quieter logs, or `R2_PROGRESS_EVERY=0` to disable it.

Smoke commands set `model.compile=false` by default so they do not spend time in `torch.compile` before reaching the training loop. Full source/fine-tune/scratch commands keep `model.compile=true` unless overridden with `R2_COMPILE=false`.

Full source/fine-tune/scratch runs save model-only interval checkpoints at evaluation boundaries. By default:

```text
trainer.checkpoint_every = trainer.eval_every
trainer.checkpoint_keep = 8 for T4 scaled, 6 for balanced A100 vision
```

Those files are written under each run directory:

```text
/content/drive/MyDrive/wm_poc/logs/r2dreamer/source_base/checkpoints/step_000010000.pt
```

They contain `agent_state_dict` and small `wm_poc_meta` only; optimizer state and replay buffers are intentionally omitted. The final successful run still writes `latest.pt` with optimizer state.

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

Leave `R2_MODEL` unset for the balanced A100 vision run so it keeps the configured `size25M` model. `R2_SOURCE_CKPT` changes the actual `+pretrained=...` path emitted for `target_finetune`. `R2_LOAD_OPTIMIZER=false` is intentional for transfer; loading optimizer state is for resuming the same run, not for a clean source-to-target comparison.

## Outputs

Large outputs stay in Google Drive:

```text
/content/drive/MyDrive/wm_poc/logs/r2dreamer/
/content/drive/MyDrive/wm_poc/figures/r2dreamer/
```

Do not commit checkpoints, replay buffers, TensorBoard event files, videos, or external repo contents.

## Troubleshooting

If `run_smoke.sh` fails around `Create envs.` and child workers print `BrokenPipeError`, treat that as a secondary multiprocessing symptom. The parent process usually died while starting or initializing too many env workers. Pull the latest repository version and confirm:

```bash
python scripts/r2dreamer/build_commands.py --run smoke --print-only
```

The printed command should include:

```text
env.env_num=1
env.eval_episode_num=0
trainer.eval_episode_num=0
```

After a failed worker-start attempt, restart the notebook kernel before rerunning the smoke script so leaked multiprocessing semaphores do not carry over.

If `run_smoke.sh` reaches `Simulate agent.` and then fails with `RuntimeError: Lost connection to worker` during `envs.step(...)`, the DMC subprocess probably died during MuJoCo rendering. For `dmc_vision`, do not keep `WM_POC_DMC_DISABLE_IMAGE_RENDER=true` for a real run because that trains on blank images. Restart the runtime, use OSMesa, and rerun setup:

```bash
export R2_MUJOCO_GL=osmesa
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

bash scripts/r2dreamer/setup_r2dreamer.sh \
  --extras dmc \
  --target-dir /content/external_repos/r2dreamer
```

Then confirm the generated smoke command includes:

```text
MUJOCO_GL=${MUJOCO_GL:-osmesa}
PYOPENGL_PLATFORM="${MUJOCO_GL}"
WM_POC_DMC_DISABLE_IMAGE_RENDER=false
WM_POC_R2_SERIAL_ENVS=false
```

The generated command sets `WM_POC_DMC_DISABLE_IMAGE_RENDER` explicitly from the
wrapper config, so a stale internal `WM_POC_DMC_DISABLE_IMAGE_RENDER=true` value
does not leak into `dmc_vision` runs. Use the public
`R2_DISABLE_DMC_IMAGE_RENDER=true` override only for deliberate render-debug
runs, not for source/fine-tune/scratch results.

Before a long vision run, this lightweight check should report nonzero pixel
variation and can write a one-frame PNG preview:

```bash
python scripts/r2dreamer/check_dmc_vision_render.py \
  --r2-repo /content/external_repos/r2dreamer \
  --out /content/drive/MyDrive/wm_poc/figures/r2dreamer/dmc_vision_render_check.png
```

If the same worker loss persists despite Python 3.11 and OSMesa, try the serial fallback:

```bash
export R2_SERIAL_ENVS=true
```

Then rerun the patch and command-print cells. The generated command should include `WM_POC_R2_SERIAL_ENVS=true`, and the run should print `[wm_poc] Using serial envs ...` after `Create envs.`. For `dmc_proprio`, `WM_POC_DMC_DISABLE_IMAGE_RENDER=true` remains safe because the model does not use rendered pixels. This worker loss is usually the DMC subprocess dying, not a temporary VSCode or browser disconnect.

Older versions of the WM POC checkpoint patch wrote `latest.pt` in a `finally` block. Rerunning the patch command upgrades existing external checkouts so `latest.pt` is written only after the run succeeds.
