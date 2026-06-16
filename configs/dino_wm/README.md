# DINO-WM Configs

These configs define a standalone DINO-WM experiment track. They do not depend on the R2-Dreamer configs or scripts.

All long-running scripts dry-run by default. Set `RUN_DINO_WM=1` only when you are ready to train, fine-tune, precompute latents, or plan on a GPU runtime.

Required environment variables are configurable, with Colab/Drive-oriented defaults:

- `DINO_WM_REPO`: upstream checkout, default `/content/drive/MyDrive/wm_poc/external_repos/dino_wm`
- `DINO_WM_DATA_ROOT`: dataset root, default `/content/drive/MyDrive/wm_poc/data/dino_wm`
- `DINO_LOG_ROOT`: run logs, default `/content/drive/MyDrive/wm_poc/logs/dino_wm`
- `DINO_CKPT_ROOT`: checkpoints, default `/content/drive/MyDrive/wm_poc/checkpoints/dino_wm`
- `DINO_WM_FEATURE_CACHE`: frozen DINOv2 feature cache
- `DINO_MIXED_PRECISION`: upstream Accelerate precision mode, for example `bf16` on A100/L4 or `fp16`; default upstream behavior remains `no`
- `DINO_SAVE_EVERY_EPOCHS`: optional runtime override for checkpoint frequency
- `DINO_SAVE_EVERY_STEPS`: optional runtime override for rolling latest checkpoints inside an epoch
- `DINO_PATCH_STEP_CHECKPOINTING=1`: opt into the experimental upstream train-loop patch needed for step checkpoints
- `DINO_FORCE_RESTART=1`: move existing checkpoints aside before training, forcing a fresh run

Upstream datasets come from the official DINO-WM OSF project: <https://osf.io/bmw48/>. The supported first-pass environments are `point_maze`, `wall_single`, and `pusht_noise`.

Training resumes from `${DINO_CKPT_ROOT}/outputs/<run_name>/checkpoints/model_latest.pth`
by default when that file exists. The base config saves every epoch for Colab
resilience. To genuinely restart a run with the same `run_name`, set
`DINO_FORCE_RESTART=1`; the wrapper moves the old `checkpoints/` directory to a
timestamped backup instead of deleting it.

Intra-epoch checkpointing is experimental and disabled by default. With the
default `DINO_PATCH_STEP_CHECKPOINTING=0`, the wrapper restores the upstream
DINO-WM `train.py` if our step-checkpointing patch had been applied earlier. To
opt in, set `DINO_PATCH_STEP_CHECKPOINTING=1` and a positive
`DINO_SAVE_EVERY_STEPS` value. Step checkpoints do not create `model_step_*.pth`
files and do not overwrite the upstream epoch checkpoint. They atomically update
one rolling CPU `state_dict` checkpoint at `model_latest_step.pth`, record the
batch index, and are cleared after a successful epoch-level checkpoint so future
resumes prefer the newest complete epoch.

For first-batch CUDA OOM debugging, start with
`pointmaze_oom_safe.yaml`. It uses a small dataset cap, `batch_size=2`,
disables planning, and passes upstream overrides that turn off the decoder and
reconstruction samples. After that completes, `pointmaze_full_nodecoder_bf16.yaml`
is the A100-oriented full-data profile. It keeps the decoder off, restores
`batch_size=32` to keep epoch length close to the original baseline, and runs an
initial 10-epoch train loop. Use `DINO_MIXED_PRECISION=bf16` on A100. Do not run
this full-data profile on T4; use the OOM-safe or low-data configs there.

List the official archives without downloading:

```bash
python scripts/dino_wm/download_data.py --dataset point_maze --list
```

Download the smoke dataset to the configured Drive data root:

```bash
python scripts/dino_wm/download_data.py --dataset point_maze --execute
```

Download the larger transfer/planning datasets only when needed:

```bash
python scripts/dino_wm/download_data.py --dataset wall_single --dataset pusht_noise --execute
```

## Experiment Matrix

| ID | Config | Mode | Purpose | Max expected A100 time |
|---|---:|---|---|---:|
| D0 | `smoke_pointmaze.yaml` | smoke | Verify install, data path, feature extraction, train loop, checkpoint, plan loop. | 5-15 min |
| D0a | `pointmaze_oom_safe.yaml` | scratch | OOM-safe predictor-only PointMaze debug run. | <1 h |
| D0b | `pointmaze_full_nodecoder_bf16.yaml` | scratch | A100 full-data no-decoder PointMaze run with mixed precision. | 2-4 h |
| D1 | `pointmaze_scratch_a100.yaml` | scratch | Main PointMaze baseline. | 1-2 h |
| D2 | `pointmaze_lowdata_scratch_a100.yaml` | scratch | Low-data target baseline. | <1 h |
| D3 | `pointmaze_lowdata_finetune_a100.yaml` | fine-tune | Fine-tune D1 on the same low-data target split as D2. | <1 h |
| D4 | `wall_scratch_a100.yaml` | scratch | Navigation transfer target. | 1-2 h |
| D5 | `wall_finetune_from_pointmaze_a100.yaml` | fine-tune | PointMaze to Wall transfer. | <1 h |
| D6 | `pusht_1k_scratch_a100.yaml` | scratch | Contact-rich Push-T subset baseline. | 2-3.5 h |
| D7 | `pusht_1k_finetune_a100.yaml` | fine-tune | Fine-tune Push-T source checkpoint on a small target split. | 1-2 h |
| D8a | `pusht_causal_mask_true_a100.yaml` | ablation | Causal-mask enabled. | 1-2 h |
| D8b | `pusht_causal_mask_false_a100.yaml` | ablation | Causal-mask disabled. | 1-2 h |
| D9 | `planner_cem_vs_gd_pointmaze.yaml` | planner-only | CEM vs GD from a fixed checkpoint. | <1 h |

## Common Commands

Render a command without running it:

```bash
python scripts/dino_wm/build_commands.py \
  --config configs/dino_wm/smoke_pointmaze.yaml \
  --stage train \
  --print
```

Dry-run the smoke wrapper:

```bash
bash scripts/dino_wm/run_smoke.sh
```

Run the smoke wrapper on a GPU runtime:

```bash
RUN_DINO_WM=1 bash scripts/dino_wm/run_smoke.sh
```

Summarize runs:

```bash
python scripts/dino_wm/summarize_runs.py \
  --root "${DINO_LOG_ROOT:-/content/drive/MyDrive/wm_poc/logs/dino_wm}" \
  --out "${DINO_LOG_ROOT:-/content/drive/MyDrive/wm_poc/logs/dino_wm}/_summary"
```
