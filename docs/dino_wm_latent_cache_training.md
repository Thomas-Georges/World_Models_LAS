# DINO-WM Latent-Cache Training

## Why one epoch took 30+ hours

Profiling the upstream data path showed the slowdown was never the predictor
and was only partly the encoder. Upstream `TrajSlicerDataset.__getitem__` calls
`PointMazeDataset.__getitem__(traj_idx)`, which:

1. `torch.load`s the **entire episode image tensor**
   (`obses/episode_XXX.pth`, ~90 frames) from Google Drive,
2. converts and transforms **every frame** of that episode,
3. then throws away everything except the 4 frames the slice needs.

With ~147,000 sliced windows per epoch, each epoch re-reads and re-transforms
roughly 1,600x the dataset size through the Drive FUSE mount. On top of that,
the frozen DINO encoder re-encodes those frames online every batch, which is
pure recomputation since `model.train_encoder=false`. This also explains why
batch size made no difference: the cost is per-sample I/O and CPU transform
work, not per-step GPU work.

## The fix

Frozen encoder means latents are constants: compute them once, reuse them every
epoch.

### 1. One-time latent precompute

`scripts/dino_wm/precompute_latents.py --config <config> --no-dry-run` now
installs `wm_poc_precompute_latents.py` into the upstream checkout and runs it.
The script:

- loads each episode image file exactly once (sequential read),
- replicates the exact upstream preprocessing (`default_transform(img_size)`
  followed by the `VWorldModel` resize to `(img_size // 16) * patch_size`,
  i.e. 196 px for the default 224 config, giving 196 patch tokens),
- encodes frames through the frozen DINO encoder in batches under autocast
  (`DINO_MIXED_PRECISION`, default bf16 on CUDA),
- writes per-episode fp16 latents to `episode_XXX.npy` plus a
  `wm_poc_latent_manifest.json` (encoder name, image size, patch count,
  embedding dim, episode lengths).

It is idempotent and resumable: existing episode files with the right length
are skipped, so an interrupted precompute continues where it stopped, and
configs needing fewer rollouts reuse a larger cache.

Cost on A100: ~205k frames through ViT-S/14 is minutes of GPU time; the bound
is reading the ~30 GB of episode files from Drive once (~10-20 minutes total).

### 2. Latent-backed training dataset

When `features.cache_enabled` is true and the env is supported (`point_maze`),
the train command swaps the Hydra dataset target:

```
env.dataset._target_=wm_poc_latent_dataset.load_point_maze_latent_slice_train_val
+env.dataset.latent_cache_dir=<cache>/point_maze/dinov2_vits14_img224
+env.dataset.slice_stride=1
```

`wm_poc_latent_dataset.py` (installed into the upstream checkout) subclasses
the upstream `PointMazeDataset`, so actions, states, proprioception,
normalization statistics, and the seed-42 train/val split are bit-identical to
the online path. Its slicer mirrors `TrajSlicerDataset` but memmap-reads only
the ~0.6 MB of latents each 4-frame sample needs, instead of materializing a
~13 MB episode. It also resolves `TrajSubset` wrappers to base indices at
construction (upstream's `TrajSubset.__getattr__` recurses if it is ever
pickled to a spawn-context DataLoader worker).

`slice_stride` (wrapper config `dataset.slice_stride`) optionally subsamples
window start positions; stride 1 reproduces the upstream window set exactly.

### 3. Encoder bypass in the model

`scripts/dino_wm/patch_latent_cache.py` applies a marker-guarded patch to
`models/visual_world_model.py`: `encode_obs` dispatches on input shape. Image
batches (5-D) take the original transform+encoder path; precomputed latents
(4-D, `(b, t, patches, emb_dim)`) skip straight to the predictor. The real
DINO encoder stays in the model and in checkpoints, so `plan.py`, smoke runs,
and any image-based evaluation work unchanged.

### 4. Dataloader workers restored

The full no-decoder config no longer inherits the OOM-probe's
`env.num_workers=0`; workers come from `training.num_workers` (default 4).
The OOM-safe and smoke configs explicitly set `features.cache_enabled: false`
so they keep exercising the online image path as diagnostics.

## Storage layout

```
${DINO_WM_FEATURE_CACHE:-/content/wm_poc_latent_cache}/
  point_maze/
    feature_cache_manifest.yaml          # wrapper-level run metadata
    dinov2_vits14_img224/
      wm_poc_latent_manifest.json        # cache identity + episode lengths
      episode_000.npy ... episode_2199.npy   # (T, 196, 384) float16
```

The full PointMaze cache is ~30 GB, so it defaults to fast local disk and is
recomputed per Colab session (the precompute is the only heavy Drive read).
Point `DINO_WM_FEATURE_CACHE` at Drive only if that space is available there.

## Measured budget (full 2,200-rollout config, batch 32)

| Phase | Old (online, workers=0, T4) | New latent cache, T4 fp16 (measured) | New latent cache, A100 bf16 (expected) |
| --- | --- | --- | --- |
| Precompute | n/a | ~30-60 min first session | ~10-20 min first session |
| Train epoch (4,602 steps) | 30+ h (~16 s/it) | ~55-60 min (1.32 it/s) | minutes (~5-10x T4 step rate) |
| 10-epoch run | impossible | ~9-10 h, or ~4-5 h with stride 2 | comfortably inside 2-3 h |

The measured T4 rate is ~21x the old path and is now genuinely
GPU-compute-bound: each step pushes batch 32 through the depth-6 ViT
predictor over 588 tokens (~2 TFLOPs forward+backward), which is what a T4
realistically sustains at ~0.7-0.8 s/step. That predictor cost is the thing
the experiment is supposed to measure, so further T4 speedups come from doing
fewer steps, not cheaper ones: `configs/dino_wm/pointmaze_full_nodecoder_t4.yaml`
sets `dataset.slice_stride: 2` (consecutive windows overlap by 19/20 frames,
so halving the window set loses little diversity) and `num_workers: 2` for
the 2-vCPU T4 VMs, giving ~30 min/epoch. Stride 4 would halve it again if a
T4-only schedule is needed. `DINO_NUM_WORKERS` overrides the worker count at
launch without editing configs.

## Numerical note

Cached latents are computed under bf16 autocast and stored as fp16, then cast
to fp32 at load. This matches the precision regime of the bf16 training run
itself; expect bit-level differences from a fully fp32 online run, but not
behaviorally meaningful ones for predictor training.

## Operational notes (from the first T4 relaunch)

The first relaunch after the latent-cache change looked unchanged ("Epoch 1
Train ... 15.9s/it, 19h projected") but was actually three stacked artifacts,
all now handled:

1. **Orphaned trainer from an interrupted cell.** Stopping the notebook cell
   sent SIGTERM to the process group, but a second interrupt during the 20 s
   grace wait skipped the SIGKILL escalation, and wandb traps SIGTERM — so the
   old image-path trainer survived, kept writing its progress bar into the run
   dir, and held ~13.7 GB of T4 memory against the new run. The monitor now
   escalates to SIGKILL even on a second interrupt, and every launch first
   kills leftover `dino_wm/train.py` / `wm_poc_precompute_latents.py`
   processes (`DINO_KILL_STALE=1`, the default; set `0` to get an error
   instead).
2. **Ghost logs.** `stdout.log`/`stderr.log` are only truncated when the train
   stage starts, so the live panel tailed the dead run's progress bars while
   the new launcher was still in verify/precompute. Launches now rotate
   non-empty logs to `*.prev` first; if you see `Epoch N Train` lines, they
   are from the current run.
3. **bf16 on T4.** Turing has no native bf16; autocast emulates it slowly. The
   notebook cell now picks bf16 on compute capability >= 8.0 (A100/L4) and
   fp16 otherwise, matching the earlier T4 guidance.

A fourth artifact surfaced at the first epoch boundary of the full T4 run:
**CUDA OOM at validation batch 2** in the predictor attention, after training
itself ran fine. Upstream `Trainer.val()` runs the model forward with
autograd enabled and never backpropagates, so each validation batch builds a
full graph and the previous batch's graph is still alive during the next
forward — two batch-32 graphs on top of optimizer state, which a 16 GB T4
cannot hold (the authors' 80 GB H100s never noticed). `patch_val_no_grad.py`
wraps the validation forward in `torch.no_grad()`, applied by `run_train.sh`
alongside the other patches. Memory is flat across epochs with this in
place; the OOM was a per-validation-pass effect, not accumulation.

Also: training has genuinely not started until the monitor shows
`Encoded latents for episode .../...` finishing. On a T4 the first session
spends its first ~30-60 minutes on dependency install + reading the raw
episodes from Drive for precompute; subsequent epochs are then ~10-15 minutes
(fp16). If the latent cache directory is under `/content/drive`, both the
precompute script and the dataset print a warning — random latent reads
through the Drive FUSE mount would reproduce the original ~16 s/it
bottleneck.

### torch/torchvision mismatch on Colab

If the smoke or precompute fails at the encoder import with `RuntimeError:
operator torchvision::nms does not exist`, Colab's preinstalled torch and
torchvision are out of sync — the legacy DINO-WM pins (`numpy==1.26.4`, old
`gym`/`d4rl`/`mujoco-py`, etc.) shifted `torch`, leaving its `torchvision`
compiled against a different build. `scripts/dino_wm/install_colab_deps.py` now
self-heals this: after the dependency install it imports `torchvision` in a
subprocess and, if broken, reinstalls the matching release (torch 2.N pairs with
torchvision 0.(N+15)) with `--no-deps` so `torch` is untouched. Set
`DINO_TORCHVISION_SPEC=torchvision==<ver>` to override the derived pin for an
unusual torch build. To repair a runtime by hand without re-running setup:

```python
import torch, subprocess, sys
maj, minr = torch.__version__.split("+")[0].split(".")[:2]
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                "--force-reinstall", f"torchvision==0.{int(minr) + 15}.0"], check=True)
```

## Checkpoint cadence and crash recovery

A100 time is paid time, so both checkpoint levels are armed by default and
both knobs are configurable from the notebook preflight cell:

- **Every `SAVE_EVERY_EPOCHS` epochs (default 1)** the upstream trainer
  writes `model_latest.pth` plus `model_<epoch>.pth` to
  `<ckpt_root>/outputs/<run_name>/checkpoints/` on Drive
  (`training.save_every_epochs`, env `DINO_SAVE_EVERY_EPOCHS`).
- **Every `SAVE_EVERY_STEPS` optimizer steps** the step-checkpoint patch
  writes a rolling `model_latest_step.pth` (CPU state_dicts, optimizer
  state, RNG state, and the batch index) inside the epoch
  (`training.save_every_steps`, env `DINO_SAVE_EVERY_STEPS`, patch armed by
  `DINO_PATCH_STEP_CHECKPOINTING=1`; 0 disables and restores upstream).
  Defaults: 2000 steps on A100 (~3-4 min of work at risk) and 500 on T4
  (~6 min). Each write is a few hundred MB to Drive, so very small values
  mostly buy Drive churn.

The wall clock is the wrapper's own `timeout` around the train command —
`DINO_MAX_WALL_MINUTES` (env, set per launch cell) overriding
`training.max_wall_minutes` (config, sanity-capped at 480) — not anything
internal to the GPU or Colab. The genuinely external constraint is the Colab
session itself (free-tier sessions rarely survive many hours and idle tabs
disconnect); the checkpoint cadence below exists precisely so a long T4
schedule can be run in chunks across session interruptions.

Recovery after a disconnect is just re-running the same launch cell: the
launcher kills any leftover trainer, re-runs the (resumable, mostly no-op)
precompute to restore the session-local latent cache, and the trainer then
resumes from `model_latest_step.pth` if present — skipping forward to the
exact batch index with RNG restored, which is deterministic because the
slice permutation is seeded — or from `model_latest.pth` otherwise. The
rolling step file is deleted whenever a full epoch checkpoint lands. Set
`DINO_FORCE_RESTART=1` to start over instead (existing checkpoints are
moved aside, not deleted). Details of the patch live in
`docs/dino_wm_safe_intra_epoch_checkpointing_handoff.md`.

## Notebook flow (02_dino_wm_foundation.ipynb)

The notebook is the central pipeline for training and fine-tuning:

1. **Run Profile and Preflight** — one cell sets the launch environment for
   everything below: GPU detection picks bf16/fp16 and the stride-1 vs
   stride-2 full-run config (`CONFIG_FULL`), pins the latent cache to local
   disk, arms stale-process cleanup, and checks free disk space.
2. **Smoke Run (latent-path crash test)** — `smoke_pointmaze_latent.yaml`
   runs the exact full-run pipeline on 20 rollouts in ~10-15 min: patches,
   DINO hub download, latent precompute + manifest, one no-decoder latent
   training epoch with workers, checkpoint/summary writes, and a tiny CEM
   planning eval that loads the checkpoint and drives the real encoder.
   The 20 encoded episodes are reused by the full cache (precompute resumes).
   The legacy online-encoding smoke (`smoke_pointmaze.yaml`) remains
   available for diagnosing the image path itself.
3. **Full no-decoder run** — launches `CONFIG_FULL` from the preflight.
4. **Low-data scratch + fine-tune** — the scratch baseline and a fine-tune
   cell that auto-resolves the source checkpoint (latest checkpoint of the
   full run, or `DINO_POINTMAZE_SOURCE_CKPT`) and launches
   `run_finetune.sh` through the same monitored pipeline.
5. **Planning evaluation and summary table** — standalone planning from a
   checkpoint (needs the latent cache present in fresh sessions, since the
   saved train config references the latent dataset) and the cross-run
   summary CSV.

## Knobs

- `features.cache_enabled` — route training through the latent cache
  (point_maze only for now; pusht/wall would need a latent loader for their
  video-backed datasets).
- `features.precompute_batch_size` — encoder batch during precompute.
- `training.num_workers` — dataloader workers (`env.num_workers` upstream);
  `DINO_NUM_WORKERS` overrides at launch.
- `dataset.slice_stride` — subsample slice starts; 2 halves steps per epoch.
- `DINO_WM_FEATURE_CACHE` — cache root override.
- `configs/dino_wm/pointmaze_full_nodecoder_t4.yaml` — T4 profile (stride 2,
  2 workers, fp16 via the notebook's auto-selection).
