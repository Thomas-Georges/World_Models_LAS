# DINO-WM Full Training Throughput Conclusion

## Summary

The OOM remediation worked as a memory fix, but the full PointMaze run is now
limited by throughput. The T4 logs show that increasing batch size from 4 to 32
reduces the number of optimizer steps per epoch by roughly 8x, as expected, but
does not make the full epoch cheap enough. Larger batches do more work per step,
so total epoch wall time can remain very high.

This means the current full no-decoder run is a useful viability probe, not yet
the final practical training loop.

## Observed Behavior

The slow T4 run reported roughly:

- `batch_size=4`: about 36,815 train steps per epoch.
- `batch_size=32`: about 4,602 train steps per epoch.
- Even after the step-count reduction, projected epoch time remained too high for
  ordinary iteration.

The step-count drop confirms that the wrapper config reaches upstream Hydra
correctly. The remaining problem is sample throughput.

## Root Cause

The main cost is that frozen DINO features are still recomputed online every
epoch. In upstream DINO-WM, `model.train_encoder=false` freezes the DINO encoder
weights, but it does not skip the encoder forward pass. Each training batch still
pushes image frames through the DINO patch encoder and predictor path.

Disabling the decoder reduces peak memory, but it does not remove the online
DINO feature extraction cost.

There is also a throughput-specific issue in the current no-decoder config: it
inherits the OOM-safe override:

```yaml
upstream:
  train_overrides:
    - env.num_workers=0
```

That was appropriate for the first CUDA OOM debugging probe because it removed
dataloader concurrency as a confounder. It is not appropriate for full training,
especially when reading from Google Drive. Low GPU utilization during the run is
consistent with input/data or CPU-side bottlenecks.

## Practical Interpretation

The T4 full-data run should be stopped. It has already shown the important
result: full-data no-decoder training can start without immediate CUDA OOM. It
also shows that T4 is not a practical target for full DINO-WM PointMaze training
with online DINO feature extraction.

The A100 run should not be treated as solved purely by setting `batch_size=32`.
It should first be used for a short throughput probe.

## Recommended Next Steps

1. Keep the OOM-safe config as the memory diagnostic.
2. Keep the full no-decoder config as the A100-oriented path, but remove
   `env.num_workers=0` or replace it with `env.num_workers=2` or `4`.
3. Run a short A100 throughput probe before committing to a 10-epoch run.
4. Use `DINO_MIXED_PRECISION=bf16` on A100.
5. Use `DINO_MIXED_PRECISION=fp16` on T4 if a small smoke or low-data run is
   needed there.
6. Do not use T4 for the full 2,200-rollout online-DINO training loop.

To reach a 2-4 hour full experiment target, batch size alone is unlikely to be
enough. The next real speedup should come from one or more of:

- real latent caching, so DINO patch features are computed once and reused;
- fewer sliced windows per epoch, via fewer rollouts or a stride/window cap;
- local disk staging instead of reading training data from Drive;
- dataloader workers tuned for the GPU runtime;
- fewer initial epochs followed by planning evaluation before extending.

## Resolution (2026-06-10)

Implemented in this repository; see `docs/dino_wm_latent_cache_training.md`.
Profiling upstream showed the dominant cost was not the encoder forward alone:
`TrajSlicerDataset` reloads and re-transforms the entire episode image file
for every 4-frame sliced sample. The fix combines real latent caching
(one-time frozen-DINO precompute, memmap latent slices at train time, an
`encode_obs` bypass patch) with restored dataloader workers. The
`pointmaze_full_nodecoder_bf16` config now trains on cached latents and the
notebook cell runs the precompute stage automatically (`skip_cache=False`).

## Current Role of the Full No-Decoder Cell

The `PointMaze Full No-Decoder BF16 Run` notebook cell is best understood as a
main-loop viability and throughput calibration step. It is meant to answer:

- Does full-data training start without first-batch OOM?
- What batch size fits on the target GPU?
- Is epoch throughput acceptable?
- Are checkpoints, logs, and summaries written correctly?

It is not yet proof that the final scientific run will complete in the desired
2-4 hour window.
