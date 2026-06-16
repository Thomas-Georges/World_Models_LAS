# DINO-WM Patches

This directory documents patches applied to the external `gaoyuezhou/dino_wm`
checkout.

Patch scripts live in `scripts/dino_wm/` and modify only the external checkout
selected by `DINO_WM_REPO` or the wrapper config. The external repository is not
committed here.

Current patches and installed files:

- `patch_mixed_precision.py` — wires `DINO_MIXED_PRECISION` into the
  `Accelerator` constructor in `train.py`.
- `patch_step_checkpointing.py` — opt-in intra-epoch state_dict checkpoints in
  `train.py` (`DINO_PATCH_STEP_CHECKPOINTING=1`).
- `patch_latent_cache.py` — installs `wm_poc_latent_dataset.py` and
  `wm_poc_precompute_latents.py` into the checkout root and adds an
  input-dispatch bypass to `models/visual_world_model.py` so precomputed
  latent batches skip the frozen DINO forward pass. See
  `docs/dino_wm_latent_cache_training.md`.

All patches are marker-guarded and idempotent; originals are backed up under
`.wm_poc_backups/` next to the patched file.

- `patch_val_no_grad.py` — wraps the upstream validation forward in
  `torch.no_grad()`; upstream builds (and keeps two of) full autograd graphs
  per validation batch, which OOMs 16 GB GPUs at the first epoch boundary.

- `patch_finetune_loading.py` — appends a fine-tune init hook to
  `init_models` in `train.py`: fresh runs with `++finetuning.enabled=true`
  load predictor/action-encoder/proprio-encoder (optionally decoder) weights
  from `++finetuning.init_from`; resumed runs skip it. Fine-tune learning
  rates and epochs are mapped onto plain `training.*` overrides by the
  command builder, so upstream needs no schema changes.

- `patch_evaluator_video.py` — adds a no-decoder branch to
  `planning/evaluator.py` video recording: executed rollout vs goal with the
  imagined panel zeroed out, so decoder-free checkpoints get planning videos.
