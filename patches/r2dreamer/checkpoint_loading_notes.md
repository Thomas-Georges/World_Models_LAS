# Checkpoint Loading Patch Notes

`scripts/r2dreamer/patch_checkpoint_loading.py` updates external `train.py` so that a target run can initialize from a source checkpoint.

Added Hydra keys:

```text
+pretrained=/path/to/latest.pt
+pretrained_strict=true
+load_optimizer=false
```

Main fine-tuning comparison uses `load_optimizer=false` to transfer network weights while resetting optimizer state.

The patch also wraps training in `try/finally` so `latest.pt` is written after interruption once the agent exists.
