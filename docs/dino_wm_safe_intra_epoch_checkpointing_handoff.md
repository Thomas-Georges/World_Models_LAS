# DINO-WM Safe Intra-Epoch Checkpointing Handoff

## Context

The DINO-WM experiment track is implemented in this repository without touching
the R2Dreamer track. The current `main` branch contains DINO-WM wrappers,
configs, live monitoring, dependency setup, and epoch-level checkpoint resume.

Important recent commits:

- `cde259a Add DINO-WM checkpoint resume controls`
- `6fedb37 Add DINO-WM intra-epoch checkpointing`
- `436886b Make DINO-WM step checkpoint patch opt-in`

The intra-epoch checkpoint patch from `6fedb37` is now disabled by default after
an A100 run hit CUDA OOM very early in epoch 1. The OOM happened before the first
configured step checkpoint, but the patch changed the upstream training loop and
is therefore treated as suspect until a safer design is implemented.

Current safe default after `436886b`:

```python
os.environ["DINO_PATCH_STEP_CHECKPOINTING"] = "0"
os.environ.pop("DINO_SAVE_EVERY_STEPS", None)
```

With this default, the wrapper restores upstream `train.py` from the backup made
before the step-checkpoint patch. Epoch-level checkpointing still works.

## Problem

Epoch-level checkpoints are not enough if a single epoch takes a long time. In
the PointMaze A100 run, epoch 1 had about 4602 train batches. A Colab disconnect
or CUDA failure before the first epoch checkpoint loses all progress in that
epoch.

We need frequent intra-epoch checkpoints without increasing GPU memory pressure
or serializing live CUDA module objects.

## Why The First Patch Is Risky

The first intra-epoch patch reused upstream `save_ckpt()`, which saves whole
Python module objects:

```python
ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
torch.save(ckpt, "checkpoints/model_latest.pth")
```

That may be acceptable at epoch boundaries, but it is not ideal for frequent
mid-epoch saves because it serializes live module objects and CUDA-backed state.
The safer design should save CPU `state_dict()` snapshots instead.

## Desired Design

Implement a second-generation DINO-WM step checkpoint patch with these
properties:

1. It is opt-in behind `DINO_PATCH_STEP_CHECKPOINTING=1`.
2. It overwrites one rolling checkpoint file only.
3. It saves model and optimizer `state_dict()` values, not live module objects.
4. It moves tensor values to CPU before `torch.save`.
5. It writes atomically: save to `model_latest_step.pth.tmp`, then replace
   `model_latest_step.pth`.
6. It stores enough metadata to resume the correct epoch and batch:
   `epoch`, `batch_index`, `global_step`, `resume_batch_index`, and ideally RNG
   states.
7. On resume, it rebuilds model/optimizers normally, loads the saved
   `state_dict()` values, starts at the saved epoch, and skips batches through
   `resume_batch_index`.
8. It should not create `model_step_*.pth` files.
9. It should not change R2Dreamer files.

## Proposed File Changes

Likely files to edit:

- `src/wm_poc/dino_wm/step_checkpoint_patch.py`
- `scripts/dino_wm/patch_step_checkpointing.py`
- `scripts/dino_wm/run_train.sh`
- `scripts/dino_wm/setup_dino_wm.sh`
- `src/wm_poc/dino_wm/commands.py`
- `configs/dino_wm/base.yaml`
- `configs/dino_wm/README.md`
- `tests/test_dino_wm.py`

Do not edit:

- Any `r2dreamer` files.
- Notebook output unless explicitly requested.

## Implementation Sketch

Patch upstream `train.py` to add a separate step-checkpoint path, not to reuse
the existing full-object `save_ckpt()`.

Suggested upstream methods to inject into `Trainer`:

```python
def _cpu_state_dict(self, obj):
    unwrapped = self.accelerator.unwrap_model(obj) if hasattr(obj, "module") else obj
    state = unwrapped.state_dict()
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def save_step_ckpt(self, batch_index):
    self.accelerator.wait_for_everyone()
    if not self.accelerator.is_main_process:
        return
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "format": "wm_poc_dino_step_state_dict_v1",
        "epoch": int(self.epoch),
        "batch_index": int(batch_index),
        "resume_batch_index": int(batch_index),
        "global_step": int(getattr(self, "train_step", 0)),
        "epoch_log": self.epoch_log,
    }
    if self.train_encoder:
        ckpt["encoder"] = self._cpu_state_dict(self.encoder)
        ckpt["encoder_optimizer"] = self.encoder_optimizer.state_dict()
    if self.cfg.has_predictor and self.train_predictor:
        ckpt["predictor"] = self._cpu_state_dict(self.predictor)
        ckpt["predictor_optimizer"] = self.predictor_optimizer.state_dict()
        ckpt["action_encoder"] = self._cpu_state_dict(self.action_encoder)
        ckpt["proprio_encoder"] = self._cpu_state_dict(self.proprio_encoder)
        ckpt["action_encoder_optimizer"] = self.action_encoder_optimizer.state_dict()
    if self.cfg.has_decoder and self.train_decoder:
        ckpt["decoder"] = self._cpu_state_dict(self.decoder)
        ckpt["decoder_optimizer"] = self.decoder_optimizer.state_dict()
    tmp_path = "checkpoints/model_latest_step.pth.tmp"
    final_path = "checkpoints/model_latest_step.pth"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, final_path)
    log.info(
        "Saved rolling state_dict checkpoint at epoch %s step %s batch %s to %s",
        self.epoch,
        getattr(self, "train_step", 0),
        batch_index,
        final_path,
    )
```

Resume logic should prefer `model_latest_step.pth` if it exists and has
`format == "wm_poc_dino_step_state_dict_v1"`. It should then load state dicts
after models and optimizers are instantiated:

```python
def load_step_ckpt(self, filename):
    ckpt = torch.load(filename, map_location="cpu", weights_only=False)
    self.epoch = int(ckpt["epoch"])
    self.train_step = int(ckpt.get("global_step", 0))
    self.resume_batch_index = int(ckpt.get("resume_batch_index", ckpt["batch_index"]))
    self.epoch_log = ckpt.get("epoch_log", OrderedDict())
    if "predictor" in ckpt:
        self.accelerator.unwrap_model(self.predictor).load_state_dict(ckpt["predictor"])
    if "action_encoder" in ckpt:
        self.accelerator.unwrap_model(self.action_encoder).load_state_dict(ckpt["action_encoder"])
    if "proprio_encoder" in ckpt:
        self.accelerator.unwrap_model(self.proprio_encoder).load_state_dict(ckpt["proprio_encoder"])
    if "decoder" in ckpt:
        self.accelerator.unwrap_model(self.decoder).load_state_dict(ckpt["decoder"])
    if "predictor_optimizer" in ckpt:
        self.predictor_optimizer.load_state_dict(ckpt["predictor_optimizer"])
    if "action_encoder_optimizer" in ckpt:
        self.action_encoder_optimizer.load_state_dict(ckpt["action_encoder_optimizer"])
    if "decoder_optimizer" in ckpt:
        self.decoder_optimizer.load_state_dict(ckpt["decoder_optimizer"])
```

Be careful with `Accelerator.prepare()`: optimizer and model objects must exist
before loading optimizer state. This likely means:

1. Keep upstream full-object `model_latest.pth` loading behavior unchanged for
   epoch checkpoints.
2. Add step-checkpoint loading after `init_models()` and `init_optimizers()`.
3. If step checkpoint is loaded, skip the old full-object `load_ckpt()` path or
   only use it when no step checkpoint exists.

## Runtime Controls

Recommended defaults:

```yaml
training:
  save_every_epochs: 1
  save_every_steps: 0
```

Opt in from Colab only when testing:

```python
os.environ["DINO_PATCH_STEP_CHECKPOINTING"] = "1"
os.environ["DINO_SAVE_EVERY_STEPS"] = "100"
```

Return to safe upstream behavior:

```python
os.environ["DINO_PATCH_STEP_CHECKPOINTING"] = "0"
os.environ.pop("DINO_SAVE_EVERY_STEPS", None)
```

## Validation Plan

Do not run long GPU jobs in Codex.

Required lightweight checks:

```bash
python scripts/verify_environment.py --cpu-only
python scripts/verify_drive_layout.py --dry-run
pytest -q
```

Add focused tests for:

- Patch is idempotent on a synthetic upstream-like `train.py`.
- Restore returns patched upstream source back to the pre-patch backup.
- Rendered train command omits `++training.save_every_steps` by default.
- Rendered train command includes `++training.save_every_steps=<N>` when
  `DINO_SAVE_EVERY_STEPS=N`.
- The generated patched source compiles with `python -m py_compile` against a
  temporary copy of the upstream `train.py`.

## Manual Colab Test Sequence

1. Pull latest `main`.
2. Restart runtime after the prior OOM.
3. Confirm safe restore first:

```python
os.environ["DINO_PATCH_STEP_CHECKPOINTING"] = "0"
os.environ.pop("DINO_SAVE_EVERY_STEPS", None)
```

4. Run the DINO setup/train path and confirm it prints either:

```text
Restored upstream DINO-WM train.py from step-checkpointing backup
```

or:

```text
DINO-WM step checkpointing patch is not applied; no restore needed
```

5. Re-run PointMaze scratch without step checkpointing and confirm the early OOM
   disappears.
6. Only after that, test the new CPU-state-dict patch with a very small smoke
   run and `DINO_SAVE_EVERY_STEPS=2`.

## Open Questions

- Does upstream DINO-WM require optimizer state for useful resume, or is model
  state enough for the first safe version?
- Should step checkpoints be saved to `model_latest_step.pth` and copied to
  `model_latest.pth`, or should resume explicitly prefer the step file while
  preserving upstream `model_latest.pth` semantics?
- Should RNG state be included for exact deterministic resume? It is useful, but
  not mandatory for crash recovery.
- If the Dataloader uses multiprocessing workers, skipping batches may still
  take time. That is acceptable for correctness but should be noted in monitor
  output.
