# AGENTS.md

## Project goal

This repository supports a PyTorch proof of concept for world models, predictive control, and reinforcement learning.

The intended experiments are:

1. DreamerV3 / R2-Dreamer fine-tuning:
   - train a model,
   - save/reload checkpoints,
   - adapt to a shifted task or dataset,
   - compare fine-tuning against training from scratch.

2. Local/global world-model planning:
   - use a larger latent world model for prediction/planning,
   - train a smaller local surrogate for differentiable dynamics,
   - compare CEM-style zero-order planning with first-order planning through the local model.

## Hard constraints

- Do not run long training jobs unless explicitly asked.
- Do not download large datasets unless explicitly asked.
- Do not commit datasets, checkpoints, TensorBoard logs, videos, or model weights.
- Keep code PyTorch-based.
- Keep paths configurable through environment variables.
- Save large artifacts to Google Drive, not GitHub.
- Prefer small smoke tests before long runs.
- Never include secrets, tokens, API keys, or personal credentials.

## Code style

- Use typed Python where practical.
- Use `pathlib.Path` for filesystem paths.
- Make scripts runnable from the repository root.
- Prefer explicit command-line arguments.
- Keep functions small and testable.
- Use clear error messages when expected folders or environment variables are missing.

## Validation

Before proposing changes, run lightweight checks only:

```bash
python scripts/verify_environment.py --cpu-only
python scripts/verify_drive_layout.py --dry-run
pytest -q
```

Do not run GPU training as part of ordinary Codex tasks.
