#!/usr/bin/env python
"""Train the local latent-dynamics surrogate on cached DINO-WM latents.

Smoke runs (``--smoke``) are always allowed and finish in seconds on CPU with
the synthetic task. Full runs are gated on ``RUN_LOCAL_GLOBAL=1`` like the
other heavy scripts in this repository.

Example:
    python scripts/local_global/train_local_surrogate.py \
        --config configs/local_global/pointmaze_surrogate_a100.yaml \
        --run-dir "$LG_RUN_ROOT/pointmaze_local_v1"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wm_poc.dino_wm.configs import get_config_value, set_config_value  # noqa: E402
from wm_poc.local_global.configs import (  # noqa: E402
    RUN_GATE_ENV,
    action_data_dir,
    latent_cache_dir,
    load_local_global_config,
    resolve_run_dir,
    save_resolved_config,
    typed_config,
)
from wm_poc.local_global.datasets import (  # noqa: E402
    LatentTrajectoryStore,
    LatentWindowDataset,
    collate_latent_windows,
    split_store_episodes,
)

SMOKE_OVERRIDES = (
    ("training.max_steps", 200),
    ("training.batch_size", 16),
    ("training.val_every", 100),
    ("training.save_every", 100),
    ("training.num_workers", 0),
    ("training.max_episodes", 8),
    ("training.max_windows", 512),
)


def resolve_device(config_device: str, override: str | None) -> str:
    import torch

    requested = override or config_device or "auto"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def loss_weights(config: dict) -> dict[str, float]:
    return {
        "lambda_rollout": float(get_config_value(config, "training.lambda_rollout", 1.0)),
        "lambda_one_step": float(get_config_value(config, "training.lambda_one_step", 1.0)),
        "lambda_delta": float(get_config_value(config, "training.lambda_delta", 0.1)),
        "lambda_jacobian": float(get_config_value(config, "training.lambda_jacobian", 0.0)),
        "lambda_variance": float(get_config_value(config, "training.lambda_variance", 0.0)),
        "rollout_discount": float(get_config_value(config, "training.rollout_discount", 1.0)),
    }


def try_resume(run_dir, build_kwargs, model, optimizer, device):
    """Restore model/optimizer/step from local_latest.pt; returns (step, best_val).

    Interrupted sessions (Colab disconnects, wall-clock stops) continue from
    the last rolling checkpoint instead of restarting at step 0.
    """
    import torch

    latest = run_dir / "checkpoints" / "local_latest.pt"
    if not latest.is_file():
        return 0, float("inf")
    payload = torch.load(latest, map_location=device)
    if payload.get("build_kwargs") != build_kwargs:
        raise RuntimeError(
            f"{latest} was trained with different model settings than this config; "
            "pass --no-resume to start over or use a fresh --run-dir."
        )
    model.load_state_dict(payload["model_state"])
    if payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    step = int(payload.get("step", 0))
    best = float(payload.get("metrics", {}).get("best_val_loss", float("inf")))
    print(f"Resumed {latest} at step {step} (best val loss {best:.5f}).")
    return step, best


def run_validation(model, loader, device, weights, max_batches: int = 50):
    import torch

    from wm_poc.local_global.losses import combined_local_loss

    model.eval()
    totals: dict[str, float] = {}
    per_step_sums: list[float] = []
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            _, metrics = combined_local_loss(batch, model, weights)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            x_targets = model.encode_global_latent(batch["z_targets"])
            x_context = model.encode_global_latent(batch["z_context"])
            pred = model.rollout_from_context(
                x_context, batch["actions_context"], batch["actions"]
            )
            step_mse = torch.mean((pred - x_targets) ** 2, dim=(0, 2))
            if not per_step_sums:
                per_step_sums = [0.0] * step_mse.shape[0]
            for i, value in enumerate(step_mse.tolist()):
                per_step_sums[i] += value
            batches += 1
            if batches >= max_batches:
                break
    model.train()
    if batches == 0:
        return None
    metrics = {k: v / batches for k, v in totals.items()}
    metrics["rollout_mse_per_step"] = [v / batches for v in per_step_sums]
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true", help="tiny capped run, always allowed")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from <run-dir>/checkpoints/local_latest.pt when present",
    )
    args = parser.parse_args()

    config = load_local_global_config(args.config)
    if args.smoke:
        for key, value in SMOKE_OVERRIDES:
            set_config_value(config, key, value)
    if args.max_steps is not None:
        set_config_value(config, "training.max_steps", int(args.max_steps))

    max_steps = int(get_config_value(config, "training.max_steps"))
    print(f"Config: {args.config} | task={config['task']} | max_steps={max_steps}")
    if args.dry_run:
        print(f"Latent cache: {latent_cache_dir(config)}")
        print(f"Action data:  {action_data_dir(config)}")
        print("Dry run: not training.")
        return 0
    if not args.smoke and os.environ.get(RUN_GATE_ENV) != "1":
        print(
            f"Full training is disabled for safety. Set {RUN_GATE_ENV}=1 to launch, "
            "or pass --smoke for a tiny capped run."
        )
        return 0

    import torch
    from torch.utils.data import DataLoader

    from wm_poc.local_global.losses import combined_local_loss
    from wm_poc.local_global.models import build_local_model, save_local_checkpoint

    # The synthetic smoke path generates its own latent cache when missing.
    from wm_poc.local_global.datasets import ensure_synthetic_task_data

    ensure_synthetic_task_data(config)

    typed = typed_config(config)
    device = resolve_device(typed.device, args.device)
    torch.manual_seed(typed.seed)

    store = LatentTrajectoryStore(
        latent_cache_dir(config),
        action_data_dir(config),
        max_episodes=int(get_config_value(config, "training.max_episodes", 0)),
    )
    train_eps, val_eps = split_store_episodes(
        store,
        float(get_config_value(config, "training.val_fraction", 0.1)),
        int(get_config_value(config, "training.split_seed", 42)),
    )
    max_windows = int(get_config_value(config, "training.max_windows", 0))
    dataset_kwargs = dict(
        context_len=typed.local_model.context_len,
        rollout_steps=typed.local_model.rollout_steps,
        frameskip=typed.global_model.frameskip,
        max_windows=max_windows,
    )
    train_set = LatentWindowDataset(store, train_eps, **dataset_kwargs)
    val_set = LatentWindowDataset(store, val_eps, **dataset_kwargs)
    if len(train_set) == 0 or len(val_set) == 0:
        raise RuntimeError(
            f"Empty dataset (train={len(train_set)}, val={len(val_set)}); episodes too "
            "short for the configured context/rollout/frameskip."
        )
    loader_kwargs = dict(
        batch_size=int(get_config_value(config, "training.batch_size", 128)),
        num_workers=int(get_config_value(config, "training.num_workers", 2)),
        collate_fn=collate_latent_windows,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    build_kwargs = dict(
        patches=store.patches,
        embed_dim=store.embed_dim,
        step_action_dim=store.action_dim * typed.global_model.frameskip,
        model_type=typed.local_model.model_type,
        projection=typed.local_model.projection,
        projection_grid=typed.local_model.projection_grid,
        projection_trainable=typed.local_model.projection_trainable,
        local_dim=typed.local_model.local_dim,
        hidden_dim=typed.local_model.hidden_dim,
        num_layers=typed.local_model.num_layers,
        layer_norm=typed.local_model.layer_norm,
        seed=typed.seed,
    )
    model = build_local_model(**build_kwargs).to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(get_config_value(config, "training.lr", 3e-4)),
        weight_decay=float(get_config_value(config, "training.weight_decay", 1e-4)),
    )
    weights = loss_weights(config)
    max_grad_norm = float(get_config_value(config, "training.max_grad_norm", 10.0))
    val_every = int(get_config_value(config, "training.val_every", 500))
    save_every = int(get_config_value(config, "training.save_every", 1000))
    log_every = int(get_config_value(config, "training.log_every", 50))
    max_wall_minutes = float(get_config_value(config, "training.max_wall_minutes", 110))

    run_dir = resolve_run_dir(config, args.run_dir)
    save_resolved_config(config, run_dir)

    start_step, best_val = 0, float("inf")
    if args.resume:
        start_step, best_val = try_resume(run_dir, build_kwargs, model, optimizer, device)
    if start_step >= max_steps:
        print(f"Training already complete ({start_step}/{max_steps} steps); nothing to do.")
        print("Pass --no-resume (or use a fresh --run-dir) to retrain from scratch.")
        return 0

    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    train_log = (metrics_dir / "train_metrics.jsonl").open("a", encoding="utf-8")
    val_log = (metrics_dir / "val_rollouts.jsonl").open("a", encoding="utf-8")
    print(
        f"Run dir: {run_dir} | device={device} | params="
        f"{sum(p.numel() for p in params):,} trainable | "
        f"{len(train_set):,} train / {len(val_set):,} val windows"
    )

    def log_row(handle, row: dict) -> None:
        handle.write(json.dumps(row) + "\n")
        handle.flush()

    start_time = time.time()
    step = start_step
    stop = False
    while not stop:
        for batch in train_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            loss, metrics = combined_local_loss(batch, model, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()
            step += 1
            if step % log_every == 0 or step == 1:
                log_row(train_log, {"step": step, "split": "train", **metrics})
            if step % val_every == 0 or step == max_steps:
                val_metrics = run_validation(model, val_loader, device, weights)
                if val_metrics is not None:
                    per_step = val_metrics.pop("rollout_mse_per_step")
                    log_row(train_log, {"step": step, "split": "val", **val_metrics})
                    log_row(
                        val_log,
                        {
                            "step": step,
                            "rollout_mse_per_step": per_step,
                            "one_step_mse": val_metrics.get("loss_one_step"),
                        },
                    )
                    rate = (step - start_step) / max(time.time() - start_time, 1e-9)
                    eta_min = (max_steps - step) / max(rate, 1e-9) / 60
                    print(
                        f"step {step}/{max_steps} | train loss {metrics['loss_total']:.5f} | "
                        f"val loss {val_metrics['loss_total']:.5f} | "
                        f"val vs-static {val_metrics.get('rollout_mse_vs_static', float('nan')):.3f} | "
                        f"{rate:.2f} steps/s | ETA {eta_min:.0f} min"
                    )
                    if val_metrics["loss_total"] < best_val:
                        best_val = val_metrics["loss_total"]
                        save_local_checkpoint(
                            run_dir / "checkpoints" / "local_best.pt",
                            model,
                            build_kwargs,
                            step=step,
                            metrics=val_metrics,
                        )
            if step % save_every == 0 or step == max_steps:
                save_local_checkpoint(
                    run_dir / "checkpoints" / "local_latest.pt",
                    model,
                    build_kwargs,
                    step=step,
                    metrics={"best_val_loss": best_val},
                    optimizer_state=optimizer.state_dict(),
                )
            elapsed_minutes = (time.time() - start_time) / 60
            if step >= max_steps or elapsed_minutes > max_wall_minutes:
                if elapsed_minutes > max_wall_minutes:
                    print(f"Stopping: wall-clock limit {max_wall_minutes} min reached.")
                stop = True
                break

    save_local_checkpoint(
        run_dir / "checkpoints" / "local_latest.pt",
        model,
        build_kwargs,
        step=step,
        metrics={"best_val_loss": best_val},
        optimizer_state=optimizer.state_dict(),
    )
    train_log.close()
    val_log.close()
    print(
        f"Done: {step} steps in {(time.time() - start_time) / 60:.1f} min | "
        f"best val loss {best_val:.5f} | checkpoints under {run_dir / 'checkpoints'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
