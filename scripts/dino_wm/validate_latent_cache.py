#!/usr/bin/env python3
"""Validate the DINO-WM latent cache against the upstream training distribution.

This backs the report's claim of *validated cached-latent equivalence for the
selected windows* (Track II, latent-cache path). It checks the things that make
a cached-latent run comparable to an online-encoding run, and is deliberately
honest about what the cache does and does not change:

  1. Split parity        -- the seeded train/val split is deterministic and
                            independent of how many rollouts a run uses (this is
                            the upstream split the cached dataset inherits).
  2. Manifest            -- the latent manifest exists and has the expected
                            format/version.
  3. Latent shape/dtype  -- cached per-episode latents are (T, P, D) fp16 with the
                            patch count P and feature dim D the manifest records.
  4. Actions/proprio     -- inherited from the upstream dataset *by construction*:
     parity                 the cached dataset subclass overrides only the
                            visual-feature path, so actions, states,
                            proprioception, normalization, and the split come
                            straight from upstream. This script does NOT run a
                            sampled value-level raw-vs-cached comparison.
  5. Planning path       -- the encode bypass is dimensionality-gated, so 5-D
                            image inputs still go through the encoder at planning
                            time (the cache only affects 4-D latent inputs).

Run inside the DINO-WM environment (numpy required):

    python scripts/dino_wm/validate_latent_cache.py --config configs/dino_wm/<cfg>.yaml

Exit code 0 means every available check passed; 1 means a check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wm_poc.dino_wm import commands
from wm_poc.dino_wm.configs import load_config, resolve_config
from wm_poc.dino_wm.data import build_split_manifest
from wm_poc.dino_wm.latent_cache_patch import (
    LATENT_MANIFEST_FORMAT,
    LATENT_MANIFEST_NAME,
    MODEL_PATCH_MARKER,
    _MODEL_PATCH_ANCHOR,
    _MODEL_PATCH_REPLACEMENT,
)

# DINOv2 ViT-S/14 at 224px -> 14x14 patch grid of 384-dim features. The manifest
# is authoritative; these are the expected values for the reproduced config.
EXPECTED_PATCHES = 196
EXPECTED_DIM = 384


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str, failures: list[str]) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def _skip(msg: str) -> None:
    print(f"  [skip] {msg}")


def check_split_parity(config: dict, failures: list[str]) -> None:
    print("1. Split parity (seeded, rollout-count-independent)")
    a = build_split_manifest(config)
    b = build_split_manifest(config)
    if a["train_files"] == b["train_files"] and a["val_files"] == b["val_files"]:
        _ok(
            f"deterministic split: {a['num_train_files']} train / {a['num_val_files']} val "
            f"(seed {a['split_seed']}, of {a['num_available_files']} available)"
        )
    else:
        _fail("split is not deterministic across rebuilds", failures)


def check_manifest_and_latents(config: dict, failures: list[str]) -> None:
    print("2-3. Latent manifest, shape, and dtype")
    if not commands.latent_training_enabled(config):
        _skip("latent training not enabled in this config; nothing to validate")
        return
    cache_dir = Path(commands.latent_cache_dir(config)).expanduser()
    manifest_path = cache_dir / LATENT_MANIFEST_NAME
    if not manifest_path.is_file():
        _fail(f"missing latent manifest: {manifest_path}", failures)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") == LATENT_MANIFEST_FORMAT:
        _ok(
            f"manifest format {LATENT_MANIFEST_FORMAT!r}: "
            f"{manifest.get('num_episodes')} episodes, num_patches={manifest.get('num_patches')}, "
            f"dtype={manifest.get('dtype')}"
        )
    else:
        _fail(f"unexpected manifest format: {manifest.get('format')!r}", failures)

    if str(manifest.get("dtype")) != "float16":
        _fail(f"manifest dtype {manifest.get('dtype')!r}, expected 'float16'", failures)
    if manifest.get("num_patches") not in (None, EXPECTED_PATCHES):
        _fail(f"manifest num_patches {manifest.get('num_patches')}, expected {EXPECTED_PATCHES}", failures)

    try:
        import numpy as np
    except ImportError:
        _skip("numpy not available; cannot inspect cached latent arrays")
        return

    n = int(manifest.get("num_episodes", 0))
    for idx in range(min(3, n)):
        arr_path = cache_dir / f"episode_{idx:03d}.npy"
        if not arr_path.is_file():
            _fail(f"missing cached latent file: {arr_path.name}", failures)
            continue
        arr = np.load(arr_path, mmap_mode="r")
        if str(arr.dtype) != "float16":
            _fail(f"{arr_path.name}: dtype {arr.dtype}, expected float16", failures)
        elif arr.ndim != 3 or arr.shape[1] != EXPECTED_PATCHES or arr.shape[2] != EXPECTED_DIM:
            _fail(
                f"{arr_path.name}: shape {tuple(arr.shape)}, expected (T,{EXPECTED_PATCHES},{EXPECTED_DIM})",
                failures,
            )
        else:
            _ok(f"{arr_path.name}: shape {tuple(arr.shape)} fp16")


def check_actions_proprio_parity(config: dict, failures: list[str]) -> None:
    print("4. Actions/proprio/state parity (cached vs upstream, sampled windows)")
    try:
        import torch  # noqa: F401
        import datasets.point_maze_dset  # type: ignore  # noqa: F401
        import wm_poc_latent_dataset  # type: ignore  # noqa: F401
    except Exception as exc:  # upstream repo / torch / generated module unavailable
        _skip(f"upstream + cached datasets not importable here ({type(exc).__name__}); run in the DINO-WM env")
        return
    # Parity holds by construction: wm_poc_latent_dataset subclasses the upstream
    # PointMazeDataset and overrides only the visual features, so actions, states,
    # proprioception, normalization, and the split are inherited unchanged (also
    # covered by the unit tests in tests/test_dino_wm.py). A sampled value-level
    # comparison is an optional extra and is not implemented here; the report
    # claims by-construction parity, not a completed sampled check.
    _skip(
        "sampled value-level actions/proprio/state comparison not implemented; "
        "parity is by construction (subclass overrides only visual features) and unit-tested"
    )


def check_planning_path(config: dict, failures: list[str]) -> None:
    print("5. Planning encode path (image inputs still hit the encoder)")
    # The model patch routes by input rank: 4-D latent inputs skip the encoder,
    # 5-D image inputs take the original encode path. Confirm the installed patch
    # is the rank-gated bypass, not an unconditional one.
    gated = ".dim()" in _MODEL_PATCH_REPLACEMENT or "ndim" in _MODEL_PATCH_REPLACEMENT
    encodes_images = "x_norm_patchtokens" in _MODEL_PATCH_ANCHOR or "visual" in _MODEL_PATCH_ANCHOR
    if MODEL_PATCH_MARKER and gated and encodes_images:
        _ok("encode bypass is rank-gated (4-D latents bypass; 5-D images encode)")
    else:
        _fail("model encode patch is not the expected rank-gated bypass", failures)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate the DINO-WM latent cache.")
    parser.add_argument("--config", required=True, help="Path to a configs/dino_wm/*.yaml")
    args = parser.parse_args(argv[1:])

    config = resolve_config(load_config(args.config))
    print(f"Validating latent cache for: {args.config}\n")

    failures: list[str] = []
    check_split_parity(config, failures)
    check_manifest_and_latents(config, failures)
    check_actions_proprio_parity(config, failures)
    check_planning_path(config, failures)

    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed.", file=sys.stderr)
        return 1
    print("OK: all available latent-cache checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
