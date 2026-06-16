"""Latent-cache support for upstream DINO-WM training.

The frozen DINO encoder makes online feature extraction pure recomputation:
upstream re-reads every full episode image file and re-encodes its frames for
each 4-frame training slice. This module installs two files into the upstream
checkout and applies one small textual patch so that:

1. ``wm_poc_precompute_latents.py`` encodes each episode exactly once and
   stores fp16 patch latents as per-episode ``.npy`` files plus a manifest.
2. ``wm_poc_latent_dataset.py`` provides a drop-in Hydra dataset target whose
   slicer memmap-reads only the frames each sample needs.
3. ``models/visual_world_model.py`` gains an input-dispatch bypass in
   ``encode_obs`` so already-encoded latents skip the DINO forward pass.

The real encoder stays in the model and in checkpoints, so planning and any
image-based evaluation keep working unchanged.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


MODEL_PATCH_MARKER = "WM_POC_DINO_LATENT_BYPASS_PATCH"
LATENT_DATASET_MODULE_NAME = "wm_poc_latent_dataset.py"
PRECOMPUTE_SCRIPT_NAME = "wm_poc_precompute_latents.py"
LATENT_MANIFEST_NAME = "wm_poc_latent_manifest.json"
LATENT_MANIFEST_FORMAT = "wm_poc_dino_latents_v1"


LATENT_DATASET_MODULE_SOURCE = '''"""Latent-cache dataset for DINO-WM (installed by wm-prediction).

Marker: WM_POC_DINO_LATENT_DATASET_V1

Serves precomputed frozen DINO patch latents instead of raw image frames so
training skips both the per-sample full-episode reload and the online encoder
forward pass. Build the cache first with wm_poc_precompute_latents.py.
"""
import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from datasets.point_maze_dset import PointMazeDataset
from datasets.traj_dset import TrajDataset, TrajSubset, split_traj_datasets

MANIFEST_NAME = "wm_poc_latent_manifest.json"
MANIFEST_FORMAT = "wm_poc_dino_latents_v1"


def episode_file_name(idx):
    return f"episode_{idx:03d}.npy"


def load_manifest(latent_dir):
    path = Path(latent_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing latent cache manifest: {path}. Run "
            "scripts/dino_wm/precompute_latents.py --config <config> --no-dry-run first."
        )
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"Unexpected latent manifest format in {path}: {manifest.get('format')!r}"
        )
    return manifest


class LatentPointMazeDataset(PointMazeDataset):
    """PointMazeDataset whose obs['visual'] is precomputed DINO patch latents.

    Actions, states, proprioception, normalization statistics, and the
    train/val split are inherited unchanged from the upstream dataset, so a
    latent run is directly comparable to an online-encoding run.
    """

    def __init__(
        self,
        latent_dir,
        data_path="data/point_maze",
        n_rollout=None,
        transform=None,
        normalize_action=False,
        action_scale=1.0,
    ):
        # The image transform is never applied when serving cached latents,
        # but it must stay populated: plan.py builds its observation
        # Preprocessor from dset.transform to process raw env renders.
        super().__init__(
            data_path=data_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            action_scale=action_scale,
        )
        self.latent_dir = Path(latent_dir)
        if str(self.latent_dir).startswith("/content/drive"):
            print(
                "WARNING: reading DINO latents from Google Drive; per-sample random "
                "reads through the FUSE mount will dominate step time. Stage the "
                "cache on local disk (e.g. /content/wm_poc_latent_cache) instead."
            )
        self.manifest = load_manifest(self.latent_dir)
        n = len(self.seq_lengths)
        covered = int(self.manifest.get("num_episodes", 0))
        if covered < n:
            raise ValueError(
                f"Latent cache at {self.latent_dir} covers {covered} episodes but "
                f"{n} are required; re-run precompute with --n-rollout {n} or higher."
            )
        for idx in (0, n - 1):
            path = self.latent_dir / episode_file_name(idx)
            if not path.is_file():
                raise FileNotFoundError(f"Latent cache is missing {path}.")
            arr = np.load(path, mmap_mode="r")
            if int(arr.shape[0]) != int(self.seq_lengths[idx]):
                raise ValueError(
                    f"Latent cache length mismatch for episode {idx}: cache has "
                    f"{arr.shape[0]} frames, dataset expects {int(self.seq_lengths[idx])}. "
                    "Re-run precompute with --force."
                )
        self.num_patches = int(self.manifest["num_patches"])
        self.latent_emb_dim = int(self.manifest["emb_dim"])

    def get_latents(self, idx, frames):
        arr = np.load(self.latent_dir / episode_file_name(idx), mmap_mode="r")
        frames = np.asarray(list(frames), dtype=np.int64)
        return torch.from_numpy(np.ascontiguousarray(arr[frames]))

    def get_frames(self, idx, frames):
        frames = list(frames)
        obs = {
            "visual": self.get_latents(idx, frames),
            "proprio": self.proprios[idx, frames],
        }
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        return obs, act, state, {}


class LatentTrajSlicerDataset(TrajDataset):
    """Mirror of datasets.traj_dset.TrajSlicerDataset that reads only the
    sliced frames from the latent cache instead of materializing whole
    episodes per sample. ``stride`` > 1 subsamples slice start positions to
    shrink an epoch; stride 1 reproduces the upstream window set exactly.

    TrajSubset wrappers are resolved to base-dataset indices at construction
    time: TrajSubset.__getattr__ recurses during unpickling, so it must never
    be shipped to DataLoader workers under a spawn start method.
    """

    def __init__(self, dataset, num_frames, frameskip=1, stride=1):
        self.num_frames = num_frames
        self.frameskip = frameskip
        self.stride = max(1, int(stride))
        base, base_indices = self._resolve_dataset(dataset)
        self._base = base
        self.slices = []
        for i in range(len(dataset)):
            T = dataset.get_seq_length(i)
            if T - num_frames < 0:
                print(f"Ignored short sequence #{i}: len={T}, num_frames={num_frames}")
            else:
                self.slices += [
                    (base_indices[i], start, start + num_frames * self.frameskip)
                    for start in range(0, T - num_frames * frameskip + 1, self.stride)
                ]  # slice indices follow convention [start, end)
        self.slices = np.random.permutation(self.slices)

        self.proprio_dim = base.proprio_dim
        self.action_dim = base.action_dim * self.frameskip
        self.state_dim = base.state_dim

    @staticmethod
    def _resolve_dataset(dataset):
        indices = list(range(len(dataset)))
        while isinstance(dataset, TrajSubset):
            indices = [int(dataset.indices[i]) for i in indices]
            dataset = dataset.dataset
        return dataset, indices

    def get_seq_length(self, idx):
        return self.num_frames

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        traj_idx, start, end = (int(v) for v in self.slices[idx])
        base = self._base
        frames = list(range(start, end, self.frameskip))
        obs = {
            "visual": base.get_latents(traj_idx, frames),
            "proprio": base.proprios[traj_idx, frames],
        }
        state = base.states[traj_idx, frames]
        act = base.actions[traj_idx, start:end]
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)
        return tuple([obs, act, state])


def load_point_maze_latent_slice_train_val(
    transform=None,
    n_rollout=50,
    data_path="data/point_maze",
    normalize_action=False,
    split_ratio=0.8,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    latent_cache_dir=None,
    slice_stride=1,
    **unused_kwargs,
):
    if not latent_cache_dir:
        raise ValueError("latent_cache_dir is required for the latent-cache dataset.")
    dset = LatentPointMazeDataset(
        latent_dir=latent_cache_dir,
        data_path=data_path,
        n_rollout=n_rollout,
        transform=transform,
        normalize_action=normalize_action,
    )
    # Same split helper and seed as upstream get_train_val_sliced.
    dset_train, dset_val = split_traj_datasets(
        dset, train_fraction=split_ratio, random_seed=42
    )
    num_frames = num_hist + num_pred
    train_slices = LatentTrajSlicerDataset(
        dset_train, num_frames, frameskip, stride=slice_stride
    )
    val_slices = LatentTrajSlicerDataset(
        dset_val, num_frames, frameskip, stride=slice_stride
    )
    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
'''


PRECOMPUTE_SCRIPT_SOURCE = '''"""Precompute frozen DINO patch latents for DINO-WM (installed by wm-prediction).

Marker: WM_POC_DINO_LATENT_PRECOMPUTE_V1

Encodes each episode exactly once with the frozen DINO encoder and writes
fp16 patch latents to per-episode .npy files plus a manifest. Idempotent:
episodes whose cache file already exists with the right length are skipped,
so an interrupted run resumes where it stopped.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torchvision import transforms

from datasets.img_transforms import default_transform
from models.dino import DinoV2Encoder

MANIFEST_NAME = "wm_poc_latent_manifest.json"
MANIFEST_FORMAT = "wm_poc_dino_latents_v1"
DECODER_SCALE = 16  # matches the vqvae-derived patch grid in VWorldModel


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute frozen DINO patch latents.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--n-rollout", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--encoder-name", default="dinov2_vits14")
    parser.add_argument("--feature-key", default="x_norm_patchtokens")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp or os.environ.get(
        "DINO_MIXED_PRECISION", "bf16" if device.startswith("cuda") else "no"
    )
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)

    cache_dir = args.cache_dir.expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data_path.expanduser()

    seq_lengths = torch.load(data_path / "seq_lengths.pth")
    n = len(seq_lengths) if args.n_rollout is None else min(args.n_rollout, len(seq_lengths))

    encoder = DinoV2Encoder(name=args.encoder_name, feature_key=args.feature_key)
    encoder.eval()
    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)

    # Mirror VWorldModel: DINO consumes (img_size // 16) * patch_size pixels.
    num_side_patches = args.img_size // DECODER_SCALE
    encoder_image_size = num_side_patches * encoder.patch_size
    base_transform = default_transform(args.img_size)
    encoder_resize = transforms.Resize(encoder_image_size)

    num_patches = None
    emb_dim = None
    encoded = 0
    skipped = 0
    started = time.time()
    print(f"Latent cache dir: {cache_dir}", flush=True)
    if str(cache_dir).startswith("/content/drive"):
        print(
            "WARNING: latent cache is on Google Drive. Training reads random latent "
            "slices and will be I/O-bound through the Drive FUSE mount; use local "
            "disk (e.g. /content/wm_poc_latent_cache) unless you must persist latents.",
            flush=True,
        )
    print(
        f"Precomputing DINO latents for {n} episodes "
        f"(encoder={args.encoder_name}, img_size={args.img_size}, "
        f"encoder_input={encoder_image_size}px, device={device}, amp={amp})",
        flush=True,
    )
    for idx in range(n):
        T = int(seq_lengths[idx])
        out_path = cache_dir / f"episode_{idx:03d}.npy"
        if out_path.is_file() and not args.force:
            arr = np.load(out_path, mmap_mode="r")
            if int(arr.shape[0]) == T:
                if num_patches is None:
                    num_patches, emb_dim = int(arr.shape[1]), int(arr.shape[2])
                skipped += 1
                continue

        frames = torch.load(data_path / "obses" / f"episode_{idx:03d}.pth")[:T]
        frames = rearrange(frames, "t h w c -> t c h w").float() / 255.0
        frames = encoder_resize(base_transform(frames))

        chunks = []
        with torch.no_grad():
            for chunk in frames.split(args.batch_size):
                chunk = chunk.to(device, non_blocking=True)
                if amp_dtype is not None and device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        emb = encoder(chunk)
                else:
                    emb = encoder(chunk)
                chunks.append(emb.to(torch.float16).cpu())
        latents = torch.cat(chunks, dim=0).numpy()
        num_patches, emb_dim = int(latents.shape[1]), int(latents.shape[2])

        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with tmp_path.open("wb") as f:
            np.save(f, latents)
        os.replace(tmp_path, out_path)
        encoded += 1
        if encoded % 25 == 0 or idx == n - 1:
            rate = (encoded + skipped) / max(time.time() - started, 1e-6)
            print(
                f"Encoded latents for episode {idx + 1}/{n} "
                f"({rate:.2f} episodes/s, {encoded} new, {skipped} reused)",
                flush=True,
            )

    if num_patches is None:
        raise RuntimeError("No episodes were encoded or reused; nothing to cache.")

    manifest = {
        "format": MANIFEST_FORMAT,
        "encoder_name": args.encoder_name,
        "feature_key": args.feature_key,
        "img_size": int(args.img_size),
        "encoder_image_size": int(encoder_image_size),
        "num_patches": num_patches,
        "emb_dim": emb_dim,
        "dtype": "float16",
        "num_episodes": int(n),
        # Total episodes the raw dataset provides; once num_episodes reaches
        # this, the cache is complete for any rollout request.
        "dataset_episodes": int(len(seq_lengths)),
        "episode_lengths": [int(seq_lengths[i]) for i in range(n)],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = cache_dir / MANIFEST_NAME
    tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    print(
        f"Latent precompute complete: {encoded} encoded, {skipped} reused, "
        f"manifest: {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


_MODEL_PATCH_ANCHOR = """        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)
"""

_MODEL_PATCH_REPLACEMENT = f"""        visual = obs['visual']
        if visual.ndim == 4:  # {MODEL_PATCH_MARKER}: precomputed (b, t, p, d) patch latents
            visual_embs = visual.float()
        else:
            b = visual.shape[0]
            visual = rearrange(visual, "b t ... -> (b t) ...")
            visual = self.encoder_transform(visual)
            visual_embs = self.encoder.forward(visual)
            visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)
"""


def install_latent_support(repo_path: Path) -> list[str]:
    """Write the latent dataset module and precompute script into the upstream
    checkout. Returns the file names that were created or updated."""

    repo_path = Path(repo_path).expanduser()
    if not repo_path.is_dir():
        raise FileNotFoundError(f"Upstream DINO-WM checkout does not exist: {repo_path}")
    changed: list[str] = []
    for name, source in (
        (LATENT_DATASET_MODULE_NAME, LATENT_DATASET_MODULE_SOURCE),
        (PRECOMPUTE_SCRIPT_NAME, PRECOMPUTE_SCRIPT_SOURCE),
    ):
        target = repo_path / name
        if target.is_file() and target.read_text(encoding="utf-8") == source:
            continue
        target.write_text(source, encoding="utf-8")
        changed.append(name)
    return changed


def patch_model_source(source: str) -> tuple[str, bool]:
    """Add the latent bypass to VWorldModel.encode_obs."""

    if MODEL_PATCH_MARKER in source:
        return source, False
    if _MODEL_PATCH_ANCHOR not in source:
        raise ValueError(
            "Could not apply DINO-WM latent bypass patch; encode_obs anchor not "
            "found in models/visual_world_model.py."
        )
    return source.replace(_MODEL_PATCH_ANCHOR, _MODEL_PATCH_REPLACEMENT, 1), True


def patch_model_file(model_path: Path) -> bool:
    model_path = model_path.expanduser()
    source = model_path.read_text(encoding="utf-8")
    patched, changed = patch_model_source(source)
    if not changed:
        return False

    backup_dir = model_path.parent / ".wm_poc_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"visual_world_model.py.latent_bypass.{stamp}"
    shutil.copy2(model_path, backup_path)
    model_path.write_text(patched, encoding="utf-8")
    return True
