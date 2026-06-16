"""Latent transition datasets for the local/global track.

Reads the DINO-WM latent cache produced by ``scripts/dino_wm/precompute_latents.py``
(``episode_XXX.npy`` of shape ``(T, patches, embed_dim)`` float16 plus
``wm_poc_latent_manifest.json``) together with the upstream action tensors
(``actions.pth`` / ``seq_lengths.pth`` / ``states.pth``). A synthetic point-mass
task writes the same layout with ``.npy`` files so the full pipeline runs on CPU
without torch or real data.

Frameskip handling mirrors upstream DINO-WM: model-step ``k`` uses the latent
frame at ``t0 + k * frameskip`` and the concatenation of the ``frameskip`` raw
actions covering ``[t0 + k * frameskip, t0 + (k + 1) * frameskip)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LATENT_MANIFEST_NAME = "wm_poc_latent_manifest.json"
SYNTHETIC_DYNAMICS_NAME = "synthetic_dynamics.json"
TRANSITION_MANIFEST_NAME = "manifest.json"
DATASET_STATS_NAME = "dataset_stats.json"


def episode_file_name(index: int) -> str:
    return f"episode_{index:03d}.npy"


@dataclass(frozen=True)
class LatentDatasetSpec:
    """Metadata for a latent transition dataset."""

    name: str
    path: Path
    source_model: str


def _load_action_array(action_dir: Path) -> np.ndarray:
    """Load the per-episode action tensor as a numpy array of shape (N, T, A)."""
    npy = action_dir / "actions.npy"
    if npy.is_file():
        return np.asarray(np.load(npy), dtype=np.float32)
    pth = action_dir / "actions.pth"
    if pth.is_file():
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only without torch
            raise RuntimeError(f"PyTorch is required to read {pth}.") from exc
        return torch.load(pth, map_location="cpu").numpy().astype(np.float32)
    raise FileNotFoundError(f"No actions.npy or actions.pth under {action_dir}.")


def _load_optional_array(action_dir: Path, stem: str) -> np.ndarray | None:
    npy = action_dir / f"{stem}.npy"
    if npy.is_file():
        return np.asarray(np.load(npy))
    pth = action_dir / f"{stem}.pth"
    if pth.is_file():
        try:
            import torch
        except ImportError:  # pragma: no cover - exercised only without torch
            return None
        return torch.load(pth, map_location="cpu").numpy()
    return None


class LatentTrajectoryStore:
    """Episode-level access to cached latents and aligned raw actions."""

    def __init__(
        self,
        cache_dir: str | Path,
        action_dir: str | Path,
        *,
        max_episodes: int = 0,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.action_dir = Path(action_dir).expanduser()
        manifest_path = self.cache_dir / LATENT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Latent cache manifest not found: {manifest_path}. "
                "Run scripts/dino_wm/precompute_latents.py (or the synthetic generator) first."
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest: dict[str, Any] = json.load(f)
        lengths = list(self.manifest.get("episode_lengths", []))
        available = int(self.manifest.get("num_episodes", len(lengths)))
        self.total_episodes = min(available, len(lengths))
        count = self.total_episodes
        if max_episodes:
            count = min(count, int(max_episodes))
        if count < 1:
            raise ValueError(f"Latent cache at {self.cache_dir} reports no episodes.")
        self._lengths = [int(t) for t in lengths[:count]]
        self._actions = _load_action_array(self.action_dir)
        if self._actions.ndim != 3:
            raise ValueError(
                f"Expected actions of shape (episodes, steps, action_dim); got {self._actions.shape}."
            )
        if self._actions.shape[0] < count:
            raise ValueError(
                f"Action tensor covers {self._actions.shape[0]} episodes but the latent "
                f"cache needs {count}."
            )
        seq_lengths = _load_optional_array(self.action_dir, "seq_lengths")
        self._seq_lengths = None if seq_lengths is None else np.asarray(seq_lengths).astype(int)
        self._states = _load_optional_array(self.action_dir, "states")
        self.patches = int(self.manifest["num_patches"])
        self.embed_dim = int(self.manifest["emb_dim"])
        self.action_dim = int(self._actions.shape[-1])

    @property
    def num_episodes(self) -> int:
        return len(self._lengths)

    def episode_length(self, index: int) -> int:
        return self._lengths[index]

    def action_length(self, index: int) -> int:
        max_len = int(self._actions.shape[1])
        if self._seq_lengths is not None and index < len(self._seq_lengths):
            return min(max_len, int(self._seq_lengths[index]))
        return max_len

    def latents(self, index: int) -> np.ndarray:
        path = self.cache_dir / episode_file_name(index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing latent episode file: {path}")
        return np.load(path, mmap_mode="r")

    def actions(self, index: int) -> np.ndarray:
        return self._actions[index, : self.action_length(index)]

    def states(self, index: int) -> np.ndarray | None:
        if self._states is None:
            return None
        return self._states[index]


def split_episodes(
    num_episodes: int, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Deterministic episode-level train/val split (no window-level leakage)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    order = np.random.default_rng(int(seed)).permutation(num_episodes)
    num_val = max(1, int(round(num_episodes * val_fraction)))
    num_val = min(num_val, num_episodes - 1) if num_episodes > 1 else 1
    val = sorted(int(i) for i in order[:num_val])
    train = sorted(int(i) for i in order[num_val:])
    return train, val


def split_store_episodes(
    store: "LatentTrajectoryStore", val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split over the FULL cached episode set, then apply the store's cap.

    Deriving the permutation from the uncapped count keeps train/val membership
    stable when ``training.max_episodes`` differs between training, export, and
    planning evaluation - otherwise a capped run would silently leak training
    episodes into another run's validation set.
    """
    train, val = split_episodes(store.total_episodes, val_fraction, seed)
    count = store.num_episodes
    train = [e for e in train if e < count]
    val = [e for e in val if e < count]
    if not train or not val:
        raise ValueError(
            f"Episode cap max_episodes={count} leaves an empty split "
            f"(train={len(train)}, val={len(val)} of {store.total_episodes} total); "
            "raise training.max_episodes."
        )
    return train, val


def fold_actions(actions: np.ndarray, frameskip: int) -> np.ndarray:
    """Fold ``(n * frameskip, A)`` raw actions into ``(n, frameskip * A)`` blocks."""
    if actions.shape[0] % frameskip != 0:
        raise ValueError(
            f"Cannot fold {actions.shape[0]} raw actions with frameskip {frameskip}."
        )
    n = actions.shape[0] // frameskip
    return actions.reshape(n, frameskip * actions.shape[1])


def window_frame_count(context_len: int, rollout_steps: int) -> int:
    return context_len + rollout_steps


def max_window_start(
    latent_len: int, action_len: int, context_len: int, rollout_steps: int, frameskip: int
) -> int:
    """Largest valid window start ``t0`` (inclusive); negative when none fits."""
    span = (window_frame_count(context_len, rollout_steps) - 1) * frameskip
    return min(latent_len - 1 - span, action_len - span)


def build_window_index(
    store: LatentTrajectoryStore,
    episodes: list[int],
    *,
    context_len: int,
    rollout_steps: int,
    frameskip: int,
    stride: int = 1,
    max_windows: int = 0,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for ep in episodes:
        last = max_window_start(
            store.episode_length(ep),
            store.action_length(ep),
            context_len,
            rollout_steps,
            frameskip,
        )
        for t0 in range(0, last + 1, max(1, stride)):
            windows.append((ep, t0))
            if max_windows and len(windows) >= max_windows:
                return windows
    return windows


class LatentWindowDataset:
    """Context windows plus K-step rollout targets from a latent store.

    Implements the torch ``Dataset`` protocol (``__len__``/``__getitem__``)
    without inheriting from it so the module stays importable without torch.
    """

    def __init__(
        self,
        store: LatentTrajectoryStore,
        episodes: list[int],
        *,
        context_len: int,
        rollout_steps: int,
        frameskip: int,
        stride: int = 1,
        max_windows: int = 0,
    ) -> None:
        if context_len < 1 or rollout_steps < 1 or frameskip < 1:
            raise ValueError("context_len, rollout_steps, and frameskip must be >= 1.")
        self.store = store
        self.context_len = int(context_len)
        self.rollout_steps = int(rollout_steps)
        self.frameskip = int(frameskip)
        self.windows = build_window_index(
            store,
            episodes,
            context_len=context_len,
            rollout_steps=rollout_steps,
            frameskip=frameskip,
            stride=stride,
            max_windows=max_windows,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        ep, t0 = self.windows[index]
        c, k, fs = self.context_len, self.rollout_steps, self.frameskip
        frames = c + k
        latents = self.store.latents(ep)
        frame_idx = [t0 + j * fs for j in range(frames)]
        z = np.asarray(latents[frame_idx], dtype=np.float32)
        raw = np.asarray(
            self.store.actions(ep)[t0 : t0 + (frames - 1) * fs], dtype=np.float32
        )
        blocks = fold_actions(raw, fs)  # (frames - 1, fs * A)
        return {
            "z_context": z[:c],
            "z_targets": z[c:],
            "actions_context": blocks[: c - 1],
            "actions": blocks[c - 1 :],
            "episode_id": episode_file_name(ep),
            "start_t": int(t0),
        }


class LatentTransitionDataset(LatentWindowDataset):
    """One-step transitions: context of one frame, one rollout step."""

    def __init__(
        self,
        store: LatentTrajectoryStore,
        episodes: list[int],
        *,
        frameskip: int,
        stride: int = 1,
        max_windows: int = 0,
    ) -> None:
        super().__init__(
            store,
            episodes,
            context_len=1,
            rollout_steps=1,
            frameskip=frameskip,
            stride=stride,
            max_windows=max_windows,
        )


def collate_latent_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack window samples into torch tensors (requires torch)."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise RuntimeError("PyTorch is required to collate latent windows.") from exc
    out: dict[str, Any] = {}
    for key in ("z_context", "z_targets", "actions_context", "actions"):
        out[key] = torch.from_numpy(np.stack([sample[key] for sample in batch]))
    out["episode_id"] = [sample["episode_id"] for sample in batch]
    out["start_t"] = torch.tensor([sample["start_t"] for sample in batch], dtype=torch.long)
    return out


def export_transition_manifest(
    store: LatentTrajectoryStore,
    out_dir: str | Path,
    *,
    context_len: int,
    rollout_steps: int,
    frameskip: int,
    val_fraction: float,
    split_seed: int,
    max_windows: int = 0,
) -> dict[str, Any]:
    """Write manifest.json + dataset_stats.json describing the transition dataset."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    train_eps, val_eps = split_store_episodes(store, val_fraction, split_seed)
    train_windows = build_window_index(
        store,
        train_eps,
        context_len=context_len,
        rollout_steps=rollout_steps,
        frameskip=frameskip,
        max_windows=max_windows,
    )
    val_windows = build_window_index(
        store,
        val_eps,
        context_len=context_len,
        rollout_steps=rollout_steps,
        frameskip=frameskip,
        max_windows=max_windows,
    )
    manifest = {
        "format": "wm_poc_local_global_transitions_v1",
        "cache_dir": str(store.cache_dir),
        "action_dir": str(store.action_dir),
        "num_episodes": store.num_episodes,
        "total_episodes": store.total_episodes,
        "patches": store.patches,
        "embed_dim": store.embed_dim,
        "action_dim": store.action_dim,
        "context_len": context_len,
        "rollout_steps": rollout_steps,
        "frameskip": frameskip,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "train_episodes": train_eps,
        "val_episodes": val_eps,
        "num_train_windows": len(train_windows),
        "num_val_windows": len(val_windows),
    }
    sample_ep = train_eps[0] if train_eps else 0
    sample = np.asarray(store.latents(sample_ep)[:1], dtype=np.float32)
    stats = {
        "latent_mean_abs": float(np.mean(np.abs(sample))),
        "latent_dtype_on_disk": str(store.latents(sample_ep).dtype),
        "episode_length_min": int(min(store.episode_length(i) for i in range(store.num_episodes))),
        "episode_length_max": int(max(store.episode_length(i) for i in range(store.num_episodes))),
        "action_dim_raw": store.action_dim,
        "action_dim_folded": store.action_dim * frameskip,
    }
    with (out / TRANSITION_MANIFEST_NAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with (out / DATASET_STATS_NAME).open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return manifest


def compute_action_state_stats(store: "LatentTrajectoryStore") -> dict[str, np.ndarray]:
    """Per-dimension mean/std of raw actions and states, masked by seq_lengths.

    Mirrors upstream ``PointMazeDataset.get_data_mean_std``: checkpoints trained
    with ``normalize_action: true`` consumed ``(x - mean) / std`` actions and
    proprio, so the global-model adapter must apply the same statistics.
    """
    actions = []
    states = []
    for ep in range(store.num_episodes):
        actions.append(store.actions(ep))
        state = store.states(ep)
        if state is not None:
            states.append(np.asarray(state[: store.action_length(ep)], dtype=np.float32))
    all_actions = np.vstack(actions)
    stats = {
        "action_mean": all_actions.mean(axis=0),
        "action_std": all_actions.std(axis=0, ddof=1) + 1e-12,
    }
    if states:
        all_states = np.vstack(states)
        stats["state_mean"] = all_states.mean(axis=0)
        stats["state_std"] = all_states.std(axis=0, ddof=1) + 1e-12
    return stats


def ensure_synthetic_task_data(config: dict[str, Any]) -> bool:
    """Generate the synthetic latent cache for a config when needed.

    Returns True when data was generated. Applies to ``task: synthetic`` configs
    or any config with ``smoke.use_synthetic_latents_if_missing: true``.
    """
    from wm_poc.dino_wm.configs import get_config_value
    from wm_poc.local_global.configs import action_data_dir, latent_cache_dir

    cache_dir = latent_cache_dir(config)
    if (cache_dir / LATENT_MANIFEST_NAME).is_file():
        return False
    if config.get("task") != "synthetic" and not get_config_value(
        config, "smoke.use_synthetic_latents_if_missing", False
    ):
        return False
    generate_synthetic_task(
        cache_dir,
        action_data_dir(config),
        episodes=int(get_config_value(config, "smoke.synthetic_episodes", 8)),
        episode_length=int(get_config_value(config, "smoke.synthetic_episode_length", 64)),
        patches=int(get_config_value(config, "global_model.latent_patches", 16)),
        embed_dim=int(get_config_value(config, "global_model.latent_dim", 32)),
        action_dim=int(get_config_value(config, "planning.action_dim", 2)),
        seed=int(config.get("seed", 0)),
    )
    return True


# ---------------------------------------------------------------------------
# Synthetic point-mass task (CPU-only smoke path)
# ---------------------------------------------------------------------------

SYNTHETIC_DT = 0.1
SYNTHETIC_DAMPING = 0.9
SYNTHETIC_STATE_DIM = 4


def synthetic_step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Ground-truth point-mass dynamics: state = [px, py, vx, vy], action in R^2."""
    pos = state[..., :2]
    vel = state[..., 2:]
    new_vel = SYNTHETIC_DAMPING * vel + SYNTHETIC_DT * action
    new_pos = np.clip(pos + SYNTHETIC_DT * new_vel, -1.0, 1.0)
    return np.concatenate([new_pos, new_vel], axis=-1)


def generate_synthetic_task(
    cache_dir: str | Path,
    action_dir: str | Path,
    *,
    episodes: int = 8,
    episode_length: int = 64,
    patches: int = 16,
    embed_dim: int = 32,
    action_dim: int = 2,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Write a synthetic latent cache + action dir mirroring the DINO-WM layout.

    Returns ``(cache_dir, action_dir)``. Latents are an exact linear encoding of
    the point-mass state, so a perfect global model exists in closed form.
    Refuses to write over directories that already hold (real) data.
    """
    cache_dir = Path(cache_dir).expanduser()
    action_dir = Path(action_dir).expanduser()
    if cache_dir.is_dir() and any(cache_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to generate synthetic latents into non-empty {cache_dir} "
            "(it may hold a partially built real cache). If it is a stale synthetic "
            f"cache from an interrupted run, delete it first: rm -rf {cache_dir}"
        )
    if (action_dir / "actions.pth").is_file():
        raise FileExistsError(
            f"Refusing to generate synthetic actions next to real data in {action_dir}."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    weight = rng.normal(0.0, 1.0, size=(patches * embed_dim, SYNTHETIC_STATE_DIM)).astype(
        np.float32
    ) / np.sqrt(SYNTHETIC_STATE_DIM)
    bias = rng.normal(0.0, 0.05, size=(patches * embed_dim,)).astype(np.float32)

    states = np.zeros((episodes, episode_length, SYNTHETIC_STATE_DIM), dtype=np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(episodes, episode_length, action_dim)).astype(
        np.float32
    )
    for ep in range(episodes):
        state = np.concatenate(
            [rng.uniform(-0.8, 0.8, size=2), rng.uniform(-0.2, 0.2, size=2)]
        ).astype(np.float32)
        for t in range(episode_length):
            states[ep, t] = state
            state = synthetic_step(state, actions[ep, t]).astype(np.float32)
        flat = states[ep] @ weight.T + bias  # (T, patches * embed_dim)
        latents = flat.reshape(episode_length, patches, embed_dim).astype(np.float16)
        np.save(cache_dir / episode_file_name(ep), latents)

    manifest = {
        "format": "wm_poc_dino_latents_v1",
        "encoder_name": "synthetic_linear",
        "feature_key": "synthetic",
        "img_size": 0,
        "encoder_image_size": 0,
        "num_patches": patches,
        "emb_dim": embed_dim,
        "dtype": "float16",
        "num_episodes": episodes,
        "dataset_episodes": episodes,
        "episode_lengths": [episode_length] * episodes,
        "synthetic": True,
    }
    with (cache_dir / LATENT_MANIFEST_NAME).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    np.save(action_dir / "actions.npy", actions)
    np.save(action_dir / "states.npy", states)
    np.save(action_dir / "seq_lengths.npy", np.full(episodes, episode_length, dtype=np.int64))
    np.save(action_dir / "encoder_weight.npy", weight)
    np.save(action_dir / "encoder_bias.npy", bias)
    dynamics = {
        "type": "point_mass",
        "dt": SYNTHETIC_DT,
        "damping": SYNTHETIC_DAMPING,
        "state_dim": SYNTHETIC_STATE_DIM,
        "action_dim": action_dim,
        "patches": patches,
        "embed_dim": embed_dim,
        "seed": int(seed),
    }
    with (action_dir / SYNTHETIC_DYNAMICS_NAME).open("w", encoding="utf-8") as f:
        json.dump(dynamics, f, indent=2)
    return cache_dir, action_dir
