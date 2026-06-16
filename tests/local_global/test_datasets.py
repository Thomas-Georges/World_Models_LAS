from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from wm_poc.local_global.datasets import (  # noqa: E402
    LATENT_MANIFEST_NAME,
    LatentTrajectoryStore,
    LatentTransitionDataset,
    LatentWindowDataset,
    build_window_index,
    export_transition_manifest,
    fold_actions,
    generate_synthetic_task,
    max_window_start,
    split_episodes,
)

EPISODES = 6
LENGTH = 40
PATCHES = 9
DIM = 8


@pytest.fixture()
def synthetic_dirs(tmp_path):
    return generate_synthetic_task(
        tmp_path / "cache",
        tmp_path / "actions",
        episodes=EPISODES,
        episode_length=LENGTH,
        patches=PATCHES,
        embed_dim=DIM,
        seed=3,
    )


def test_generate_synthetic_layout(synthetic_dirs):
    cache_dir, action_dir = synthetic_dirs
    manifest = json.loads((cache_dir / LATENT_MANIFEST_NAME).read_text())
    assert manifest["num_episodes"] == EPISODES
    assert manifest["episode_lengths"] == [LENGTH] * EPISODES
    episode = np.load(cache_dir / "episode_000.npy")
    assert episode.shape == (LENGTH, PATCHES, DIM)
    assert episode.dtype == np.float16
    actions = np.load(action_dir / "actions.npy")
    assert actions.shape == (EPISODES, LENGTH, 2)


def test_generate_refuses_nonempty_cache(synthetic_dirs, tmp_path):
    cache_dir, _ = synthetic_dirs
    with pytest.raises(FileExistsError):
        generate_synthetic_task(cache_dir, tmp_path / "other_actions")


def test_store_reads_episodes(synthetic_dirs):
    cache_dir, action_dir = synthetic_dirs
    store = LatentTrajectoryStore(cache_dir, action_dir)
    assert store.num_episodes == EPISODES
    assert store.patches == PATCHES
    assert store.embed_dim == DIM
    assert store.action_dim == 2
    assert store.latents(0).shape == (LENGTH, PATCHES, DIM)
    assert store.actions(0).shape == (LENGTH, 2)
    assert store.states(0).shape == (LENGTH, 4)


def test_store_max_episodes_cap(synthetic_dirs):
    cache_dir, action_dir = synthetic_dirs
    store = LatentTrajectoryStore(cache_dir, action_dir, max_episodes=2)
    assert store.num_episodes == 2


def test_split_is_deterministic_and_disjoint():
    train_a, val_a = split_episodes(20, 0.2, seed=42)
    train_b, val_b = split_episodes(20, 0.2, seed=42)
    assert train_a == train_b and val_a == val_b
    assert set(train_a).isdisjoint(val_a)
    assert sorted(train_a + val_a) == list(range(20))
    train_c, _ = split_episodes(20, 0.2, seed=7)
    assert train_c != train_a


def test_fold_actions():
    raw = np.arange(12, dtype=np.float32).reshape(6, 2)
    folded = fold_actions(raw, 3)
    assert folded.shape == (2, 6)
    assert folded[0].tolist() == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        fold_actions(raw[:5], 3)


def test_max_window_start_math():
    # context 2 + rollout 3 -> 5 frames spanning (5-1)*fs raw steps.
    assert max_window_start(40, 40, context_len=2, rollout_steps=3, frameskip=2) == 31
    assert max_window_start(10, 10, context_len=2, rollout_steps=3, frameskip=4) < 0


def test_window_dataset_shapes(synthetic_dirs):
    cache_dir, action_dir = synthetic_dirs
    store = LatentTrajectoryStore(cache_dir, action_dir)
    dataset = LatentWindowDataset(
        store, [0, 1], context_len=2, rollout_steps=3, frameskip=2
    )
    sample = dataset[0]
    assert sample["z_context"].shape == (2, PATCHES, DIM)
    assert sample["z_targets"].shape == (3, PATCHES, DIM)
    assert sample["actions_context"].shape == (1, 4)
    assert sample["actions"].shape == (3, 4)
    assert sample["z_context"].dtype == np.float32
    assert sample["episode_id"] == "episode_000.npy"


def test_window_targets_align_with_source(synthetic_dirs):
    cache_dir, action_dir = synthetic_dirs
    store = LatentTrajectoryStore(cache_dir, action_dir)
    dataset = LatentWindowDataset(
        store, [1], context_len=2, rollout_steps=2, frameskip=3
    )
    sample = dataset[5]
    t0 = sample["start_t"]
    raw = np.asarray(store.latents(1), dtype=np.float32)
    np.testing.assert_allclose(sample["z_context"][0], raw[t0])
    np.testing.assert_allclose(sample["z_context"][1], raw[t0 + 3])
    np.testing.assert_allclose(sample["z_targets"][0], raw[t0 + 6])
    np.testing.assert_allclose(sample["z_targets"][1], raw[t0 + 9])
    raw_actions = store.actions(1)
    np.testing.assert_allclose(
        sample["actions"][0], raw_actions[t0 + 3 : t0 + 6].reshape(-1)
    )


def test_transition_dataset_is_one_step(synthetic_dirs):
    cache_dir, action_dir = synthetic_dirs
    store = LatentTrajectoryStore(cache_dir, action_dir)
    dataset = LatentTransitionDataset(store, [0], frameskip=1)
    sample = dataset[0]
    assert sample["z_context"].shape == (1, PATCHES, DIM)
    assert sample["z_targets"].shape == (1, PATCHES, DIM)
    assert sample["actions_context"].shape == (0, 2)


def test_max_windows_cap(synthetic_dirs):
    cache_dir, action_dir = synthetic_dirs
    store = LatentTrajectoryStore(cache_dir, action_dir)
    windows = build_window_index(
        store, [0, 1, 2], context_len=1, rollout_steps=1, frameskip=1, max_windows=7
    )
    assert len(windows) == 7


def test_export_transition_manifest(synthetic_dirs, tmp_path):
    cache_dir, action_dir = synthetic_dirs
    store = LatentTrajectoryStore(cache_dir, action_dir)
    out = tmp_path / "transition_data"
    manifest = export_transition_manifest(
        store,
        out,
        context_len=2,
        rollout_steps=3,
        frameskip=2,
        val_fraction=0.25,
        split_seed=42,
    )
    assert (out / "manifest.json").is_file()
    assert (out / "dataset_stats.json").is_file()
    assert set(manifest["train_episodes"]).isdisjoint(manifest["val_episodes"])
    assert manifest["num_train_windows"] > 0
    assert manifest["num_val_windows"] > 0


def test_collate_latent_windows(synthetic_dirs):
    torch = pytest.importorskip("torch")
    from wm_poc.local_global.datasets import collate_latent_windows

    cache_dir, action_dir = synthetic_dirs
    store = LatentTrajectoryStore(cache_dir, action_dir)
    dataset = LatentWindowDataset(store, [0], context_len=2, rollout_steps=3, frameskip=2)
    batch = collate_latent_windows([dataset[0], dataset[1]])
    assert batch["z_context"].shape == (2, 2, PATCHES, DIM)
    assert batch["actions"].shape == (2, 3, 4)
    assert batch["z_context"].dtype == torch.float32
    assert batch["start_t"].tolist() == [0, 1]
