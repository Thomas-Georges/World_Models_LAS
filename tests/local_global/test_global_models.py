from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from wm_poc.local_global.global_models import DinoWMGlobalModel  # noqa: E402

PATCHES = 4
DIM = 3
ACTION = 2


class FakeUpstreamWM:
    """Deterministic stand-in for the upstream VWorldModel.rollout."""

    def rollout(self, obs_0, act):
        base = obs_0["visual"][:, -1]  # (B, P, D)
        # The final frame depends on every action block, so any chunking bug
        # (wrong prefix expansion, dropped candidates) changes the output.
        shift = act.sum(dim=(1, 2)).view(-1, 1, 1)
        return torch.stack([base + shift, base + 2 * shift], dim=1)  # (B, 2, P, D)


def _adapter(rollout_batch_size: int, tmp_path):
    stats = {
        "action_mean": np.zeros(ACTION, dtype=np.float32),
        "action_std": np.ones(ACTION, dtype=np.float32),
    }
    return DinoWMGlobalModel(
        tmp_path,
        tmp_path,
        device="cpu",
        latent_dim=DIM,
        frameskip=1,
        num_hist=2,
        rollout_batch_size=rollout_batch_size,
        action_stats=stats,
        wm=FakeUpstreamWM(),
    )


def test_rollout_chunking_is_exact(tmp_path):
    z_context = torch.randn(2, PATCHES, DIM)
    actions_context = torch.randn(1, ACTION)
    candidates = torch.randn(10, 3, ACTION)
    small = _adapter(3, tmp_path)  # 10 candidates -> chunks of 3, 3, 3, 1
    big = _adapter(100, tmp_path)  # single chunk
    state_small = small.init_state(z_context, None, actions_context)
    state_big = big.init_state(z_context, None, actions_context)
    torch.testing.assert_close(
        small.rollout_final(state_small, candidates),
        big.rollout_final(state_big, candidates),
    )
    # Per-candidate call accounting is chunking-independent too.
    assert small.num_forward_calls == big.num_forward_calls == 10


def test_advance_returns_per_step_latents(tmp_path):
    adapter = _adapter(4, tmp_path)
    state = adapter.init_state(torch.randn(2, PATCHES, DIM), None, torch.randn(1, ACTION))
    new_state = adapter.advance(state, torch.randn(2, ACTION).reshape(2, ACTION))
    assert new_state["step_latents"].shape == (2, PATCHES, DIM)
    torch.testing.assert_close(adapter.current_latent(new_state), new_state["step_latents"][-1])
