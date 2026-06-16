from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wm_poc.local_global.models import (  # noqa: E402
    ContextLocalDynamics,
    LocalDynamics,
    PatchProjector,
    build_local_model,
    load_local_checkpoint,
    save_local_checkpoint,
)

PATCHES = 16
DIM = 12
LOCAL = 8
ACTION = 4


def _build(model_type="residual_mlp", projection="mean_pool_linear", trainable=False):
    return build_local_model(
        patches=PATCHES,
        embed_dim=DIM,
        step_action_dim=ACTION,
        model_type=model_type,
        projection=projection,
        projection_grid=2,
        projection_trainable=trainable,
        local_dim=LOCAL,
        hidden_dim=32,
        num_layers=2,
        seed=0,
    )


@pytest.mark.parametrize("projection", ["mean_pool_linear", "grid_pool_linear"])
def test_projector_shapes(projection):
    projector = PatchProjector(PATCHES, DIM, LOCAL, mode=projection, grid=2)
    z3 = torch.randn(5, PATCHES, DIM)
    z4 = torch.randn(5, 3, PATCHES, DIM)
    assert projector(z3).shape == (5, LOCAL)
    assert projector(z4).shape == (5, 3, LOCAL)


def test_projector_frozen_by_default():
    projector = PatchProjector(PATCHES, DIM, LOCAL)
    assert all(not p.requires_grad for p in projector.linear.parameters())
    trainable = PatchProjector(PATCHES, DIM, LOCAL, trainable=True)
    assert all(p.requires_grad for p in trainable.linear.parameters())


def test_projector_deterministic_seed():
    a = PatchProjector(PATCHES, DIM, LOCAL, seed=7)
    b = PatchProjector(PATCHES, DIM, LOCAL, seed=7)
    assert torch.equal(a.linear.weight, b.linear.weight)


def test_projector_rejects_bad_shape():
    projector = PatchProjector(PATCHES, DIM, LOCAL)
    with pytest.raises(ValueError):
        projector(torch.randn(2, PATCHES + 1, DIM))


def test_local_dynamics_residual():
    dynamics = LocalDynamics(LOCAL, ACTION, hidden_dim=16, num_layers=1)
    x = torch.randn(3, LOCAL)
    a = torch.zeros(3, ACTION)
    out = dynamics(x, a)
    assert out.shape == (3, LOCAL)


@pytest.mark.parametrize("model_type", ["residual_mlp", "gru_residual"])
def test_rollout_shape_and_action_gradients(model_type):
    model = _build(model_type)
    batch, ctx, k = 4, 2, 3
    z_context = torch.randn(batch, ctx, PATCHES, DIM)
    actions_context = torch.randn(batch, ctx - 1, ACTION)
    actions = torch.randn(batch, k, ACTION, requires_grad=True)
    x_context = model.encode_global_latent(z_context)
    rollout = model.rollout_from_context(x_context, actions_context, actions)
    assert rollout.shape == (batch, k, LOCAL)
    rollout.sum().backward()
    assert actions.grad is not None
    assert float(actions.grad.abs().sum()) > 0


def test_simple_rollout_api():
    model = _build()
    x0 = torch.randn(2, LOCAL)
    actions = torch.randn(2, 5, ACTION)
    assert model.rollout(x0, actions).shape == (2, 5, LOCAL)


def test_gru_context_changes_prediction():
    model = _build("gru_residual")
    z_context = torch.randn(1, 2, PATCHES, DIM)
    actions = torch.randn(1, 2, ACTION)
    x_context = model.encode_global_latent(z_context)
    ctx_a = torch.zeros(1, 1, ACTION)
    ctx_b = torch.ones(1, 1, ACTION)
    out_a = model.rollout_from_context(x_context, ctx_a, actions)
    out_b = model.rollout_from_context(x_context, ctx_b, actions)
    assert not torch.allclose(out_a, out_b)


def test_context_local_dynamics_hidden():
    dynamics = ContextLocalDynamics(LOCAL, ACTION, hidden_dim=16)
    x_context = torch.randn(3, 2, LOCAL)
    actions_context = torch.randn(3, 1, ACTION)
    hidden = dynamics.init_hidden(x_context, actions_context)
    assert hidden.shape == (3, 16)
    next_x, hidden2 = dynamics(x_context[:, -1], torch.randn(3, ACTION), hidden)
    assert next_x.shape == (3, LOCAL)
    assert hidden2.shape == (3, 16)


def test_checkpoint_roundtrip(tmp_path):
    model = _build()
    build_kwargs = dict(
        patches=PATCHES,
        embed_dim=DIM,
        step_action_dim=ACTION,
        model_type="residual_mlp",
        projection="mean_pool_linear",
        projection_grid=2,
        projection_trainable=False,
        local_dim=LOCAL,
        hidden_dim=32,
        num_layers=2,
        seed=0,
    )
    path = tmp_path / "ckpt" / "local_best.pt"
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    save_local_checkpoint(
        path,
        model,
        build_kwargs,
        step=42,
        metrics={"loss_total": 0.5},
        optimizer_state=optimizer.state_dict(),
    )
    loaded, meta = load_local_checkpoint(path)
    assert meta["step"] == 42
    # The trainer resumes from the raw payload; optimizer state must roundtrip.
    payload = torch.load(path, map_location="cpu")
    rebuilt = torch.optim.AdamW(
        [p for p in loaded.parameters() if p.requires_grad], lr=1e-3
    )
    rebuilt.load_state_dict(payload["optimizer_state"])
    z = torch.randn(2, 1, PATCHES, DIM)
    a_ctx = torch.zeros(2, 0, ACTION)
    a = torch.randn(2, 2, ACTION)
    x = model.encode_global_latent(z)
    torch.testing.assert_close(
        model.rollout_from_context(x, a_ctx, a),
        loaded.rollout_from_context(loaded.encode_global_latent(z), a_ctx, a),
    )
