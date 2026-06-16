"""Local surrogate models operating on compressed DINO-WM latents.

The local model is intentionally small: a projector maps global patch latents
``(patches, embed_dim)`` to a compact local state, and a residual dynamics
network rolls that state forward under (frameskip-folded) actions. Everything
is differentiable with respect to actions so first-order planners can
backpropagate through rollouts.
"""

from __future__ import annotations

import math
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required to use local_global.models.") from exc


class PatchProjector(nn.Module):
    """Project global patch latents ``[..., P, D]`` to local states ``[..., local_dim]``.

    The linear map is orthogonally initialized and frozen by default: with a
    jointly trained projector, a rollout MSE computed in projected space admits
    the degenerate solution of collapsing every latent to a constant. A frozen
    (near-)orthogonal projection of the LayerNormed pooled features preserves
    latent distances well enough for a first pass; set ``trainable=True`` only
    together with a variance penalty (see losses.variance_penalty).
    """

    def __init__(
        self,
        patches: int,
        embed_dim: int,
        local_dim: int,
        *,
        mode: str = "mean_pool_linear",
        grid: int = 4,
        trainable: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if mode not in {"mean_pool_linear", "grid_pool_linear"}:
            raise ValueError(f"Unknown projection mode: {mode}")
        self.patches = patches
        self.embed_dim = embed_dim
        self.local_dim = local_dim
        self.mode = mode
        self.grid = grid
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        if mode == "grid_pool_linear":
            side = int(math.isqrt(patches))
            if side * side != patches:
                raise ValueError(
                    f"grid_pool_linear needs a square patch count; got {patches}."
                )
            in_dim = grid * grid * embed_dim
        else:
            in_dim = embed_dim
        self.linear = nn.Linear(in_dim, local_dim)
        generator = torch.Generator().manual_seed(int(seed))
        # torch.nn.init.orthogonal_ has no generator argument on older torch;
        # build a seeded (semi-)orthogonal weight from a thin QR factor.
        rows, cols = self.linear.weight.shape
        gauss = torch.randn(max(rows, cols), min(rows, cols), generator=generator)
        q, _ = torch.linalg.qr(gauss)
        with torch.no_grad():
            self.linear.weight.copy_(q.T if cols >= rows else q)
            self.linear.bias.zero_()
        if not trainable:
            for param in self.linear.parameters():
                param.requires_grad_(False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-2] != self.patches or z.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected latents [..., {self.patches}, {self.embed_dim}]; got {tuple(z.shape)}."
            )
        z = self.norm(z.float())
        if self.mode == "mean_pool_linear":
            pooled = z.mean(dim=-2)
        else:
            side = int(math.isqrt(self.patches))
            lead = z.shape[:-2]
            grid_z = z.reshape(-1, side, side, self.embed_dim).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(grid_z, self.grid)
            pooled = pooled.permute(0, 2, 3, 1).reshape(*lead, self.grid * self.grid * self.embed_dim)
        return self.linear(pooled)


class LocalDynamics(nn.Module):
    """Residual MLP transition model: ``x_{t+1} = x_t + f_theta(x_t, a_t)``."""

    def __init__(
        self,
        local_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        in_dim = local_dim + action_dim
        layers: list[nn.Module] = []
        if layer_norm:
            layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(max(0, num_layers - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, local_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([latent, action], dim=-1))
        return latent + delta


class ContextLocalDynamics(nn.Module):
    """GRU residual dynamics: a recurrent state summarizes the context window.

    The same GRU cell consumes context transitions (to infer velocity-like
    information) and predicts forward steps.
    """

    def __init__(
        self,
        local_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        in_dim = local_dim + action_dim
        self.norm = nn.LayerNorm(in_dim) if layer_norm else nn.Identity()
        self.cell = nn.GRUCell(in_dim, hidden_dim)
        head: list[nn.Module] = []
        for _ in range(max(0, num_layers - 1)):
            head.append(nn.Linear(hidden_dim, hidden_dim))
            head.append(nn.SiLU())
        head.append(nn.Linear(hidden_dim, local_dim))
        self.head = nn.Sequential(*head)
        self.hidden_dim = hidden_dim

    def init_hidden(
        self, x_context: torch.Tensor, actions_context: torch.Tensor
    ) -> torch.Tensor:
        """Consume ``context_len - 1`` transitions; returns hidden state (B, H)."""
        batch = x_context.shape[0]
        hidden = x_context.new_zeros(batch, self.hidden_dim)
        for t in range(actions_context.shape[1]):
            inp = self.norm(torch.cat([x_context[:, t], actions_context[:, t]], dim=-1))
            hidden = self.cell(inp, hidden)
        return hidden

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = self.norm(torch.cat([latent, action], dim=-1))
        hidden = self.cell(inp, hidden)
        return latent + self.head(hidden), hidden


class LocalRolloutModel(nn.Module):
    """Projector + dynamics wrapper exposing encode/step/rollout."""

    def __init__(
        self,
        projector: PatchProjector,
        dynamics: nn.Module,
        *,
        model_type: str = "residual_mlp",
    ) -> None:
        super().__init__()
        if model_type not in {"residual_mlp", "gru_residual"}:
            raise ValueError(f"Unknown local model type: {model_type}")
        self.projector = projector
        self.dynamics = dynamics
        self.model_type = model_type

    @property
    def local_dim(self) -> int:
        return self.projector.local_dim

    def encode_global_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z)

    def step(
        self, x: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.model_type == "gru_residual":
            if hidden is None:
                hidden = x.new_zeros(x.shape[0], self.dynamics.hidden_dim)
            next_x, hidden = self.dynamics(x, action, hidden)
            return next_x, hidden
        return self.dynamics(x, action), None

    def rollout(self, x0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll a single local state forward: ``(B, X), (B, K, A) -> (B, K, X)``."""
        x = x0
        hidden: torch.Tensor | None = None
        states = []
        for t in range(actions.shape[1]):
            x, hidden = self.step(x, actions[:, t], hidden)
            states.append(x)
        return torch.stack(states, dim=1)

    def rollout_from_context(
        self,
        x_context: torch.Tensor,
        actions_context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Rollout conditioned on a context window: ``(B, C, X), (B, C-1, A), (B, K, A)``."""
        hidden: torch.Tensor | None = None
        if self.model_type == "gru_residual" and actions_context.shape[1] > 0:
            hidden = self.dynamics.init_hidden(x_context, actions_context)
        x = x_context[:, -1]
        states = []
        for t in range(actions.shape[1]):
            x, hidden = self.step(x, actions[:, t], hidden)
            states.append(x)
        return torch.stack(states, dim=1)


def save_local_checkpoint(
    path: str | Path,
    model: LocalRolloutModel,
    build_kwargs: dict,
    *,
    step: int,
    metrics: dict | None = None,
    optimizer_state: dict | None = None,
) -> None:
    """Save the surrogate with everything needed to rebuild it standalone.

    ``optimizer_state`` (saved into ``local_latest.pt`` by the trainer) makes
    interrupted runs resumable without losing Adam moment estimates.
    """
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "wm_poc_local_surrogate_v1",
        "build_kwargs": dict(build_kwargs),
        "model_state": model.state_dict(),
        "step": int(step),
        "metrics": dict(metrics or {}),
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)


def load_local_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[LocalRolloutModel, dict]:
    """Rebuild a surrogate from a checkpoint; returns ``(model, payload_meta)``."""
    payload = torch.load(path, map_location=device)
    if payload.get("format") != "wm_poc_local_surrogate_v1":
        raise ValueError(f"Unexpected local checkpoint format in {path}.")
    model = build_local_model(**payload["build_kwargs"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    meta = {k: payload[k] for k in ("step", "metrics", "build_kwargs")}
    return model, meta


def build_local_model(
    *,
    patches: int,
    embed_dim: int,
    step_action_dim: int,
    model_type: str = "residual_mlp",
    projection: str = "mean_pool_linear",
    projection_grid: int = 4,
    projection_trainable: bool = False,
    local_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 3,
    layer_norm: bool = True,
    seed: int = 0,
) -> LocalRolloutModel:
    projector = PatchProjector(
        patches,
        embed_dim,
        local_dim,
        mode=projection,
        grid=projection_grid,
        trainable=projection_trainable,
        seed=seed,
    )
    if model_type == "gru_residual":
        dynamics: nn.Module = ContextLocalDynamics(
            local_dim, step_action_dim, hidden_dim, num_layers, layer_norm
        )
    else:
        dynamics = LocalDynamics(local_dim, step_action_dim, hidden_dim, num_layers, layer_norm)
    return LocalRolloutModel(projector, dynamics, model_type=model_type)
