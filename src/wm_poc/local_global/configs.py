"""Config loading and validation for the local/global planning track.

Reuses the generic YAML conventions from :mod:`wm_poc.dino_wm.configs`:
``extends:`` single-parent inheritance and ``${oc.env:VAR,default}``
environment placeholders. The dict stays the source of truth; the typed
dataclasses below are validated views used by models/planners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wm_poc.dino_wm.configs import (
    get_config_value,
    load_config,
    resolve_config,
    set_config_value,
    write_yaml,
)

REQUIRED_SECTIONS = ("paths", "global_model", "local_model", "training", "planning", "evaluation")
VALID_MODEL_TYPES = {"residual_mlp", "gru_residual"}
VALID_PROJECTIONS = {"mean_pool_linear", "grid_pool_linear"}
VALID_PLANNERS = (
    "global_cem",
    "local_cem",
    "local_gd",
    "local_adam",
    "hybrid_cem_local_refine",
    "hybrid_cem_local_refine_global_rescore",
)
RUN_GATE_ENV = "RUN_LOCAL_GLOBAL"


@dataclass(frozen=True)
class GlobalModelConfig:
    source: str
    encoder: str
    image_size: int
    latent_patches: int
    latent_dim: int
    frameskip: int
    num_hist: int
    proprio_dim: int


@dataclass(frozen=True)
class LocalModelConfig:
    model_type: str
    projection: str
    projection_grid: int
    projection_trainable: bool
    local_dim: int
    hidden_dim: int
    num_layers: int
    context_len: int
    rollout_steps: int
    layer_norm: bool


@dataclass(frozen=True)
class PlannerConfig:
    action_dim: int
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    horizon: int
    goal_steps: int
    mpc_exec_steps: int
    cem_population: int
    cem_elites: int
    cem_iters: int
    cem_init_std: float
    gd_iters: int
    gd_lr: float
    gradient_clip: float | None
    action_smoothness: float
    # Re-score rejection is selected by planner name (hybrid_cem_local_refine
    # vs hybrid_cem_local_refine_global_rescore); this knob only tunes it.
    reject_refine_if_worse_by: float
    frameskip: int = 1

    @property
    def step_action_dim(self) -> int:
        """Action dimension per model step (frameskip raw actions folded)."""
        return self.action_dim * self.frameskip

    @property
    def step_action_low(self) -> tuple[float, ...]:
        return tuple(self.action_low) * self.frameskip

    @property
    def step_action_high(self) -> tuple[float, ...]:
        return tuple(self.action_high) * self.frameskip


@dataclass(frozen=True)
class LocalGlobalConfig:
    raw: dict[str, Any] = field(repr=False)
    task: str
    run_name: str
    seed: int
    device: str
    global_model: GlobalModelConfig
    local_model: LocalModelConfig
    planner: PlannerConfig


def validate_local_global_config(config: dict[str, Any]) -> None:
    track = config.get("track")
    if track != "local_global":
        raise ValueError(f"Expected track=local_global, got {track!r}.")
    for section in REQUIRED_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid local_global config section: {section}")
    if not config.get("task"):
        raise ValueError("task is required (e.g. point_maze or synthetic).")

    model_type = get_config_value(config, "local_model.model_type")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(f"local_model.model_type must be one of {sorted(VALID_MODEL_TYPES)}.")
    projection = get_config_value(config, "local_model.projection")
    if projection not in VALID_PROJECTIONS:
        raise ValueError(f"local_model.projection must be one of {sorted(VALID_PROJECTIONS)}.")
    if int(get_config_value(config, "local_model.context_len", 1)) < 1:
        raise ValueError("local_model.context_len must be >= 1.")
    if int(get_config_value(config, "local_model.rollout_steps", 1)) < 1:
        raise ValueError("local_model.rollout_steps must be >= 1.")
    if int(get_config_value(config, "local_model.local_dim", 0)) < 1:
        raise ValueError("local_model.local_dim must be >= 1.")

    action_dim = int(get_config_value(config, "planning.action_dim", 0))
    low = get_config_value(config, "planning.action_low")
    high = get_config_value(config, "planning.action_high")
    if action_dim < 1:
        raise ValueError("planning.action_dim must be >= 1.")
    if not isinstance(low, list) or not isinstance(high, list):
        raise ValueError("planning.action_low/action_high must be lists.")
    if len(low) != action_dim or len(high) != action_dim:
        raise ValueError("planning.action_low/action_high must have length action_dim.")
    if any(float(lo) >= float(hi) for lo, hi in zip(low, high)):
        raise ValueError("planning.action_low must be elementwise below action_high.")
    if int(get_config_value(config, "planning.horizon", 0)) < 1:
        raise ValueError("planning.horizon must be >= 1.")
    if int(get_config_value(config, "planning.goal_steps", 0)) < 1:
        raise ValueError("planning.goal_steps must be >= 1.")
    if int(get_config_value(config, "planning.mpc_exec_steps", 0)) < 1:
        raise ValueError("planning.mpc_exec_steps must be >= 1.")
    if int(get_config_value(config, "planning.cem_elites", 1)) > int(
        get_config_value(config, "planning.cem_population", 0)
    ):
        raise ValueError("planning.cem_elites must be <= planning.cem_population.")

    val_fraction = float(get_config_value(config, "training.val_fraction", 0.1))
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("training.val_fraction must be in (0, 1).")

    planners = get_config_value(config, "evaluation.planners", [])
    unknown = [p for p in planners if p not in VALID_PLANNERS]
    if unknown:
        raise ValueError(f"Unknown evaluation.planners entries: {unknown}.")


def load_local_global_config(path: str | Path) -> dict[str, Any]:
    """Load, env-resolve, and validate a local/global YAML config."""
    config = resolve_config(load_config(path))
    validate_local_global_config(config)
    return config


def _planner_config(config: dict[str, Any]) -> PlannerConfig:
    clip = get_config_value(config, "planning.gradient_clip")
    return PlannerConfig(
        action_dim=int(get_config_value(config, "planning.action_dim")),
        action_low=tuple(float(v) for v in get_config_value(config, "planning.action_low")),
        action_high=tuple(float(v) for v in get_config_value(config, "planning.action_high")),
        horizon=int(get_config_value(config, "planning.horizon")),
        goal_steps=int(get_config_value(config, "planning.goal_steps")),
        mpc_exec_steps=int(get_config_value(config, "planning.mpc_exec_steps")),
        cem_population=int(get_config_value(config, "planning.cem_population")),
        cem_elites=int(get_config_value(config, "planning.cem_elites")),
        cem_iters=int(get_config_value(config, "planning.cem_iters")),
        cem_init_std=float(get_config_value(config, "planning.cem_init_std", 0.5)),
        gd_iters=int(get_config_value(config, "planning.gd_iters")),
        gd_lr=float(get_config_value(config, "planning.gd_lr")),
        gradient_clip=None if clip in (None, "") else float(clip),
        action_smoothness=float(get_config_value(config, "planning.action_smoothness", 0.0)),
        reject_refine_if_worse_by=float(
            get_config_value(config, "planning.reject_refine_if_worse_by", 0.05)
        ),
        frameskip=int(get_config_value(config, "global_model.frameskip", 1)),
    )


def typed_config(config: dict[str, Any]) -> LocalGlobalConfig:
    """Build the typed view of an already loaded+resolved config dict."""
    validate_local_global_config(config)
    global_model = GlobalModelConfig(
        source=str(get_config_value(config, "global_model.source")),
        encoder=str(get_config_value(config, "global_model.encoder")),
        image_size=int(get_config_value(config, "global_model.image_size", 224)),
        latent_patches=int(get_config_value(config, "global_model.latent_patches", 196)),
        latent_dim=int(get_config_value(config, "global_model.latent_dim", 384)),
        frameskip=int(get_config_value(config, "global_model.frameskip", 1)),
        num_hist=int(get_config_value(config, "global_model.num_hist", 3)),
        proprio_dim=int(get_config_value(config, "global_model.proprio_dim", 0)),
    )
    local_model = LocalModelConfig(
        model_type=str(get_config_value(config, "local_model.model_type")),
        projection=str(get_config_value(config, "local_model.projection")),
        projection_grid=int(get_config_value(config, "local_model.projection_grid", 4)),
        projection_trainable=bool(get_config_value(config, "local_model.projection_trainable", False)),
        local_dim=int(get_config_value(config, "local_model.local_dim")),
        hidden_dim=int(get_config_value(config, "local_model.hidden_dim")),
        num_layers=int(get_config_value(config, "local_model.num_layers", 3)),
        context_len=int(get_config_value(config, "local_model.context_len", 1)),
        rollout_steps=int(get_config_value(config, "local_model.rollout_steps", 3)),
        layer_norm=bool(get_config_value(config, "local_model.layer_norm", True)),
    )
    return LocalGlobalConfig(
        raw=config,
        task=str(config["task"]),
        run_name=str(config.get("run_name") or ""),
        seed=int(config.get("seed", 0)),
        device=str(config.get("device", "auto")),
        global_model=global_model,
        local_model=local_model,
        planner=_planner_config(config),
    )


def resolve_run_dir(config: dict[str, Any], run_dir: str | Path | None = None) -> Path:
    """Resolve (and create) the run directory plus its standard subdirectories."""
    if run_dir is not None:
        path = Path(run_dir).expanduser()
    else:
        run_name = config.get("run_name")
        if not run_name:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_name = f"local_global_{config.get('task', 'task')}_{stamp}"
            config["run_name"] = run_name
        run_root = Path(str(get_config_value(config, "paths.run_root"))).expanduser()
        path = run_root / str(run_name)
    path.mkdir(parents=True, exist_ok=True)
    for subdir in ("transition_data", "checkpoints", "metrics", "planning", "figures"):
        (path / subdir).mkdir(exist_ok=True)
    set_config_value(config, "paths.run_dir", str(path))
    return path


def latent_cache_dir(config: dict[str, Any]) -> Path:
    """Cache dir for the configured task, matching the DINO-WM layout.

    ``paths.latent_cache_dir`` wins when set; otherwise the directory is
    derived as ``<paths.latent_cache_root>/<task>/<encoder>_img<image_size>``.
    """
    explicit = get_config_value(config, "paths.latent_cache_dir")
    if explicit:
        return Path(str(explicit)).expanduser()
    root = Path(str(get_config_value(config, "paths.latent_cache_root"))).expanduser()
    encoder = str(get_config_value(config, "global_model.encoder", "dinov2_vits14"))
    image_size = int(get_config_value(config, "global_model.image_size", 224))
    return root / str(config["task"]) / f"{encoder}_img{image_size}"


def action_data_dir(config: dict[str, Any]) -> Path:
    """Directory holding the upstream actions/states/seq_lengths tensors."""
    explicit = get_config_value(config, "paths.action_data_dir")
    if explicit:
        return Path(str(explicit)).expanduser()
    return Path(str(get_config_value(config, "paths.data_root"))).expanduser() / str(config["task"])


def save_resolved_config(config: dict[str, Any], run_dir: Path) -> Path:
    out = Path(run_dir) / "config_resolved.yaml"
    write_yaml(out, config)
    return out
