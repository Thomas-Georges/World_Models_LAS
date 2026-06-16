from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wm_poc.dino_wm.configs import get_config_value


COMPONENT_PREFIXES = {
    "predictor": ("predictor.", "model.predictor.", "world_model.predictor."),
    "action_encoder": ("action_encoder.", "model.action_encoder.", "world_model.action_encoder."),
    "decoder": ("decoder.", "model.decoder.", "world_model.decoder."),
    "visual_encoder": ("visual_encoder.", "encoder.", "dinov2.", "model.visual_encoder."),
}


@dataclass
class FinetuneLoadResult:
    checkpoint_path: str
    loaded_keys: list[str]
    skipped_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    action_dimension_mismatch: bool
    optimizer_loaded: bool
    epoch_reset: bool
    strict: bool
    notes: list[str]

    def to_manifest(self) -> dict[str, Any]:
        return asdict(self)


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for DINO-WM fine-tune loading.") from exc
    payload = torch.load(path.expanduser(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected checkpoint dictionary at {path}.")
    return payload


def _state_dict_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("model_state_dict", "state_dict", "agent_state_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    tensor_like = [key for key, value in payload.items() if hasattr(value, "shape")]
    if tensor_like:
        return payload
    raise KeyError(f"Could not find a model state dict in checkpoint keys: {list(payload.keys())}")


def _component_for_key(key: str) -> str | None:
    for component, prefixes in COMPONENT_PREFIXES.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            return component
    return None


def _allowed_components(config: dict[str, Any]) -> set[str]:
    allowed = set()
    for component in ("predictor", "action_encoder", "decoder"):
        if get_config_value(config, f"finetuning.load.{component}", False):
            allowed.add(component)
    return allowed


def _shapes_match(model_state: dict[str, Any], checkpoint_state: dict[str, Any], key: str) -> bool:
    if key not in model_state:
        return False
    left = getattr(model_state[key], "shape", None)
    right = getattr(checkpoint_state[key], "shape", None)
    return left == right


def _detect_action_dimension_mismatch(
    model_state: dict[str, Any],
    checkpoint_state: dict[str, Any],
) -> bool:
    for key in checkpoint_state:
        if _component_for_key(key) == "action_encoder" and key in model_state:
            if not _shapes_match(model_state, checkpoint_state, key):
                return True
    return False


def apply_freeze_config(model: Any, config: dict[str, Any]) -> None:
    freeze = get_config_value(config, "finetuning.freeze", {}) or {}
    for component, should_freeze in freeze.items():
        module = getattr(model, component, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = not bool(should_freeze)


def initialize_from_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    config: dict[str, Any],
    optimizer: Any | None = None,
) -> FinetuneLoadResult:
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Fine-tune checkpoint does not exist: {checkpoint}")

    payload = _torch_load(checkpoint)
    checkpoint_state = _state_dict_from_payload(payload)
    model_state = model.state_dict()
    allowed = _allowed_components(config)
    strict = bool(get_config_value(config, "finetuning.strict", True))
    load_optimizer = bool(get_config_value(config, "finetuning.load.optimizer", False))
    reset_epoch = bool(get_config_value(config, "finetuning.reset_epoch", True))

    action_mismatch = _detect_action_dimension_mismatch(model_state, checkpoint_state)
    if action_mismatch:
        allowed.discard("action_encoder")
        strict = False

    filtered: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in checkpoint_state.items():
        component = _component_for_key(key)
        if component not in allowed:
            skipped.append(key)
            continue
        if not _shapes_match(model_state, checkpoint_state, key):
            skipped.append(key)
            strict = False
            continue
        filtered[key] = value

    result = model.load_state_dict(filtered, strict=False)
    optimizer_loaded = False
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        optimizer_loaded = True

    apply_freeze_config(model, config)

    notes: list[str] = []
    if action_mismatch:
        notes.append("action_encoder_shape_mismatch_reinitialized")
    if not optimizer_loaded and load_optimizer:
        notes.append("optimizer_requested_but_missing_or_not_provided")
    if reset_epoch:
        notes.append("epoch_step_counters_reset")

    return FinetuneLoadResult(
        checkpoint_path=str(checkpoint),
        loaded_keys=sorted(filtered.keys()),
        skipped_keys=sorted(skipped),
        missing_keys=list(getattr(result, "missing_keys", [])),
        unexpected_keys=list(getattr(result, "unexpected_keys", [])),
        action_dimension_mismatch=action_mismatch,
        optimizer_loaded=optimizer_loaded,
        epoch_reset=reset_epoch,
        strict=strict,
        notes=notes,
    )
