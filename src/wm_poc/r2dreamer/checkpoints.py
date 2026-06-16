from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    """Load a DreamerV3/R2-Dreamer training checkpoint for inspection.

    These checkpoints are *full* training states (model weights + optimizer
    moments + ``wm_poc_meta``), not bare tensor state dicts, so they require the
    full unpickler (``weights_only=False``). ``torch.load`` then executes
    arbitrary code from the pickle, so this verifier must only be pointed at
    **trusted** local checkpoints produced by this project -- never untrusted
    downloads. Passing ``weights_only=False`` explicitly keeps behaviour stable
    across the PyTorch>=2.6 default flip to ``weights_only=True`` (which cannot
    load these full checkpoints) and removes the ambiguous-default warning.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to inspect .pt checkpoints.") from exc

    expanded = path.expanduser()
    try:
        checkpoint = torch.load(expanded, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - torch<1.13 has no weights_only kwarg
        checkpoint = torch.load(expanded, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dictionary in {path}.")
    return checkpoint


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = _load_torch_checkpoint(path)
    if "agent_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint {path} is missing 'agent_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    agent_state = checkpoint["agent_state_dict"]
    if not isinstance(agent_state, dict):
        raise ValueError("'agent_state_dict' is not a dictionary.")

    tensor_count = 0
    parameter_count = 0
    for value in agent_state.values():
        if hasattr(value, "numel"):
            tensor_count += 1
            parameter_count += int(value.numel())

    return {
        "path": str(path.expanduser()),
        "top_level_keys": sorted(checkpoint.keys()),
        "agent_tensor_count": tensor_count,
        "agent_parameter_count": parameter_count,
        "has_optimizer_state": "optims_state_dict" in checkpoint,
        "wm_poc_meta": checkpoint.get("wm_poc_meta", {}),
    }
