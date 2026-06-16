#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from wm_poc.dino_wm.finetune import initialize_from_checkpoint  # noqa: E402


def _config(action_encoder: bool = True, strict: bool = True) -> dict:
    return {
        "finetuning": {
            "load": {
                "predictor": True,
                "action_encoder": action_encoder,
                "decoder": False,
                "optimizer": False,
            },
            "strict": strict,
            "reset_epoch": True,
            "freeze": {
                "visual_encoder": True,
                "predictor": False,
                "action_encoder": False,
                "decoder": True,
            },
        }
    }


def main() -> int:
    try:
        import torch
        from torch import nn
    except ImportError:
        print("PyTorch is not installed; skipped tensor-level fine-tune loading checks.")
        print("Run this script again in the DINO-WM training environment for full verification.")
        return 0

    class TinyDinoWM(nn.Module):
        def __init__(self, action_dim: int = 2) -> None:
            super().__init__()
            self.visual_encoder = nn.Linear(4, 4)
            self.predictor = nn.Linear(4, 4)
            self.action_encoder = nn.Linear(action_dim, 4)
            self.decoder = nn.Linear(4, 2)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = TinyDinoWM(action_dim=2)
        checkpoint = tmp_path / "tiny.pt"
        torch.save(
            {
                "model_state_dict": source.state_dict(),
                "optimizer_state_dict": {"ignored": True},
                "epoch": 3,
            },
            checkpoint,
        )

        predictor_only = TinyDinoWM(action_dim=2)
        result = initialize_from_checkpoint(predictor_only, checkpoint, _config(action_encoder=False))
        assert any(key.startswith("predictor.") for key in result.loaded_keys)
        assert all(not key.startswith("action_encoder.") for key in result.loaded_keys)
        assert not result.optimizer_loaded

        predictor_action = TinyDinoWM(action_dim=2)
        result = initialize_from_checkpoint(predictor_action, checkpoint, _config(action_encoder=True))
        assert any(key.startswith("predictor.") for key in result.loaded_keys)
        assert any(key.startswith("action_encoder.") for key in result.loaded_keys)

        mismatched = TinyDinoWM(action_dim=3)
        result = initialize_from_checkpoint(mismatched, checkpoint, _config(action_encoder=True))
        assert result.action_dimension_mismatch
        assert not any(key.startswith("action_encoder.") for key in result.loaded_keys)
        assert not result.strict
        assert "action_encoder_shape_mismatch_reinitialized" in result.notes

    print("DINO-WM fine-tune partial loading checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
