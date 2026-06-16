from pathlib import Path

import pytest

from wm_poc.r2dreamer.checkpoints import inspect_checkpoint


torch = pytest.importorskip("torch")


def test_checkpoint_verifier_accepts_dummy_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "latest.pt"
    torch.save(
        {
            "agent_state_dict": {
                "dummy.weight": torch.zeros(2, 2),
            },
            "optims_state_dict": {},
            "wm_poc_meta": {
                "test": True,
            },
        },
        checkpoint,
    )

    info = inspect_checkpoint(checkpoint)

    assert info["agent_tensor_count"] == 1
    assert info["agent_parameter_count"] == 4
    assert info["has_optimizer_state"] is True
    assert info["wm_poc_meta"] == {"test": True}


def test_checkpoint_verifier_rejects_missing_agent_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bad.pt"
    torch.save({"optims_state_dict": {}}, checkpoint)

    with pytest.raises(KeyError, match="agent_state_dict"):
        inspect_checkpoint(checkpoint)
