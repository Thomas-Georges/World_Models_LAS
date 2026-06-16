"""Record planning videos for decoder-free DINO-WM models.

Upstream gates all video recording on ``self.wm.decoder is not None`` because
the comparison layout includes an imagined-rollout panel decoded to pixels.
The no-decoder runs therefore never produced a single video, while the only
decoder-bearing run (the legacy online smoke) silently became the sole source
of MP4s. This patch adds an else-branch that records the executed environment
rollout against the goal image, with the imagined panel zeroed out — honest
black instead of fake pixels.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


PATCH_MARKER = "WM_POC_DINO_VIDEO_NO_DECODER_PATCH"

_ANCHOR = """        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )
"""

_REPLACEMENT = f"""        # plot trajs
        if self.wm.decoder is not None:
            i_visuals = self.wm.decode_obs(i_z_obses)[0]["visual"]
            i_visuals = self._mask_traj(
                i_visuals, action_len + 1
            )  # we have action_len + 1 states
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )
        else:  # {PATCH_MARKER}: no decoder, record executed rollout vs goal
            e_visuals = self.preprocessor.transform_obs_visual(e_visuals)
            e_visuals = self._mask_traj(e_visuals, action_len * self.frameskip + 1)
            i_visuals = torch.zeros(
                (e_visuals.shape[0], i_z_obses["visual"].shape[1]) + tuple(e_visuals.shape[2:]),
                dtype=e_visuals.dtype,
            )
            self._plot_rollout_compare(
                e_visuals=e_visuals,
                i_visuals=i_visuals,
                successes=successes,
                save_video=save_video,
                filename=filename,
            )
"""


def patch_evaluator_source(source: str) -> tuple[str, bool]:
    if PATCH_MARKER in source:
        return source, False
    if _ANCHOR not in source:
        raise ValueError(
            "Could not apply DINO-WM evaluator video patch; decoder-gated plot "
            "block not found in planning/evaluator.py."
        )
    return source.replace(_ANCHOR, _REPLACEMENT, 1), True


def patch_evaluator_file(evaluator_path: Path) -> bool:
    evaluator_path = evaluator_path.expanduser()
    source = evaluator_path.read_text(encoding="utf-8")
    patched, changed = patch_evaluator_source(source)
    if not changed:
        return False

    backup_dir = evaluator_path.parent / ".wm_poc_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"evaluator.py.video_no_decoder.{stamp}"
    shutil.copy2(evaluator_path, backup_path)
    evaluator_path.write_text(patched, encoding="utf-8")
    return True
