#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes.util
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_R2_REPO = Path(os.environ.get("R2DREAMER_REPO", "/content/external_repos/r2dreamer"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that a patched R2-Dreamer DMC vision env returns nonblank rendered images."
    )
    parser.add_argument("--r2-repo", type=Path, default=DEFAULT_R2_REPO, help="External r2dreamer repo.")
    parser.add_argument("--task", default="dmc_walker_walk", help="DMC task, e.g. dmc_walker_walk.")
    parser.add_argument("--size", type=int, default=64, help="Square rendered image size.")
    parser.add_argument("--action-repeat", type=int, default=2, help="DMC action repeat.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--min-std", type=float, default=1.0, help="Minimum pixel std for a real frame.")
    parser.add_argument("--min-range", type=float, default=2.0, help="Minimum pixel range for a real frame.")
    parser.add_argument("--out", type=Path, help="Optional output PNG for visual inspection.")
    return parser.parse_args()


def dmc_task_name(task: str) -> str:
    return task.split("_", 1)[1] if task.startswith("dmc_") else task


def save_image(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on Colab/runtime deps.
        raise RuntimeError("Pillow is required when --out is provided.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def _headless_render_error() -> RuntimeError:
    backend = os.environ.get("MUJOCO_GL", "<unset>")
    platform = os.environ.get("PYOPENGL_PLATFORM", "<unset>")
    osmesa_library = ctypes.util.find_library("OSMesa")
    return RuntimeError(
        "DMC vision rendering failed before a reset frame could be produced. "
        f"Current MUJOCO_GL={backend}, PYOPENGL_PLATFORM={platform}, "
        f"ctypes.find_library('OSMesa')={osmesa_library!r}. "
        "If MUJOCO_GL=osmesa, install OSMesa system libraries in this Colab runtime "
        "and rerun the MuJoCo setup cell: "
        "apt-get update && apt-get install -y libosmesa6 libosmesa6-dev "
        "libgl1-mesa-glx libglfw3. "
        "The earlier TensorFlow cuFFT/cuDNN/cuBLAS registration messages are usually harmless; "
        "the relevant failure is the headless OpenGL backend setup."
    )


def _looks_like_headless_render_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "glGetError" in message
        or "OSMesa" in message
        or "GLFW" in message
        or "gladLoadGL" in message
        or "DISPLAY" in message
    )


def main() -> int:
    args = parse_args()
    r2_repo = args.r2_repo.expanduser()
    if not r2_repo.is_dir():
        raise FileNotFoundError(f"R2-Dreamer repo does not exist: {r2_repo}")

    sys.path.insert(0, str(r2_repo))
    try:
        from envs.dmc import DeepMindControl  # noqa: PLC0415

        env = DeepMindControl(
            dmc_task_name(args.task),
            action_repeat=args.action_repeat,
            size=(args.size, args.size),
            seed=args.seed,
        )
    except Exception as exc:
        if _looks_like_headless_render_error(exc):
            raise _headless_render_error() from exc
        raise
    try:
        try:
            obs = env.reset()
            image = np.asarray(obs["image"])
        except Exception as exc:
            if _looks_like_headless_render_error(exc):
                raise _headless_render_error() from exc
            raise
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()

    stats = {
        "shape": tuple(image.shape),
        "dtype": str(image.dtype),
        "min": float(image.min()),
        "max": float(image.max()),
        "mean": float(image.mean()),
        "std": float(image.std()),
    }
    print(f"WM_POC_DMC_DISABLE_IMAGE_RENDER={os.environ.get('WM_POC_DMC_DISABLE_IMAGE_RENDER', '<unset>')}")
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}")
    print(f"PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM', '<unset>')}")
    for key, value in stats.items():
        print(f"image.{key}={value}")

    if args.out:
        save_image(args.out, image)
        print(f"wrote preview: {args.out}")

    pixel_range = stats["max"] - stats["min"]
    if stats["std"] < args.min_std or pixel_range < args.min_range:
        print(
            "DMC vision render check failed: image looks blank. "
            "For real dmc_vision training, set WM_POC_DMC_DISABLE_IMAGE_RENDER=false.",
            file=sys.stderr,
        )
        return 1

    print("DMC vision render check passed: image has nonblank pixel variation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
