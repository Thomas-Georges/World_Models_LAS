from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedR2Agent:
    config: Any
    agent: Any
    env: Any
    load_info: dict[str, Any]


@dataclass
class RolloutData:
    rewards: list[float]
    frames: list[Any]
    features: list[Any]
    deter: list[Any]
    stoch: list[Any]
    actions: list[Any]
    episode_index: list[int]
    step_index: list[int]

    @property
    def total_return(self) -> float:
        return float(sum(self.rewards))

    @property
    def length(self) -> int:
        return len(self.rewards)


def ensure_upstream_on_path(r2_repo: Path) -> Path:
    repo = r2_repo.expanduser().resolve()
    if not (repo / "dreamer.py").is_file():
        raise FileNotFoundError(
            f"Could not find upstream R2-Dreamer at {repo}. "
            "Set R2DREAMER_REPO or pass --r2-repo."
        )
    repo_text = str(repo)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    return repo


def configure_headless_mujoco(backend: str) -> str:
    selected = backend
    if selected == "auto":
        if _looks_like_colab():
            selected = "osmesa"
        else:
            selected = "egl" if _cuda_is_available() else "osmesa"
    elif selected == "egl" and _looks_like_colab() and os.environ.get("R2_ALLOW_EGL") != "1":
        print(
            "Colab EGL rendering commonly segfaults; using MUJOCO_GL=osmesa instead. "
            "Set R2_ALLOW_EGL=1 to force egl."
        )
        selected = "osmesa"

    os.environ["MUJOCO_GL"] = selected
    if selected in {"egl", "osmesa"}:
        os.environ["PYOPENGL_PLATFORM"] = selected
    elif selected == "glfw":
        os.environ.pop("PYOPENGL_PLATFORM", None)
    if selected == "egl":
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp/xdg-runtime"))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        runtime_dir.chmod(0o700)
    except PermissionError:
        pass
    os.environ["XDG_RUNTIME_DIR"] = str(runtime_dir)
    return selected


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _looks_like_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir()


def compose_upstream_config(
    *,
    r2_repo: Path,
    env_name: str,
    task: str,
    model: str,
    rep_loss: str,
    seed: int,
    device: str,
    extra_overrides: list[str] | None = None,
) -> Any:
    repo = ensure_upstream_on_path(r2_repo)
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Hydra and OmegaConf are required to compose the upstream R2-Dreamer config. "
            "Run this inside the environment where R2-Dreamer was installed."
        ) from exc

    overrides = [
        f"env={env_name}",
        f"model={model}",
        f"env.task={task}",
        f"model.rep_loss={rep_loss}",
        f"seed={int(seed)}",
        f"device={device}",
        "model.compile=false",
        "env.env_num=1",
        "env.eval_episode_num=0",
        "trainer.eval_episode_num=0",
    ]
    overrides.extend(extra_overrides or [])

    hydra_state = GlobalHydra.instance()
    if hydra_state.is_initialized():
        hydra_state.clear()
    with initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        config = compose(config_name="configs", overrides=overrides)
    OmegaConf.resolve(config)
    return config


def load_agent_and_env(
    *,
    r2_repo: Path,
    checkpoint: Path,
    config: Any,
    strict: bool = True,
) -> LoadedR2Agent:
    ensure_upstream_on_path(r2_repo)
    try:
        import torch
        from dreamer import Dreamer
        from envs import make_env
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch and the upstream R2-Dreamer modules are required for checkpoint evaluation."
        ) from exc

    checkpoint_path = checkpoint.expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    env = make_env(config.env, 0)
    agent = Dreamer(config.model, env.observation_space, env.action_space).to(config.device)
    payload = torch.load(checkpoint_path, map_location=config.device)
    if not isinstance(payload, dict) or "agent_state_dict" not in payload:
        keys = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        raise KeyError(
            f"Checkpoint {checkpoint_path} is missing agent_state_dict. Available: {keys}"
        )

    result = agent.load_state_dict(payload["agent_state_dict"], strict=strict)
    if hasattr(agent, "clone_and_freeze"):
        agent.clone_and_freeze()
    agent.eval()

    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    return LoadedR2Agent(
        config=config,
        agent=agent,
        env=env,
        load_info={
            "checkpoint": str(checkpoint_path),
            "strict": strict,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "wm_poc_meta": payload.get("wm_poc_meta", {}),
        },
    )


def _obs_to_tensordict(obs: dict[str, Any], *, device: str) -> Any:
    try:
        import torch
        from tensordict import TensorDict
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch and tensordict are required for R2-Dreamer rollouts.") from exc

    tensors = {}
    for key, value in obs.items():
        tensor = torch.as_tensor(value, device=device)
        tensor = tensor.unsqueeze(0)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        tensors[key] = tensor
    return TensorDict(tensors, batch_size=(1,), device=device)


def _tensor_to_numpy(value: Any) -> Any:
    return value.detach().cpu().numpy()


def _frame_from_obs(obs: dict[str, Any]) -> Any | None:
    frame = obs.get("image")
    if frame is None:
        return None
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required to collect rollout video frames.") from exc

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def run_policy_rollouts(
    *,
    loaded: LoadedR2Agent,
    episodes: int,
    max_steps: int,
    collect_frames: bool,
    collect_latents: bool,
) -> RolloutData:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for R2-Dreamer rollouts.") from exc

    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    agent = loaded.agent
    env = loaded.env
    all_rewards: list[float] = []
    frames: list[Any] = []
    features: list[Any] = []
    deter: list[Any] = []
    stoch: list[Any] = []
    actions: list[Any] = []
    episode_index: list[int] = []
    step_index: list[int] = []

    with torch.no_grad():
        for episode in range(episodes):
            try:
                obs = env.reset()
            except Exception as exc:
                raise _headless_mujoco_error(exc) from exc
            state = agent.get_initial_state(1)
            for step in range(max_steps):
                if collect_frames:
                    frame = _frame_from_obs(obs)
                    if frame is not None:
                        frames.append(frame)

                transition = _obs_to_tensordict(obs, device=agent.device)
                action, state = agent.act(transition, state, eval=True)

                if collect_latents:
                    feat = agent._frozen_rssm.get_feat(state["stoch"], state["deter"])
                    features.append(_tensor_to_numpy(feat[0]))
                    deter.append(_tensor_to_numpy(state["deter"][0]))
                    stoch.append(_tensor_to_numpy(state["stoch"][0]).reshape(-1))
                    actions.append(_tensor_to_numpy(action[0]))
                    episode_index.append(episode)
                    step_index.append(step)

                try:
                    obs, reward, done, _ = env.step(_tensor_to_numpy(action[0]))
                except Exception as exc:
                    raise _headless_mujoco_error(exc) from exc
                all_rewards.append(float(reward))
                if done:
                    break

    return RolloutData(
        rewards=all_rewards,
        frames=frames,
        features=features,
        deter=deter,
        stoch=stoch,
        actions=actions,
        episode_index=episode_index,
        step_index=step_index,
    )


def _headless_mujoco_error(exc: Exception) -> Exception:
    message = str(exc)
    if "gladLoadGL" not in message and "DISPLAY" not in message and "GLFW" not in message:
        return exc
    return RuntimeError(
        "MuJoCo failed to create a headless rendering context. "
        "Use --mujoco-gl osmesa on Colab, even on GPU runtimes; the policy can still run on CUDA. "
        "Try --mujoco-gl egl only after confirming EGL is stable in the runtime. "
        "For CPU Colab, OSMesa system libraries may also be required: "
        "apt-get install -y libosmesa6 libgl1-mesa-glx libglfw3."
    )


def save_video(frames: list[Any], path: Path, *, fps: int) -> None:
    if not frames:
        raise ValueError("No video frames were collected.")
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "imageio is required to write MP4 rollout videos in this environment."
            ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        iio.imwrite(path, frames, fps=fps)
    except AttributeError:
        iio.mimsave(path, frames, fps=fps)


def save_latents_npz(
    *,
    rollout: RolloutData,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("NumPy is required to write latent trajectory files.") from exc

    if not rollout.features:
        raise ValueError("No latent features were collected.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(rollout.features, dtype=np.float32),
        deter=np.asarray(rollout.deter, dtype=np.float32),
        stoch=np.asarray(rollout.stoch, dtype=np.float32),
        actions=np.asarray(rollout.actions, dtype=np.float32),
        rewards=np.asarray(rollout.rewards, dtype=np.float32),
        episode=np.asarray(rollout.episode_index, dtype=np.int32),
        step=np.asarray(rollout.step_index, dtype=np.int32),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
