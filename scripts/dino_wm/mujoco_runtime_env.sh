#!/usr/bin/env bash

DINO_MUJOCO210_DIR="${DINO_MUJOCO210_DIR:-${MUJOCO210_DIR:-${HOME}/.mujoco/mujoco210}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-${DINO_MUJOCO210_DIR}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

wm_poc_prepend_path() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" || ! -d "${value}" ]]; then
    return
  fi
  local current="${!name:-}"
  case ":${current}:" in
    *":${value}:"*) ;;
    *) export "${name}=${value}${current:+:${current}}" ;;
  esac
}

wm_poc_enable_dino_wm_imports() {
  # Make upstream DINO-WM importable in launched subprocesses. Correctness rests
  # on the PYTHONPATH prepend of the repo itself (path ordering favours the
  # upstream top-level packages); the python_startup dir only adds the
  # sitecustomize shim, which delegates to
  # wm_poc.dino_wm.import_bootstrap.enable_dino_wm_imports and is not required.
  local upstream_repo="$1"
  export DINO_WM_REPO="${upstream_repo}"
  wm_poc_prepend_path PYTHONPATH "${upstream_repo}"
  wm_poc_prepend_path PYTHONPATH "$(pwd)/scripts/dino_wm/python_startup"
}

wm_poc_prepend_path LD_LIBRARY_PATH "${DINO_MUJOCO210_DIR}/bin"
wm_poc_prepend_path LD_LIBRARY_PATH "/usr/lib/nvidia"

if [[ "${MUJOCO_GL}" == "egl" || "${MUJOCO_GL}" == "osmesa" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

wm_poc_add_python_warning_filter() {
  local filter="$1"
  case ",${PYTHONWARNINGS:-}," in
    *",${filter},"*) ;;
    ",,") export PYTHONWARNINGS="${filter}" ;;
    *) export PYTHONWARNINGS="${filter},${PYTHONWARNINGS}" ;;
  esac
}

wm_poc_add_python_warning_filter "ignore::Warning:wandb.analytics.sentry"
wm_poc_add_python_warning_filter "ignore:.*sentry_sdk\\.Hub is deprecated.*"
wm_poc_add_python_warning_filter "ignore:Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*"
