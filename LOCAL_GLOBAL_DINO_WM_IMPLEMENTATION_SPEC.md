# Local/Global World-Model Planning Implementation Spec

This document is an implementation brief for an agent that will add the local/global world-model experiment track to the existing `wm-prediction` repository. It should follow the same user-facing notebook logic already used by the DINO-WM track: one notebook that can train/test or dispatch experiment commands, and one separate notebook that only visualizes and summarizes results.

The implementation target is a practical proof-of-concept, not a full reproduction of the diffusion-based local/global paper. In this repository, the first global model should be the already integrated DINO-WM latent world model, while the local model should be a smaller differentiable surrogate trained on DINO/DINO-WM latents. The goal is to compare:

1. a global DINO-WM CEM/MPC planner,
2. a local differentiable planner that backpropagates through the small surrogate,
3. a hybrid planner that uses global search for robust candidate generation and local gradients for refinement,
4. optional ablations such as local-only CEM, global-only open-loop CEM, local Adam vs. local GD, and global re-scoring after local refinement.

Do not run long training jobs by default. Every notebook and script must have a dry-run or smoke-test path that completes quickly on CPU or a small GPU runtime.

## 1. Source concept to implement

The local/global idea is to decouple the model used for forward imagination from the model used for gradients. The global model is larger and more accurate for forward trajectories; the local model is lightweight and differentiable, supplying tractable local derivatives. The paper frames this as a decoupled first-order model-based RL setup where a high-fidelity forward model generates trajectories and a lightweight backward/local model supplies gradients for efficient policy or action optimization.

For this repository, implement the same concept in a simpler visual-goal planning setting:

- **Global model:** the DINO-WM transition model already used by the repository, operating in frozen DINOv2 patch-feature space. It is the trusted scorer/simulator for candidate action sequences.
- **Local model:** a small residual MLP or GRU/RSSM-like surrogate operating on compressed latent states. It is trained from cached DINO/DINO-WM transitions and must support gradients with respect to action sequences.
- **Hybrid planner:** global model proposes or re-scores, local model refines by gradient descent/Adam, then global model optionally re-scores the refined candidates before execution.

The DINO-WM source implementation is especially relevant because it trains the transition model entirely in latent space using teacher-forced latent consistency, keeps the optional decoder independent for visualization, and uses visual goal reaching with a CEM/MPC objective based on the latent distance between predicted future state and goal state. Keep those design choices: do not make pixel reconstruction part of the local surrogate’s training objective in the first implementation.


### 1.1 Source anchors for the implementing agent

Use these uploaded papers as the conceptual source material when making implementation decisions:

- `08_2026_coupled_local_global_world_models_fog.pdf`: source for the local/global decoupling. The global model is the accurate forward simulator; the local/backward model is the lightweight differentiable surrogate used for gradients.
- `09_2025_dino_wm_zero_shot_planning.pdf`: source for the DINO-WM global model pattern. It uses frozen DINO patch features, trains an action-conditioned latent transition model without pixel reconstruction, and plans to an image goal by minimizing final predicted latent distance to the goal latent.
- `19_2026_jepa_wms_physical_planning.pdf`: source for the broader JEPA-WM framing, notation, and practical design choices around frozen visual encoders, action encoders, rollout losses, context length, CEM-style planning, and results reporting.

## 2. Repository conventions to preserve

Follow the conventions already visible in the attached repository.

The DINO-WM train/test notebook is `notebooks/02_dino_wm_foundation.ipynb`. It is operational: setup, verification, dataset/latent-cache preparation, command construction, smoke training, full training dispatch, planning evaluation, and artifact links. The local/global train/test notebook should mirror this structure.

The DINO-WM results notebook is `notebooks/07_dino_wm_results.ipynb`. It is post-hoc and CPU-friendly: discover run directories, aggregate logs, plot curves, render tables, display videos and artifact paths. The local/global results notebook should mirror this structure and must not start training or planning.

The repository already contains placeholder local/global files:

- `notebooks/03_local_global_foundation.ipynb`
- `configs/local_global/README.md`
- `configs/local_global/pointmaze_local_surrogate_example.yaml`
- `src/wm_poc/local_global/models.py`
- `src/wm_poc/local_global/datasets.py`
- `src/wm_poc/local_global/losses.py`
- `src/wm_poc/local_global/planners.py`
- `src/wm_poc/local_global/eval.py`

Replace the placeholders incrementally rather than creating a parallel codebase. Keep paths configurable through environment variables and YAML configs. Do not commit checkpoints, videos, latent caches, TensorBoard logs, large datasets, or downloaded upstream repositories.

## 3. Deliverable notebooks

### 3.1 `notebooks/03_local_global_foundation.ipynb`

This notebook should be the train/test and experiment-dispatch notebook for the local/global track. It should have the same operational style as `02_dino_wm_foundation.ipynb`.

Required cells, in order:

1. **Title and purpose.** Explain that this notebook trains and evaluates local/global DINO-WM planning, with DINO-WM as the global model and a small differentiable surrogate as the local model.
2. **Runtime and safety flags.** Define flags such as `RUN_SMOKE`, `RUN_FULL_TRAIN`, `RUN_PLANNING`, `DRY_RUN`, `USE_DRIVE`, `TASK`, `CONFIG_NAME`, `RUN_NAME`. Default to `DRY_RUN=True` and no full training.
3. **Repository setup.** Locate repository root, install editable package if needed, set `PYTHONPATH`, print commit/hash if available.
4. **Environment verification.** Call existing checks first: `python scripts/verify_environment.py --cpu-only`, `python scripts/verify_drive_layout.py --dry-run`, and `pytest -q` only when explicitly requested.
5. **DINO-WM dependency verification.** Confirm the DINO-WM upstream integration and any required checkpoint/config paths are present. Do not download large files automatically.
6. **Config selection and display.** Load `configs/local_global/<config>.yaml`, resolve environment variables, print paths and important hyperparameters.
7. **Latent cache verification.** Verify DINOv2/DINO-WM latent caches exist for the selected task. If missing, print the exact command to create them using the existing DINO-WM latent precompute script. Only run the command if the user sets an explicit run flag.
8. **Transition export.** Build a local transition dataset from the latent cache and corresponding action sequences. This should create a lightweight manifest and optional shard files under the run directory or Drive artifact directory.
9. **Local model smoke test.** Instantiate the local surrogate, run one batch through it, assert shapes, finite loss, and nonzero gradient with respect to actions.
10. **Local model training.** Provide command cells for `scripts/local_global/train_local_surrogate.py`. Include dry-run and smoke configurations first. Full training should be gated.
11. **Validation rollouts.** Run or dispatch validation for one-step and multi-step rollout metrics. Log local-vs-target MSE and, when available, local-vs-global disagreement.
12. **Planner smoke test.** Run a toy or tiny real episode that checks all planners return valid bounded actions and that the local planner’s optimization cost decreases over iterations.
13. **Planning evaluation.** Dispatch comparison runs for `global_cem`, `local_gd`, `local_adam`, `hybrid_cem_local_refine`, and optional `hybrid_cem_local_refine_global_rescore`.
14. **Summary table.** Load `summary.csv` if available and display success rate, final latent distance, wall time, model calls, gradient steps, and artifact paths.
15. **Next commands.** Print copy-paste commands for full PointMaze and PushT runs.

Notebook behavior:

- It must be runnable top-to-bottom in dry-run mode without data downloads and without long training.
- It must make heavy steps explicit and opt-in.
- It should not contain substantial model code; it should call scripts and import library modules.
- It should write all outputs under `runs/local_global/<run_name>/` or a configurable Drive directory.

### 3.2 `notebooks/08_local_global_results.ipynb`

Create this new notebook for visualization and reporting. It should never launch training, latent precomputation, or planning. It should only read artifacts.

Required cells, in order:

1. **Title and scope.** Explain that this notebook aggregates local/global experiments.
2. **Path selection.** Let the user set `RUN_ROOT`, `TASK`, `RUN_GLOB`, and `SUMMARY_OUT`.
3. **Run discovery.** Recursively discover run directories and planning subdirectories.
4. **Summary aggregation.** Call or import `scripts/local_global/summarize_runs.py` to build `_summary/summary.csv`.
5. **Local training curves.** Plot train/validation loss, one-step MSE, multi-step rollout MSE, and optional local-vs-global disagreement.
6. **Planner comparison table.** Display success rate, final distance, time per episode, forward calls, backward calls, and action-bound violation count by planner.
7. **Planner optimization traces.** Plot CEM best/mean cost and local GD/Adam cost vs. iteration for representative episodes.
8. **Qualitative videos.** Show executed environment videos, global imagined rollouts, local imagined rollouts, and side-by-side comparisons when available.
9. **Failure cases.** List episodes where local refinement improved the global plan, worsened it, or was rejected by global re-scoring.
10. **Recommendation cell.** Summarize which planner should be used next for the task and why.

Notebook behavior:

- CPU-only.
- Robust to missing artifacts.
- Produces useful placeholder tables/plots even when only smoke runs exist.
- Saves generated figures to `runs/local_global/_summary/figures/`.

## 4. Code modules to implement

### 4.1 `src/wm_poc/local_global/configs.py`

Add a typed config loader using dataclasses or Pydantic-style validation. It should load YAML, expand environment variables, and normalize paths.

Suggested structure:

```python
@dataclass
class GlobalModelConfig:
    source: str                 # "dino_wm"
    checkpoint_path: str | None
    latent_cache_dir: str
    encoder_name: str           # "dinov2_vits14"
    latent_shape: tuple[int, int]  # e.g. (196, 384)
    frameskip: int

@dataclass
class LocalModelConfig:
    model_type: str             # "residual_mlp", "gru_residual", later "rssm"
    projection: str             # "mean_pool_linear", "pca", "flatten_linear"
    local_dim: int
    hidden_dim: int
    num_layers: int
    context_len: int
    rollout_steps: int

@dataclass
class PlannerConfig:
    action_dim: int
    action_low: list[float]
    action_high: list[float]
    horizon: int
    mpc_exec_steps: int
    cem_population: int
    cem_elites: int
    cem_iters: int
    gd_iters: int
    gd_lr: float
    gradient_clip: float | None
    action_smoothness: float
    global_rescore: bool
```

The loader should expose `load_local_global_config(path: str | Path) -> LocalGlobalConfig` and `resolve_run_dir(config) -> Path`.

### 4.2 `src/wm_poc/local_global/datasets.py`

Implement datasets that read cached DINO latents and action arrays.

Minimum required classes:

- `LatentTrajectoryStore`: loads a manifest and exposes episode-level latent/action arrays.
- `LatentTransitionDataset`: yields one-step transitions for local model training.
- `LatentWindowDataset`: yields context windows and K-step rollout targets.
- `collate_latent_windows`: pads or stacks windows and returns tensors.

Expected sample dictionary:

```python
{
    "z_context": Tensor[context_len, patches, embed_dim],
    "x_context": Tensor[context_len, local_dim],        # after projection if precomputed
    "actions": Tensor[rollout_steps, action_dim],
    "z_targets": Tensor[rollout_steps, patches, embed_dim],
    "x_targets": Tensor[rollout_steps, local_dim],
    "episode_id": str,
    "start_t": int,
}
```

Implementation details:

- Use memory mapping where possible for `.npy` episode files.
- Validate that action length aligns with latent length.
- Split by episode, not by random windows within the same episode, to avoid validation leakage.
- Support `max_episodes` and `max_windows` for smoke tests.
- Include a tiny synthetic dataset mode for CI if real latent caches are unavailable.

### 4.3 `src/wm_poc/local_global/models.py`

The existing `LocalDynamics` residual MLP can be kept but should be expanded.

Required components:

1. `PatchProjector`
   - Converts global patch latents `[B, P, D]` to local states `[B, local_dim]`.
   - Initial implementation can use mean pooling plus a learned linear layer.
   - Optional later: PCA fitted offline and saved as `projection_stats.pt`.

2. `LocalDynamics`
   - Residual transition model: `x_{t+1} = x_t + f_theta(x_t, a_t)`.
   - Input: current local state and action.
   - Output: next local state.
   - Must be fully differentiable with respect to actions.

3. `ContextLocalDynamics`
   - Optional GRU variant using `context_len > 1`.
   - Useful for velocity/momentum inference.

4. `LocalRolloutModel`
   - Wraps projector and dynamics.
   - Provides `encode_global_latent(z)`, `step(x, a)`, and `rollout(x0, actions)`.

Recommended first-pass architecture:

```text
global z [196,384]
   -> mean pool over patches [384]
   -> LayerNorm
   -> Linear(384, local_dim)
   -> residual MLP dynamics in local_dim
```

This is intentionally simple. Add more complex patch attention only after the smoke and PointMaze baselines work.

### 4.4 `src/wm_poc/local_global/losses.py`

Implement:

- `one_step_mse(pred, target)`
- `rollout_mse(pred_seq, target_seq, discount=1.0)`
- `delta_mse(x_t, pred_next, target_next)`
- `action_smoothness(actions)`
- `jacobian_norm_penalty(model, x, a)` for optional regularization
- `combined_local_loss(batch, model, weights)` returning both scalar loss and a metrics dictionary.

Default objective:

```text
L = rollout_mse(x_pred[1:K], x_target[1:K])
  + lambda_one_step * one_step_mse(x_pred[1], x_target[1])
  + lambda_delta * delta_mse(...)
  + lambda_jacobian * jacobian_norm_penalty(...)
```

The first implementation should train in the compressed local space. Global-patch reconstruction or local-to-global decoding can be added later but should not block the initial track.

### 4.5 `src/wm_poc/local_global/planners.py`

Implement planners behind a common interface.

Required interface:

```python
@dataclass
class PlanResult:
    actions: torch.Tensor
    costs: dict[str, float]
    trace: list[dict[str, float]]
    planner_name: str
    metadata: dict[str, Any]

class BasePlanner(Protocol):
    def plan(self, current_obs_or_latent, goal_obs_or_latent, context=None) -> PlanResult: ...
```

Planners:

1. `GlobalCEMPlanner`
   - Uses the DINO-WM global model in `torch.no_grad()`.
   - Samples action sequences, unrolls global latent model, scores final latent distance to goal, updates CEM distribution.
   - Should be compatible with the existing DINO-WM planner if one already exists. Prefer wrapping existing code over duplicating it.

2. `LocalGradientPlanner`
   - Encodes current and goal global latents into local states.
   - Optimizes an action tensor through the local surrogate using GD or Adam.
   - Uses bounded action parameterization, preferably tanh/sigmoid mapping rather than hard clipping inside the gradient path.

Pseudocode:

```python
a_raw = torch.zeros(H, A, device=device, requires_grad=True)
opt = torch.optim.Adam([a_raw], lr=cfg.gd_lr)
for j in range(cfg.gd_iters):
    actions = squash_to_bounds(a_raw, low, high)
    x_seq = local_model.rollout(x0, actions)
    cost = mse(x_seq[-1], x_goal) + cfg.action_smoothness * smoothness(actions)
    opt.zero_grad(set_to_none=True)
    cost.backward()
    if cfg.gradient_clip is not None:
        torch.nn.utils.clip_grad_norm_([a_raw], cfg.gradient_clip)
    opt.step()
    trace.append({"iter": j, "cost": float(cost.detach())})
```

3. `HybridCEMLocalRefinePlanner`
   - Run CEM with the global model for a small number of iterations or use the best sequence from `GlobalCEMPlanner`.
   - Initialize local planner actions from the best global sequence.
   - Refine with local gradients.
   - Re-score the refined sequence with the global model before returning.
   - If global re-score is worse than the original CEM sequence by a configurable tolerance, reject the refinement and return the original sequence.

4. Optional `LocalCEMPlanner`
   - Useful to separate “small model effect” from “gradient effect.”

Logging requirements:

- Save optimization traces as JSONL.
- Log cost components separately: `goal_cost`, `smoothness_cost`, `jacobian_cost`, `global_rescore_cost`, `accepted_refinement`.
- Log `num_global_forward_calls`, `num_local_forward_calls`, `num_backward_steps`, and wall-clock time.

### 4.6 `src/wm_poc/local_global/eval.py`

Implement MPC evaluation that can run planners in the real environment or, for smoke tests, in a toy environment.

Required functions:

- `evaluate_planner(config, planner_name, run_dir)`
- `run_mpc_episode(env, planner, goal, max_steps, exec_steps)`
- `compute_episode_metrics(trajectory, goal)`
- `save_episode_artifacts(...)`

Episode loop:

1. Reset environment and sample/set goal observation.
2. Encode current and goal observations with the frozen DINO encoder/global model interface.
3. Plan horizon `H` actions.
4. Execute the first `m` actions.
5. Repeat until success or `max_env_steps`.
6. Save logs and videos.

Metrics:

- `success`
- `final_latent_distance_global`
- `final_latent_distance_local`
- `environment_reward` if available
- `episode_steps`
- `planning_wall_time_sec`
- `mean_plan_cost_first_iter`
- `mean_plan_cost_final_iter`
- `accepted_refinement_rate`
- `action_bound_violation_count`

### 4.7 `src/wm_poc/local_global/visualization.py`

Add reusable functions for the results notebook:

- `load_train_metrics(run_dir)`
- `load_planning_logs(run_dir)`
- `aggregate_summary(run_root)`
- `plot_training_curves(df)`
- `plot_rollout_errors(df)`
- `plot_planner_bars(summary_df)`
- `plot_optimization_trace(trace_df)`
- `find_videos(run_dir)`

Use Matplotlib and Pandas. Do not assume videos always exist.

## 5. Scripts to add

Create a `scripts/local_global/` directory with these commands.

### 5.1 `scripts/local_global/export_transitions.py`

Purpose: convert a DINO latent cache and action source into an indexed local/global transition dataset.

Required CLI:

```bash
python scripts/local_global/export_transitions.py \
  --config configs/local_global/pointmaze_surrogate_t4.yaml \
  --split train \
  --out runs/local_global/<run_name>/transition_data \
  --dry-run
```

Outputs:

- `manifest.json`
- optional shard files: `train_windows.npz`, `val_windows.npz`, or per-episode index JSONL
- `dataset_stats.json`

### 5.2 `scripts/local_global/train_local_surrogate.py`

Purpose: train the local dynamics model.

Required CLI:

```bash
python scripts/local_global/train_local_surrogate.py \
  --config configs/local_global/pointmaze_surrogate_t4.yaml \
  --run-dir runs/local_global/<run_name> \
  --max-steps 1000 \
  --smoke
```

Outputs:

- `checkpoints/local_latest.pt`
- `checkpoints/local_best.pt`
- `metrics/train_metrics.jsonl`
- `metrics/val_rollouts.jsonl`
- `config_resolved.yaml`

### 5.3 `scripts/local_global/run_planning_eval.py`

Purpose: evaluate one or more planners.

Required CLI:

```bash
python scripts/local_global/run_planning_eval.py \
  --config configs/local_global/pointmaze_surrogate_t4.yaml \
  --run-dir runs/local_global/<run_name> \
  --planners global_cem local_adam hybrid_cem_local_refine \
  --num-episodes 10 \
  --smoke
```

Outputs:

- `planning/<planner>/episodes.jsonl`
- `planning/<planner>/summary.json`
- `planning/<planner>/traces/*.jsonl`
- optional videos under `planning/<planner>/videos/`

### 5.4 `scripts/local_global/summarize_runs.py`

Purpose: aggregate training and planning artifacts across runs.

Required CLI:

```bash
python scripts/local_global/summarize_runs.py \
  --run-root runs/local_global \
  --out runs/local_global/_summary/summary.csv
```

### 5.5 `scripts/local_global/run_smoke.sh`

Purpose: one small end-to-end test.

The smoke should:

1. use a tiny synthetic or tiny cached latent dataset,
2. export transitions,
3. train for a very small number of steps,
4. run planner smoke tests,
5. produce a summary CSV.

Target runtime: under 2 minutes on CPU if using synthetic data, under 5 minutes on a small GPU if using real latents.

## 6. Config files to add

### 6.1 `configs/local_global/base.yaml`

Use this as a complete schema example. Include comments and defaults.

```yaml
track: local_global
run_name: local_global_${task}_${now}
seed: 0
device: cuda

task: pointmaze
paths:
  run_root: ${WM_RUN_ROOT:-runs/local_global}
  dino_wm_repo: ${DINO_WM_REPO:-external/dino_wm}
  data_root: ${WM_DATA_ROOT:-data}
  latent_cache_dir: ${WM_LATENT_CACHE_DIR:-data/latents/point_maze/dinov2_vits14_img224}
  action_data_path: ${WM_ACTION_DATA_PATH:-data/actions/point_maze_actions.npz}
  global_checkpoint_path: ${DINO_WM_CKPT:-}

global_model:
  source: dino_wm
  encoder: dinov2_vits14
  image_size: 224
  latent_patches: 196
  latent_dim: 384
  frameskip: 5
  use_checkpoint: true

local_model:
  model_type: residual_mlp
  projection: mean_pool_linear
  local_dim: 256
  hidden_dim: 512
  num_layers: 3
  context_len: 2
  rollout_steps: 3
  layer_norm: true

training:
  batch_size: 128
  max_steps: 20000
  lr: 0.0003
  weight_decay: 0.0001
  val_every: 500
  save_every: 1000
  lambda_one_step: 1.0
  lambda_delta: 0.1
  lambda_jacobian: 0.0
  max_grad_norm: 10.0

planning:
  action_dim: 2
  action_low: [-1.0, -1.0]
  action_high: [1.0, 1.0]
  horizon: 6
  mpc_exec_steps: 1
  max_env_steps: 50
  cem_population: 300
  cem_elites: 10
  cem_iters: 5
  gd_iters: 100
  gd_lr: 0.05
  gradient_clip: 10.0
  action_smoothness: 0.01
  global_rescore: true
  reject_refine_if_worse_by: 0.05

evaluation:
  num_episodes: 50
  save_video: true
  save_imagined_rollouts: true
  success_threshold: null
```

### 6.2 `configs/local_global/smoke_pointmaze.yaml`

Override base config with tiny settings:

```yaml
inherits: configs/local_global/base.yaml
run_name: smoke_local_global_pointmaze
device: cpu
training:
  batch_size: 8
  max_steps: 10
  val_every: 5
planning:
  cem_population: 16
  cem_elites: 4
  cem_iters: 2
  gd_iters: 5
evaluation:
  num_episodes: 1
  save_video: false
smoke:
  use_synthetic_latents_if_missing: true
  max_episodes: 2
  max_windows: 32
```

### 6.3 `configs/local_global/pointmaze_surrogate_t4.yaml`

T4-friendly PointMaze config. Keep DINO latent shapes and action bounds aligned with the repository’s DINO-WM PointMaze setup.

### 6.4 `configs/local_global/pusht_surrogate_a100.yaml`

Optional second-stage config. Do not prioritize PushT until PointMaze passes smoke and at least one real latent-cache run.

## 7. Data and artifact layout

Use this layout:

```text
runs/local_global/
  <run_name>/
    config_resolved.yaml
    transition_data/
      manifest.json
      dataset_stats.json
    checkpoints/
      local_latest.pt
      local_best.pt
    metrics/
      train_metrics.jsonl
      val_rollouts.jsonl
    planning/
      global_cem/
        summary.json
        episodes.jsonl
        traces/
        videos/
      local_adam/
        summary.json
        episodes.jsonl
        traces/
        videos/
      hybrid_cem_local_refine/
        summary.json
        episodes.jsonl
        traces/
        videos/
    figures/
  _summary/
    summary.csv
    figures/
```

Do not place these artifacts inside the committed repository unless they are tiny smoke artifacts explicitly intended for tests. Large outputs should be ignored by Git and stored in Drive when running on Colab.

## 8. Implementation details that matter

### 8.1 Keep DINO-WM as the source of global truth

The local surrogate should be evaluated against true cached future latents and, when a global checkpoint is available, against global model rollouts. The hybrid planner should not blindly trust the local gradient refinement. It must optionally re-score the refined action sequence using the global model and reject the refinement if it worsens the global cost.

### 8.2 Use context to infer velocity, but keep it small

Support `context_len > 1` in the dataset and local model. The first PointMaze implementation can start with context length 2. Avoid very long contexts in the first pass, because they increase data loading complexity and reduce the number of training windows.

### 8.3 Avoid local-model exploitation

The local model is intentionally smaller and less accurate. Add at least three safeguards:

1. action bounds through a differentiable squashing function,
2. action smoothness penalty,
3. global re-scoring and refinement rejection in the hybrid planner.

Optional later safeguards:

- uncertainty ensembles for the local model,
- disagreement penalty against global rollout,
- CEM elites constrained to dataset-like action magnitudes,
- rollout truncation if local state leaves the training latent distribution.

### 8.4 Make planner comparison fair

Log compute usage. A local gradient planner may use fewer global calls but many backward passes; global CEM may use many global forward calls but no backward pass. The results notebook should compare success and final distance alongside wall-clock time and call counts.

### 8.5 Keep visualization decoupled

The local model does not need a decoder. Use existing DINO-WM decoder or visualization utilities only for qualitative rollouts if available. If no decoder is available, visualize latent distances and environment videos.

## 9. Tests and acceptance criteria

Add tests under `tests/local_global/`.

Required tests:

1. **Config test:** YAML loads, environment variables expand, defaults resolve.
2. **Dataset test:** synthetic latent store yields correctly shaped samples and deterministic train/val split.
3. **Model test:** `LocalRolloutModel.rollout()` returns `[B, H, local_dim]` and supports gradients through actions.
4. **Loss test:** combined loss is finite and metrics dict contains expected keys.
5. **Planner test:** local gradient planner reduces cost on a toy linear dynamics problem.
6. **Hybrid test:** global re-score rejection returns original action sequence when refinement is worse.
7. **Summary test:** summarizer handles missing videos, missing planner directories, and empty runs without crashing.

Acceptance criteria:

- `python scripts/local_global/run_smoke.sh` completes with synthetic latents.
- `pytest -q tests/local_global` passes.
- `notebooks/03_local_global_foundation.ipynb` runs top-to-bottom in dry-run mode.
- `notebooks/08_local_global_results.ipynb` runs top-to-bottom on smoke artifacts.
- At least one real-latent PointMaze run can train a local surrogate and evaluate the three main planners.
- All heavy operations are opt-in.
- No large artifacts are committed.

## 10. Suggested implementation sequence

### Phase 0: Spec and scaffolding

- Add this spec to the repository root as `LOCAL_GLOBAL_DINO_WM_IMPLEMENTATION_SPEC.md`.
- Expand `configs/local_global/README.md` with a short pointer to this spec.
- Add `configs/local_global/base.yaml` and `configs/local_global/smoke_pointmaze.yaml`.

### Phase 1: Data and local model

- Implement config loader.
- Implement synthetic latent dataset.
- Implement real latent-cache dataset reader.
- Expand `LocalDynamics` into `PatchProjector`, `LocalDynamics`, and `LocalRolloutModel`.
- Implement training loss and `train_local_surrogate.py`.
- Add tests for config, dataset, model, and losses.

### Phase 2: Planners

- Implement local GD/Adam planner first.
- Add toy planner test.
- Implement global CEM wrapper using existing DINO-WM code.
- Implement hybrid CEM + local refinement + global re-score.
- Add planner logs and trace output.

### Phase 3: Evaluation

- Implement `run_planning_eval.py`.
- Save uniform planner outputs under `planning/<planner>/`.
- Implement summarizer and visualization helpers.

### Phase 4: Notebooks

- Fill `notebooks/03_local_global_foundation.ipynb` with the operational cells described above.
- Add `notebooks/08_local_global_results.ipynb` with read-only visualization cells.
- Make sure both notebooks include a dry-run path and point to exact scripts.

### Phase 5: PushT extension

- Add PushT config only after PointMaze works.
- Verify action dimension, bounds, frameskip, and success metrics.
- Expect contact dynamics to expose local model errors; prioritize hybrid with global re-score.

## 11. Concrete command examples

Dry-run setup:

```bash
python scripts/local_global/export_transitions.py \
  --config configs/local_global/smoke_pointmaze.yaml \
  --dry-run
```

Synthetic smoke:

```bash
bash scripts/local_global/run_smoke.sh
```

Train local surrogate on real PointMaze latents:

```bash
python scripts/local_global/export_transitions.py \
  --config configs/local_global/pointmaze_surrogate_t4.yaml \
  --split train

python scripts/local_global/train_local_surrogate.py \
  --config configs/local_global/pointmaze_surrogate_t4.yaml \
  --run-dir runs/local_global/pointmaze_local_v1
```

Evaluate planners:

```bash
python scripts/local_global/run_planning_eval.py \
  --config configs/local_global/pointmaze_surrogate_t4.yaml \
  --run-dir runs/local_global/pointmaze_local_v1 \
  --planners global_cem local_adam hybrid_cem_local_refine hybrid_cem_local_refine_global_rescore \
  --num-episodes 50
```

Summarize:

```bash
python scripts/local_global/summarize_runs.py \
  --run-root runs/local_global \
  --out runs/local_global/_summary/summary.csv
```

## 12. What not to implement in the first pass

Do not implement a new large diffusion world model. Use DINO-WM as the global model first.

Do not backpropagate through the global DINO-WM model or DINO encoder in the initial implementation.

Do not add pixel reconstruction to the local model training objective.

Do not require reward labels for the first PointMaze visual-goal planning implementation.

Do not make PushT a blocker for merging the local/global track.

Do not put Drive-specific absolute paths into committed configs; use environment variables.

## 13. Final expected repository diff

Minimum useful diff:

```text
LOCAL_GLOBAL_DINO_WM_IMPLEMENTATION_SPEC.md
configs/local_global/base.yaml
configs/local_global/smoke_pointmaze.yaml
configs/local_global/pointmaze_surrogate_t4.yaml
notebooks/03_local_global_foundation.ipynb
notebooks/08_local_global_results.ipynb
scripts/local_global/export_transitions.py
scripts/local_global/train_local_surrogate.py
scripts/local_global/run_planning_eval.py
scripts/local_global/summarize_runs.py
scripts/local_global/run_smoke.sh
src/wm_poc/local_global/configs.py
src/wm_poc/local_global/datasets.py
src/wm_poc/local_global/models.py
src/wm_poc/local_global/losses.py
src/wm_poc/local_global/planners.py
src/wm_poc/local_global/eval.py
src/wm_poc/local_global/visualization.py
tests/local_global/test_configs.py
tests/local_global/test_datasets.py
tests/local_global/test_models.py
tests/local_global/test_planners.py
tests/local_global/test_summarize.py
```

## 14. One-sentence objective for the implementing agent

Implement a DINO-WM-based local/global planning track where the global latent world model provides accurate CEM/MPC rollouts, a small local latent surrogate provides differentiable action gradients, a hybrid planner combines both, and the results are exposed through the same two-notebook workflow used by the existing DINO-WM implementation.
