# Local/Global Planning: Methodology, Implementation, and the Link to DINO-WM

This document gives the full methodology of the local/global planning track,
the implementation that realizes it, and — the part this document foregrounds —
exactly how it connects to and reuses the repository's existing DINO-WM
implementation. It is the conceptual counterpart to two siblings:
`docs/local_global_techniques.md` (a catalogue of individual techniques) and
`docs/local_global_runbook_next_steps.md` (the operational runbook). Where those
overlap with this one, this document cross-references rather than repeats.

The one-sentence thesis: **decouple the model that imagines the future from the
model that supplies gradients** — use the trusted DINO-WM latent world model as
the forward simulator/scorer, and a small differentiable surrogate trained on
the same latents as the source of action gradients — then build planners that
span and combine the two.

---

## 1. The starting point: DINO-WM as already implemented in this repo

The local/global track does not reimplement DINO-WM; it stands on top of the
existing reproduction. Understanding the linkage requires first stating what
that reproduction is, concretely, in this repository.

### 1.1 What the repo's DINO-WM track produces

The DINO-WM track (notebook `02_dino_wm_foundation.ipynb`, `scripts/dino_wm/`,
`src/wm_poc/dino_wm/`) is a command-builder/patcher/log-parser wrapper around an
upstream `gaoyuezhou/dino_wm` checkout living on Google Drive. It never vendors
the upstream code; it applies marker-guarded textual patches at runtime and
drives upstream `train.py`/`plan.py` through Hydra CLI overrides. It produces
three artifacts the local/global track consumes:

1. **A frozen DINOv2 patch-latent cache.** A precompute step encodes each
   PointMaze episode through frozen DINOv2 (`dinov2_vits14`, image size 224) and
   writes, under `<feature_cache>/<env>/<encoder>_img<size>/`:
   - `episode_{i:03d}.npy` — shape `(T, P, D) = (T, 196, 384)`, dtype **float16**,
     written atomically;
   - `wm_poc_latent_manifest.json` — `{"format": "wm_poc_dino_latents_v1",
     encoder_name, feature_key="x_norm_patchtokens", img_size, num_patches=196,
     emb_dim=384, dtype, num_episodes, dataset_episodes, episode_lengths, ...}`.
2. **A trained latent transition model (the "world model").** Upstream
   `VWorldModel`: a frozen DINOv2 encoder + an action/proprio-conditioned ViT
   *predictor* over patch tokens, trained with teacher-forced latent
   consistency and **no pixel-reconstruction term in the planning path**
   (the decoder is optional and independent). Checkpoints live under
   `<ckpt_root>/outputs/<run_name>/checkpoints/model_<epoch>.pth` alongside the
   run's `hydra.yaml`.
3. **A planning recipe.** Upstream `plan.py` does visual goal reaching: encode
   current and goal images, run CEM/MPC over action sequences, score by the
   latent distance between the predicted final state and the goal latent.

### 1.2 The conventions the local/global track must inherit

Three upstream conventions are load-bearing for the linkage and are reproduced
exactly (not approximated) by the new track:

- **Frameskip action folding.** One model "step" advances `frameskip` raw
  environment actions. Latent frames are sampled at stride `frameskip`, while
  actions are taken densely and folded: `rearrange(act, "(n f) d -> n (f d)")`,
  so each model-step action has dimension `action_dim × frameskip`
  (PointMaze: `2 × 5 = 10`). Getting this wrong silently misaligns actions and
  latents.
- **Action/proprio normalization.** Checkpoints trained with upstream
  `normalize_action: true` consumed `(x − mean)/std` actions and proprio, with
  statistics computed from the raw dataset (`PointMazeDataset.get_data_mean_std`).
- **Checkpoint loading layout.** `plan.py` never takes a checkpoint path
  directly; it reconstructs the model from `outputs/<run>/hydra.yaml` plus
  `checkpoints/model_<epoch>.pth`. Any in-process reuse must follow the same
  layout. The rolling intra-epoch checkpoint `model_latest_step.pth` is a
  state-dict-only file that upstream loading rejects — it is **not** a loadable
  model.
- **The latent-bypass patch.** A marker-guarded patch
  (`WM_POC_DINO_LATENT_BYPASS_PATCH`) to upstream `VWorldModel.encode_obs` makes
  it accept 4-D cached latents `(b, t, P, D)` directly (bypassing the image
  transform + encoder) while leaving the 5-D image path intact. This is what
  lets anything drive the world model on cached latents instead of raw pixels.

### 1.3 Config conventions reused verbatim

The local/global config loader (`src/wm_poc/local_global/configs.py`) imports
the DINO-WM loader's primitives directly: single-parent `extends:` inheritance
with recursive deep-merge, and `${oc.env:VAR,default}` placeholders resolved in
a separate non-mutating pass. The standard call is
`resolve_config(load_config(path))` followed by track-specific validation. The
same environment variable (`DINO_WM_FEATURE_CACHE`) points both tracks at the
one cache.

---

## 2. Methodology

### 2.1 The decoupling principle

First-order (gradient) planning is sample-efficient but needs a differentiable
model, and small differentiable models are exploitable. Zero-order (sampling)
planning works with any forward model but scales poorly with horizon and action
dimension. The coupled local/global idea (FOG/DMO) resolves this by assigning
the two roles to two models:

- a **global** model `G` — large, accurate, used only forward, never
  differentiated;
- a **local** model `L` — small, differentiable, supplying gradients evaluated
  *along trajectories the global model produced*.

The crucial consequence is that `L` only needs **single-step accuracy in the
neighbourhood of states `G` visits** — it never has to be globally accurate over
a full rollout, because the rollout is always `G`'s. In this repository the idea
is expressed at the level of *planners* (open-loop action optimization), not
policy learning.

### 2.2 Notation and problem

An episode is latents `z_0, …, z_{T-1}` (`z_t ∈ R^{P×D}`, the DINO-WM cache) with
raw actions `a_t ∈ R^A`. With frameskip `f`, the model-step action block is
`ā_k = [a_{t0+kf}; …; a_{t0+(k+1)f-1}] ∈ R^{fA}`. A **goal-reaching task** gives a
context window ending at `t_cur` and the latent `z_goal = z_{t_cur + G·f}` recorded
`G` model-steps later in the same episode; a planner must produce actions that
drive the (global) model from the context to `z_goal`.

### 2.3 The global model

`G` is the trusted simulator and scorer with three operations used by planners:

```
init_state(z_context, proprio_context, actions_context) -> opaque state
rollout_final(state, actions[B,H,fA])                   -> z_final[B,P,D]   (batched, no_grad)
advance(state, actions[m,fA])                           -> state' (+ per-step latents)
```

The goal cost of an action sequence is the mean-squared latent distance of the
predicted final latent to the goal,
`c_goal(ā) = ‖G_final(state, ā) − z_goal‖² / (P·D)`. This is *exactly the DINO-WM
planning objective* (final predicted latent vs goal latent), now exposed as a
reusable scoring function rather than buried in upstream `plan.py`.

### 2.4 The local surrogate

`L` compresses global latents and predicts forward in the compressed space:

- **Projection** `x = W_p · pool(LN(z)) ∈ R^d` (`d=256`), pool = patch-mean or a
  4×4 spatial grid average, `W_p` a **frozen seeded semi-orthogonal** matrix.
  Freezing is required: training MSE in projected space makes a *trainable*
  projector collapse to `W_p = 0`. (Full argument in techniques §4.1.)
- **Dynamics** residual: `x_{k+1} = x_k + g_θ([x_k; ā_k])` (MLP, or a GRU variant
  that ingests the context window to infer velocity). Rollouts unroll under
  autograd; horizons ≤ 6 make exact backprop-through-time cheap.

`L` is trained on cached transitions with a discounted multi-step rollout MSE
plus one-step, delta, and optional Jacobian/variance terms (Eq. in report §4.3;
code in `losses.py`). It is differentiable end to end with respect to `ā`.

### 2.5 The planner family

All planners optimize an open-loop sequence `ā_{1:H}` minimizing
`J(ā) = c_goal(ā) + λ_sm · smoothness(ā)`, but report `c_goal` and the smoothness
term separately so the regularizer cannot contaminate cross-planner distance
comparisons.

| Planner | Model used | Optimizer | Isolates |
| --- | --- | --- | --- |
| `global_cem` | G (forward) | CEM (zero-order) | the trusted baseline |
| `local_cem` | L (forward) | CEM | "small model" error (vs global_cem) |
| `local_gd` / `local_adam` | L (gradients) | GD / Adam | "gradients vs sampling" (vs local_cem) |
| `hybrid_cem_local_refine` | G proposes, L refines | CEM → first-order | does cheap refinement help? |
| `hybrid_…_global_rescore` | + G re-scores & gates | + rejection rule | how exploitable is L? |

- **CEM**: population sampled from a diagonal Gaussian (init std in units of
  half the action range), clamp to bounds, refit mean/std to elites with a std
  floor, keep the incumbent in the pool, return the best-ever candidate; a
  per-round seeded CPU generator makes it device-independent.
- **First-order**: actions parameterized through a differentiable tanh squash so
  iterates are always feasible; best-iterate returned; warm starts use a
  `1e-2` desaturation margin so refinement isn't frozen at the bounds.
- **Hybrid + re-score**: G re-scores the refined sequence on the *same*
  objective; the refinement is rejected (CEM sequence kept) iff
  `J_ref > J_cem + τ·max(|J_cem|, ε)` with `τ = 0.05`. This is the central
  safeguard: **the surrogate is never trusted to overrule the global model.**

### 2.6 The MPC loop and the local/global coupling in action

```
state = G.init_state(context)
for each replanning round (until G model-steps consumed):
    H' = min(horizon, steps_remaining)          # horizon shrinks toward the goal
    ā* = planner.plan(state, z_goal, H')         # G forward + L gradients, per planner
    state = G.advance(state, ā*[:exec_steps])    # execute on the GLOBAL model
    context += G's imagined latents              # surrogate re-anchors on G's belief
```

The re-anchoring is exactly the FOG insight: every `exec_steps`, the planning
context is refreshed with latents the *global* model imagined, so the surrogate's
gradients are always taken near `G`'s trajectory rather than drifting on its own
multi-step rollout.

### 2.7 Safeguards against surrogate exploitation

Three are active by default: (1) differentiable action bounds (the first-order
planner literally cannot leave the feasible set), (2) the action-smoothness
penalty, (3) global re-scoring with rejection (§2.5). The evaluation additionally
measures **local–global disagreement**: the executed action sequence replayed
open-loop through `L`, compared step-by-step in projected space against `G`'s
imagined trajectory.

---

## 3. Implementation: methodology → code

| Methodology element (§2) | Module | Key entry points |
| --- | --- | --- |
| Config + validation, typed views | `local_global/configs.py` | `load_local_global_config`, `typed_config`, `PlannerConfig.step_action_*` |
| Latent store, frameskip folding, windows, splits, synthetic task | `local_global/datasets.py` | `LatentTrajectoryStore`, `fold_actions`, `LatentWindowDataset`, `split_store_episodes`, `generate_synthetic_task` |
| Projector + residual/GRU dynamics + checkpoints | `local_global/models.py` | `PatchProjector`, `LocalDynamics`, `ContextLocalDynamics`, `LocalRolloutModel`, `save/load_local_checkpoint` |
| Training objective | `local_global/losses.py` | `combined_local_loss` (+ component fns) |
| Global model `G` (DINO-WM and synthetic) | `local_global/global_models.py` | `DinoWMGlobalModel`, `SyntheticPointGlobalModel`, `latent_goal_cost`, `build_global_model` |
| Planner family + CEM/first-order primitives | `local_global/planners.py` | `cem_optimize`, `gradient_optimize`, `*Planner`, `build_planner` |
| MPC loop + offline protocol + metrics | `local_global/eval.py` | `run_mpc_episode`, `evaluate_planner` |
| Post-hoc loading/plots | `local_global/visualization.py` | `aggregate_summary`, `plot_*` |
| CLI (gated on `RUN_LOCAL_GLOBAL=1`) | `scripts/local_global/` | `export_transitions`, `train_local_surrogate`, `run_planning_eval`, `summarize_runs`, `run_smoke.sh` |
| Notebooks | `notebooks/` | `03_local_global_foundation` (operational), `08_local_global_results` (read-only) |

Data and control flow (condensed; full diagram in techniques §2):

```
DINO-WM latent cache + actions ─► LatentTrajectoryStore ─► windows ─► train surrogate ─► local_*.pt
                                          │                                    │
                                          └─► sample goal tasks ─► run_mpc_episode ◄── G (DinoWM/synthetic) + planner(L)
                                                                        │
                                              planning/<planner>/{episodes.jsonl, summary.json, traces} ─► summarize ─► notebook 08
```

---

## 4. The link with DINO-WM, in detail

This is the section the rest of the document builds toward: precisely what the
local/global track reuses from DINO-WM, what it adds, and where the seams are.

### 4.1 What is reused unchanged

| DINO-WM asset | How the local/global track uses it | Where |
| --- | --- | --- |
| Latent cache (`episode_XXX.npy`, `wm_poc_latent_manifest.json`) | The surrogate's entire training signal and the planner's start/goal latents — read directly, mmap'd, fp16→fp32 per window. No second representation is created. | `datasets.LatentTrajectoryStore` |
| Upstream action/state tensors (`actions.pth`, `seq_lengths.pth`, `states.pth`) | Action blocks for training and for the dataset-action *reference replay*; proprio for the global model. | `datasets._load_action_array`, `compute_action_state_stats` |
| Trained `VWorldModel` checkpoint | The global model `G` itself, loaded in-process. | `global_models.DinoWMGlobalModel` |
| Latent-bypass patch | Lets `G` run `rollout`/`encode_obs` on cached latents end to end. | applied via `scripts/dino_wm/patch_latent_cache.py` before planning |
| Frameskip folding convention | Reproduced exactly so action/latent alignment matches the checkpoint's training. | `datasets.fold_actions`, `PlannerConfig.step_action_dim` |
| Normalization statistics | Recomputed identically and applied at the adapter boundary. | `compute_action_state_stats`, `DinoWMGlobalModel._normalize_*` |
| Config system (`extends`, `${oc.env:…}`) | Imported directly from `dino_wm.configs`. | `local_global/configs.py` |
| Planning objective (final-latent vs goal-latent distance) | Becomes `latent_goal_cost`, the shared scoring function for every global planner. | `global_models.latent_goal_cost` |
| Cache directory layout (`<root>/<env>/<encoder>_img<size>`) | Mirrored so one env var serves both tracks. | `configs.latent_cache_dir` |

### 4.2 How `G` is loaded (the in-process bridge)

`DinoWMGlobalModel` loads the checkpoint *the way upstream `plan.py` does*:
`OmegaConf.load(outputs/<run>/hydra.yaml)` then upstream `plan.load_model(...)`
with the upstream repo on `sys.path`. It then drives the world model on cached
latents — only possible because the bypass patch makes `encode_obs` accept 4-D
latent input. Three adapter techniques keep this faithful (details in techniques
§5.2):

- **replay-from-anchor**: the original observed context (latents + raw proprio +
  observed action blocks) is kept and replayed each call, satisfying the upstream
  "one block per context frame, then future blocks" contract and never
  fabricating proprio for imagined frames;
- **normalization at the boundary**: only the adapter sees normalized units;
- **chunked candidate scoring**: CEM populations are scored in
  `rollout_batch_size` chunks (provably identical results, only memory changes) —
  the mechanism that lets the identical experiment run on a T4.

A `wm=` injection hook bypasses the upstream load entirely so the adapter's
tensor plumbing is unit-testable without Colab.

### 4.3 What the local/global track adds (new, not in DINO-WM)

- a **compressed differentiable surrogate** of the latent dynamics and its
  training pipeline (DINO-WM has no small/differentiable model);
- **first-order and hybrid planners** with the global re-score safeguard
  (DINO-WM has only zero-order CEM/GD inside upstream `plan.py`);
- an **offline latent goal-reaching evaluation** with fairness/compute accounting
  and the local–global disagreement metric;
- the **synthetic exact-global task** for CPU correctness testing;
- **resumable, GPU-tier-agnostic** training/evaluation orchestration.

### 4.4 Where DINO-WM and the surrogate diverge by design

- **Representation**: DINO-WM predicts in full patch space `R^{196×384}`; the
  surrogate predicts in `R^{256}`. The projection is the bridge, and it is frozen
  precisely so distances are preserved without collapse.
- **Accuracy expectation**: DINO-WM is expected to be globally accurate over a
  rollout; the surrogate is *only* expected to be locally accurate around `G`'s
  trajectory — which is why the MPC loop re-anchors and why the hybrid re-scores.
- **Gradients**: DINO-WM is never differentiated here (`no_grad` throughout);
  the surrogate exists solely to be differentiated.
- **Trust**: DINO-WM is the arbiter. Any local refinement that worsens the
  global cost beyond tolerance is discarded.

### 4.5 The shared-cache consequence (a correctness note)

Because both tracks read the *same* session-local cache (`DINO_WM_FEATURE_CACHE`,
not on Drive — FUSE random reads are too slow to train from), the local/global
notebook **rebuilds the cache once per fresh Colab session** via the idempotent
DINO-WM precompute script before training the surrogate. This is a direct
consequence of reuse: the surrogate cannot train on latents the DINO-WM track
has not yet materialized in this session.

---

## 5. Evaluation methodology

The first-version protocol is **offline latent goal reaching**: tasks are sampled
from held-out validation episodes; `G` is *both* the MPC simulator and the
scorer; the metric is the normalized final latent distance
`ρ = ‖z_fin − z_goal‖² / ‖z_start − z_goal‖²` with success `ρ < 0.5`.

The honest caveat, stated everywhere it matters: because `G` judges, `global_cem`
is structurally favoured on distance (it optimizes what it is scored on). The
comparative local/global claim is therefore read from **efficiency** (wall time,
global-forward vs backward counts) and the **acceptance rate**, not from raw
distance. Two de-confounders are built in: the **reference replay** (dataset's
true actions through the same simulator) calibrates achievable distance and
exposes `G`'s own rollout error, and the **disagreement** metric measures
surrogate drift directly. Real-environment confirmation (MuJoCo PointMaze via the
DINO-WM track's `plan.py`) is the designated next step and is out of scope for v1.

Artifacts and resumability (per-stage table in the runbook §5): each planner
writes `episodes.jsonl` / `summary.json` / `traces/`; a wall-clock cap makes a
slow session write `summary_partial.json` (not "done"), so re-running finishes
exactly the unfinished planners.

---

## 6. Correspondence summary

| FOG/DMO concept | DINO-WM asset it maps to | Local/global realization |
| --- | --- | --- |
| Global forward model (never differentiated) | Trained `VWorldModel` + latent cache + bypass patch | `DinoWMGlobalModel`, `no_grad`, chunked scoring |
| Local backward model (Jacobians) | — (new) | `LocalRolloutModel` over a frozen projection |
| Decoupling: gradients along the global trajectory | DINO-WM rollout as the trajectory | MPC re-anchors surrogate on `G`'s imagined latents |
| Exploitation safeguards | DINO-WM as the trusted arbiter | tanh bounds + smoothness + global re-score rejection |
| Goal-conditioned scoring | DINO-WM final-latent-vs-goal distance | `latent_goal_cost` shared by all global planners |
| On-distribution local accuracy | — (future) | designated future work: online surrogate fine-tuning on `G` rollouts |

---

## 7. Pointers

- Techniques catalogue: `docs/local_global_techniques.md`
- Operational runbook / next steps: `docs/local_global_runbook_next_steps.md`
- Technical report (authoritative, combined): `reports/world_models_report.tex`
  (Track III). The standalone `reports/local_global_report.tex` is a superseded
  draft, kept under `reports/drafts/`.
- Implementation brief (the spec): `LOCAL_GLOBAL_DINO_WM_IMPLEMENTATION_SPEC.md`
- Code: `src/wm_poc/local_global/`, `scripts/local_global/`,
  `configs/local_global/`; notebooks `03_local_global_foundation.ipynb`
  (operational) and `08_local_global_results.ipynb` (read-only).
