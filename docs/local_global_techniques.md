# Local/Global Track — Techniques and Design Decisions

This document records, in detail, the techniques used to build the local/global
planning track (`src/wm_poc/local_global/`, `scripts/local_global/`,
`configs/local_global/`, notebooks 03 and 08). It complements
`LOCAL_GLOBAL_DINO_WM_IMPLEMENTATION_SPEC.md` (the *what*) by explaining the
*how* and the *why* of each implementation choice, including the parts that
exist only because of Colab, Drive, or upstream-DINO-WM constraints.

---

## 1. Source-anchored design

The track is a deliberately simplified transplant of the coupled local/global
idea (Amigo et al., "Coupled Local and Global World Models for Efficient
First-Order RL", the FOG/DMO line of work) into the repository's existing
DINO-WM visual goal-reaching setting:

| FOG / DMO concept | This repository |
| --- | --- |
| Global forward model: large diffusion WM, trusted rollouts, never differentiated | Frozen DINO-WM latent world model (already trained by notebook 02), driven in-process on cached DINOv2 patch latents, always under `torch.no_grad()` |
| Local backward model: lightweight RSSM supplying Jacobians | Small residual-MLP/GRU surrogate in a compressed projection of the latent space, fully differentiable w.r.t. actions |
| Policy optimization with decoupled forward/backward passes | Planner-level decoupling: CEM searches with the global model, gradient descent refines through the local surrogate |
| Guards against model exploitation (real-robot patching loops) | Three in-code safeguards: differentiable action squashing, action-smoothness penalty, global re-score with rejection tolerance (spec §8.3) |
| Local model only needs *single-step accuracy around the forward trajectory* | The MPC loop re-anchors the surrogate on the global model's imagined latents every `mpc_exec_steps`, and the evaluation logs open-loop local-vs-global disagreement |

Two upstream papers fix the representation choices: DINO-WM (frozen
`dinov2_vits14` patch features, action-conditioned latent transition model,
no pixel reconstruction, latent-distance goal costs) and the JEPA-WM framing
(context windows, frameskip-folded actions, CEM-style planning). Per spec §12,
the first pass deliberately does **not** train a diffusion model, backprop
through the global model or encoder, reconstruct pixels, or require rewards.

---

## 2. Architecture overview

```
DINO-WM latent cache (episode_XXX.npy, fp16, (T, 196, 384))   actions/states .pth
        │                                                          │
        └────────────── LatentTrajectoryStore (mmap) ──────────────┘
                                │
              ┌─────────────────┼──────────────────────┐
              │                 │                      │
   LatentWindowDataset   export_transitions.py   sample_episode_tasks
   (context C, rollout K,   (manifest + stats)     (offline goal tasks
    frameskip folding)                              from val episodes)
              │                                        │
   train_local_surrogate.py                  run_planning_eval.py
   PatchProjector (frozen) +                 GlobalModel adapter (DINO-WM or
   LocalDynamics / GRU                       synthetic) + planner registry:
   = LocalRolloutModel ────────────────────► global_cem / local_cem / local_gd /
   (checkpoints/local_best.pt)               local_adam / hybrid (±rescore)
                                                       │
                              planning/<planner>/{episodes.jsonl, summary.json,
                                                  traces/*.jsonl}
                                                       │
                              summarize_runs.py → _summary/summary.csv
                              notebook 08 (read-only plots/tables)
```

Everything heavy is a script; the notebooks dispatch, gate, and display.

---

## 3. Data layer techniques

### 3.1 Reuse of the DINO-WM latent cache, byte for byte

The surrogate trains on the exact cache the DINO-WM track produces
(`wm_poc_latent_manifest.json` + `episode_{i:03d}.npy` of shape
`(T, patches, embed_dim)` float16). No second representation is introduced:
`LatentTrajectoryStore` reads the manifest for episode lengths/shapes and
`np.load(..., mmap_mode="r")`s episodes lazily, so a 2 200-episode cache costs
no RAM until windows are actually sliced, and fp16 disk data is cast to fp32
only per accessed window.

Actions come from the upstream data layout (`actions.pth`, `seq_lengths.pth`,
`states.pth`), loaded through a `.npy`-first/`.pth`-fallback helper so the
synthetic task (NumPy-only) and the real task (torch tensors) share one code
path. `seq_lengths` truncates per-episode action rows; the store validates the
`(episodes, steps, action_dim)` shape and that the action tensor covers the
cached episode count.

### 3.2 Frameskip folding, identical to upstream

DINO-WM models one "step" as `frameskip` raw environment actions. The dataset
mirrors the upstream `TrajSlicerDataset` convention exactly:

- latent frames are sampled at `t0, t0+fs, t0+2fs, …`;
- the action block for the transition between frame `k` and `k+1` is the
  concatenation of raw actions over `[t0+k·fs, t0+(k+1)·fs)` —
  `rearrange("(n f) d -> n (f d)")` in upstream terms, `fold_actions()` here;
- therefore `step_action_dim = action_dim × frameskip` (PointMaze: 2×5=10),
  and planner bounds are the raw bounds tiled `frameskip` times
  (`PlannerConfig.step_action_low/high` properties).

Window validity is closed-form: a window of `C` context + `K` target frames
spans `(C+K−1)·fs` raw steps, so the largest valid start is
`min(T_latents − 1 − span, T_actions − span)` (`max_window_start`), and
windows enumerate `t0` with an optional stride and `max_windows` cap for
smokes.

### 3.3 Episode-level, cap-stable splits

Two leakage hazards are addressed:

1. **Window-level leakage**: train/val are split *by episode*, never by window
   (`split_episodes`: seeded `default_rng(seed).permutation`, val = first
   `round(N·val_fraction)` of the permutation, deterministic and disjoint).
2. **Cap-instability leakage**: smokes set `training.max_episodes`, which
   shrinks the store. If the permutation were drawn over the *capped* count,
   train/val membership would differ between a capped training run and an
   uncapped evaluation, silently leaking training episodes into "held-out"
   evaluation tasks. `split_store_episodes` therefore always permutes the
   *uncapped* `store.total_episodes` and then filters to the capped range, so
   membership is invariant to the cap. Training, transition export, and
   planning evaluation all call this one helper.

### 3.4 A synthetic task that is exact by construction

`generate_synthetic_task` writes a tiny point-mass dataset in the *same
on-disk format* as the real cache (same manifest keys, same `episode_XXX.npy`
naming, fp16), so every downstream component runs unmodified on CPU:

- ground-truth dynamics: `v' = 0.9·v + 0.1·a`, `p' = clip(p + 0.1·v', ±1)`;
- latents are an exact *linear* encoding `z = W s + b` with a fixed seeded
  Gaussian `W` (saved as `encoder_weight.npy`/`encoder_bias.npy`), so a
  perfect global model exists in closed form: decode with the pseudo-inverse,
  step the true dynamics, re-encode (`SyntheticPointGlobalModel`);
- `synthetic_dynamics.json` makes the data self-describing — the global model
  reads `dt`/`damping` from the file rather than trusting module constants.

This gives the planner stack a setting where "success" is measured against
*true* dynamics, which is what makes the CPU smoke an actual correctness check
rather than a does-it-crash check. Clobber guards refuse to generate synthetic
data into a non-empty cache dir or next to a real `actions.pth` (a
half-written real cache must never be silently replaced), and the error
message includes the exact `rm -rf` remediation.

### 3.5 Dataset statistics for normalization parity

Checkpoints trained with upstream `normalize_action: true` consumed
`(x − mean)/std` actions and proprio. `compute_action_state_stats` reproduces
the upstream statistics (per-dimension mean/std over `seq_lengths`-masked raw
actions and states, `ddof=1`, epsilon-guarded), and the DINO-WM adapter
normalizes **at its own boundary** — planners, datasets, and the surrogate all
keep working in raw action units, so no other component needs to know whether
the checkpoint was normalized.

---

## 4. Local surrogate techniques

### 4.1 Frozen orthogonal projection (the collapse argument)

The projector maps `(patches, embed_dim)` patch latents to a `local_dim`
vector: per-patch LayerNorm (non-affine) → pooling → linear. The linear map is
**frozen by default**, initialized as a seeded semi-orthogonal matrix via thin
QR of a Gaussian (a generator-seeded reimplementation, because
`nn.init.orthogonal_` takes no generator on older torch — and only the thin
factor is computed, never a `max(dim)²` QR).

Why frozen: the training objective is an MSE *in projected space* between a
rollout and the projection of cached targets. If the projector were trained
jointly, `projector ≡ 0` is a global optimum — the classic representation
collapse. A frozen near-orthogonal projection of LayerNormed features
preserves distances (Johnson–Lindenstrauss-style) and removes the degenerate
solution entirely. `projection_trainable: true` exists as an escape hatch but
is documented to require `training.lambda_variance > 0`
(`variance_penalty`: hinge on per-dimension std, a VICReg-style
anti-collapse term).

Two pooling modes trade spatial fidelity against simplicity:

- `mean_pool_linear` (spec default): mean over all patches → `Linear(D, local)`;
- `grid_pool_linear` (used by the full PointMaze experiment): patches reshaped
  to their `√P×√P` grid, `adaptive_avg_pool2d` to a `g×g` grid, flattened →
  `Linear(g²·D, local)` — coarse object-position information survives pooling.

### 4.2 Residual dynamics with an optional recurrent context

- `LocalDynamics`: `x_{t+1} = x_t + f_θ([x_t; a_t])`, an MLP with input
  LayerNorm and SiLU activations. The residual parameterization makes the
  identity (no movement) the zero-function default, which both stabilizes
  early training and keeps the `delta_mse` loss meaningful.
- `ContextLocalDynamics` (model_type `gru_residual`): one `GRUCell` consumes
  the `C−1` context transitions `(x_t, a_t)` to build a hidden state (velocity
  inference, as the spec's "use context to infer velocity" suggests), and the
  *same cell* predicts forward steps; a small MLP head decodes the hidden into
  the residual delta.
- `LocalRolloutModel` wraps projector + dynamics behind three calls used
  everywhere: `encode_global_latent(z)`, `step(x, a, hidden)`,
  `rollout_from_context(x_ctx, a_ctx, actions)` (plus the spec's simple
  `rollout(x0, actions)`). Rollouts are plain Python loops over time —
  horizons are ≤ 6, so unrolled autograd through the loop is cheap and exact.

Everything is differentiable with respect to actions; tests assert nonzero
`actions.grad` through both model types.

### 4.3 Loss design

`combined_local_loss` returns `(scalar, metrics_dict)` and composes:

```
L = λ_rollout · rollout_mse(pred[1..K], target[1..K], discount)
  + λ_one_step · one_step_mse(pred[1], target[1])
  + λ_delta    · delta_mse(x_t, pred[1], target[1])
  + λ_jacobian · jacobian_norm_penalty(...)      (optional, double backward)
  + λ_variance · variance_penalty(x_targets)     (only with trainable projector)
```

- `rollout_mse` reduces per-step first, then applies a normalized geometric
  discount, so `discount=1.0` is exactly the plain mean and constant-error
  sequences are discount-invariant (unit-tested).
- `delta_mse` penalizes the *change* `(x̂_{t+1}−x_t)` against the true change —
  it keeps predicted step magnitudes calibrated even when absolute MSE is
  dominated by static features.
- `jacobian_norm_penalty` regularizes `‖∂step/∂a‖²` via
  `torch.autograd.grad(..., create_graph=True)` (off by default; the knob the
  spec reserves for taming exploitable gradients).
- A scale-free diagnostic, `rollout_mse_vs_static`, divides the rollout error
  by the "predict no change" baseline: values ≪ 1 mean the surrogate beats the
  trivial predictor regardless of the projection's numeric scale.

### 4.4 Self-contained checkpoints

`save_local_checkpoint` stores `{format_tag, build_kwargs, model_state, step,
metrics[, optimizer_state]}` with an atomic tmp-then-replace write.
`load_local_checkpoint` rebuilds the model **from `build_kwargs` alone** —
evaluation never needs the original YAML to reconstruct the exact architecture
(and a format tag guards against loading foreign checkpoints). `local_best.pt`
(lowest val loss) and `local_latest.pt` (rolling, every `save_every` steps and
at every stop) follow the repo's best/latest convention.

Training is **resumable across sessions**: `local_latest.pt` carries the AdamW
optimizer state and the step counter, and the trainer's `try_resume` restores
all three (refusing with a clear error if `build_kwargs` changed; `--no-resume`
restarts). Combined with the per-session `max_wall_minutes` stop, an
interrupted Colab session costs at most `save_every` steps of progress, and
the notebook's training cell distinguishes complete (skip) / partial (resume) /
fresh launches by reading the checkpoint's step against `max_steps`.

---

## 5. Global model adapters

### 5.1 One minimal protocol, two implementations

Planners and the MPC loop only see four methods (a `Protocol`, not a base
class, so test doubles are plain classes):

```python
init_state(z_context, proprio_context, actions_context) -> opaque dict
rollout_final(state, actions(B,K,A))                    -> (B, P, D) final latents
advance(state, actions(m,A))                            -> new state (+ per-step latents)
current_latent(state)                                   -> (P, D)
```

The state is an opaque dict on purpose: the synthetic model keeps a decoded
point-mass state; the DINO-WM adapter keeps latents/proprio/action history.
The MPC loop cannot accidentally depend on either representation.

### 5.2 The DINO-WM adapter (the one untestable-locally piece)

`DinoWMGlobalModel` loads the upstream checkpoint *exactly the way upstream
`plan.py` does*: `OmegaConf.load(outputs/<run>/hydra.yaml)` +
`plan.load_model(checkpoints/model_<epoch>.pth, …)` with the upstream repo
prepended to `sys.path`. Techniques worth noting:

- **Latent bypass dependency**: rollouts feed cached patch latents directly to
  `VWorldModel.encode_obs`, which only works because the repo's marker-guarded
  `WM_POC_DINO_LATENT_BYPASS_PATCH` (from the DINO-WM track) makes
  `encode_obs` accept 4-D latent input. The notebook verifies/installs the
  patch before planning.
- **Replay-from-anchor context handling**: upstream `rollout(obs_0, act)`
  expects one folded action block per context frame followed by future blocks.
  Instead of mutating a context window with imagined latents (and having to
  fabricate proprio for imagined frames), the adapter keeps the *original
  observed* context (latents + raw proprio + the `n−1` observed action blocks)
  plus the list of executed blocks, and **replays from that anchor on every
  call**. Proprio conditioning therefore always reflects something actually
  observed, and the upstream act-layout contract is satisfied by
  construction.
- **Normalization at the boundary** (§3.5): folded blocks are normalized by
  `mean/std` tiled `frameskip` times; proprio by the state stats; nothing
  outside the adapter sees normalized units. A checkpoint trained with
  `normalize_action: true` but constructed without stats fails fast with an
  instructive error.
- **Config self-description**: `normalize_action` and `num_action_repeat` are
  discovered by a depth-first search over the OmegaConf container
  (`_find_in_cfg`) rather than hardcoded paths, because hydra config layouts
  drift between runs.
- **Defensive output extraction**: `_extract_visual` accepts tuple/dict/tensor
  returns and slices `[..., :latent_dim]`, because the upstream rollout
  returns concatenated per-patch features (visual ⊕ proprio ⊕ action
  embeddings) and its exact return shape is version-dependent.
- **Chunked candidate scoring**: `rollout_final` processes CEM populations in
  `rollout_batch_size` chunks. Candidates are independent, so chunking is
  *provably* identical in output and per-candidate call accounting (unit test
  compares chunk=3 vs chunk=100 on the same candidates) — only peak memory
  changes. This is the knob that makes the full experiment run on a 16 GB T4.
- **Test-injection hook**: the constructor accepts `wm=` to bypass the
  upstream load entirely, so the adapter's tensor plumbing (context padding,
  normalization, chunking, `advance` bookkeeping) is unit-tested against a
  deterministic fake upstream model even though the real load needs Colab.
- All calls run under `torch.no_grad()` — the FOG separation "gradients only
  ever come from the local model" is enforced structurally, not by
  convention.

---

## 6. Planner techniques

### 6.1 Shared interfaces and honest cost bookkeeping

`PlanContext` (inputs incl. an MPC-shrunken `horizon` and the surrogate's
trained `local_context_len`) and `PlanResult` (actions, cost components,
per-iteration trace, metadata) are dataclasses shared by all planners; the
registry `build_planner(name, …)` maps the six spec names to instances and
validates required models.

Planners *optimize* `goal + λ_smooth · smoothness` but *report* the components
separately: `goal_cost` is always the pure latent goal distance of the
returned sequence (re-evaluated once at the end — the `+1` forward call is
counted), `total_cost` the optimized objective. This prevents the smoothness
weight from contaminating cross-planner distance comparisons.

Every planner counts `num_global_forward_calls` (per candidate, not per
batch), `num_local_forward_calls`, `num_backward_steps`, and wall time —
the spec §8.4 "make planner comparison fair" requirement is satisfied by
construction, and the results notebook surfaces these columns next to success
rates.

### 6.2 CEM (`cem_optimize`)

- Population sampled as `mean + σ·ε`, `ε ~ N(0,1)` from a **CPU
  `torch.Generator` seeded per planning round** (`seed + round_index`), then
  moved to device — determinism is therefore device-independent and
  reproducible (unit-tested: same seed ⇒ identical actions and cost).
- Samples are clamped to bounds (sampling-side clipping is fine; only the
  gradient path needs differentiable bounds), the current mean is kept in the
  pool (slot 0) so the distribution can never lose its incumbent, elites refit
  mean and (population) std, and the std is floored at `1e-4` to prevent
  premature collapse.
- `cem_init_std` is expressed in units of half the action range, so one number
  works for any bounds.
- A **best-ever** sequence is tracked across iterations and returned (an
  elite refit can wander away from the best sample seen).
- Per-iteration trace rows `{iter, best_cost, mean_cost, elite_mean_cost,
  std_mean}` feed the results notebook's optimization plots.

### 6.3 First-order planning (`gradient_optimize`)

- **Bounded by parameterization, not projection**: raw unconstrained
  parameters pass through `squash_to_bounds` (affine tanh) inside the autograd
  graph, so iterates are always feasible and gradients respect the boundary
  geometry. `action_bound_violations` is logged and is structurally zero.
- **Warm-start desaturation**: hybrid refinement initializes from CEM's best
  sequence via `atanh`. CEM clamps samples *onto* the bounds, and
  `atanh(1−1e-5)` sits where the tanh derivative is ≈2·10⁻⁵ — refinement
  would be frozen. Warm starts therefore unsquash with a dedicated
  `WARM_START_EPS = 1e-2` margin (derivative ≈0.02), trading a ≤1% bound
  offset for usable gradients. This constant exists because the
  zero-gradient failure mode was observed, and the rationale is documented at
  the definition site.
- Adam or SGD over the raw parameters, `clip_grad_norm_` with the configured
  clip, and again a **best-iterate** return: the best `(cost, components,
  actions)` triple seen during optimization is returned, not the last iterate
  (gradient steps can overshoot near convergence).
- The cost callback returns `(scalar, components)` so traces carry
  `goal_cost`/`smoothness_cost` per iteration without re-evaluation.

### 6.4 The hybrid planner and the rejection rule

`HybridCEMLocalRefinePlanner` composes the two primitives:

1. global CEM proposes (sharing one `init_state` so the global model's
   context is built once per round);
2. the local gradient planner refines, warm-started from the CEM sequence,
   with the *same* shrunken horizon;
3. the refined sequence is **re-scored by the global model on the identical
   objective** (goal + smoothness — comparing a goal-only number against a
   goal+smoothness number would bias rejection);
4. with `global_rescore=True` (planner name suffix `_global_rescore`), the
   refinement is rejected if it is worse than CEM by more than a *relative*
   tolerance `reject_refine_if_worse_by · max(|cem_total|, 1e-12)`, and the
   CEM sequence is returned unchanged. Without the suffix, the refined
   sequence is always returned but the re-scored cost is still logged.

Both `accepted_refinement` and `refinement_improved_global_cost` are recorded
per round, so the results notebook can distinguish "refinement helped",
"refinement hurt but was within tolerance", and "refinement rejected" — the
local-vs-global disagreement signal the FOG paper cares about, at planner
granularity. Traces from both stages are concatenated with a `stage` tag so a
single plot shows CEM descent followed by gradient descent.

`local_cem` exists purely as an ablation to separate the "small model" effect
from the "gradient" effect (spec §4.5.4).

---

## 7. Evaluation methodology

### 7.1 Offline latent goal reaching, with the caveat stated everywhere

The v1 evaluation never touches MuJoCo: episodes are sampled from held-out
validation episodes of the latent cache; the start is a context window, the
goal is the cached latent `goal_steps` model-steps later, and the **global
model is both the MPC simulator and the scorer**. For the synthetic task the
global model is exact, so results reflect true dynamics; for DINO-WM this is
"judged by the trusted global model", which *structurally favors
`global_cem`* (it optimizes the metric it is scored by). This caveat is
stated in the module docstring, both notebooks, and the recommendation cell —
the actionable local/hybrid signal is wall time, backward-step efficiency,
and re-score acceptance, with real-environment confirmation deferred to the
DINO-WM track's `plan.py`.

### 7.2 MPC loop details

- **Shrinking horizon**: each round plans `min(horizon_cfg, steps_remaining)`
  steps, so final rounds optimize *arrival at* the goal rather than a point
  `horizon` steps beyond it.
- Executed chunks advance the simulator; the per-step imagined latents
  returned by `advance` extend the planning context window, and executed
  action blocks replace the oldest context blocks — the surrogate is always
  conditioned the same way it was trained (frameskip-spaced frames, folded
  blocks between them), truncated to its trained `context_len`.
- Per-round records keep the full planner trace, costs, and metadata; they are
  written as `traces/episode_XXX.jsonl` (one JSON line per round).

### 7.3 Calibration and metrics

Raw latent MSE distances are meaningless in isolation, so two normalizers are
built into every episode record:

- `normalized_final_distance = d(final, goal) / d(start, goal)` — success is
  `< success_threshold` (default 0.5, "closed at least half the gap"),
  scale-free across encoders and tasks;
- **reference replay**: the dataset's *true* action sequence from start to
  goal is replayed through the same simulator, giving
  `reference_final_distance_global` — the achievable-by-construction
  benchmark that calibrates what "good" means for the threshold (near 0 for
  the exact synthetic model; nonzero under a learned global model, which is
  itself informative about model error).
- `local_global_disagreement`: the executed sequence is replayed *open-loop*
  through the surrogate and compared step-by-step (in projected space)
  against the global model's imagined trajectory — the spec §8.1
  local-vs-global rollout disagreement, measured on-policy.
- Plus the spec metric list: per-episode success, global/local final
  distances, episode steps, wall time, first/final-iteration plan costs,
  acceptance rate, bound violations, and the three compute counters.

### 7.4 Artifact contract and resumability

Each planner writes `planning/<planner>/episodes.jsonl` (one metrics row per
episode, appended), `summary.json` (aggregates via a None-tolerant `_mean`),
`traces/`, and `eval_state.json` (the task-defining parameters). Re-runs are
safe and resumable at **per-episode** granularity:

- a wall-clock `deadline` (`evaluation.max_wall_minutes`, CLI-overridable)
  stops *between episodes*; a capped planner writes `summary_partial.json` for
  the episodes it finished;
- on re-run, if `eval_state.json` matches (same `episode_seed`, threshold,
  `goal_steps`, frameskip, context length, and split), evaluation **resumes
  from the episodes already in `episodes.jsonl`** rather than restarting. The
  i-th task is deterministic in `episode_seed`, so this is exact — including
  after *raising* `num_episodes` (the first N tasks are unchanged), which is
  what lets a 50-episode run extend to 100 without recomputing the first 50.
  A parameter mismatch clears the stale artifacts and starts fresh;
- the notebook treats a planner as "done" only when its `summary.json` was
  produced at >= the configured `num_episodes`, so bumping the episode count
  re-runs under-sampled planners. Slow sessions cost a resume, never a
  silently smaller experiment. (100 episodes ⇒ ±0.10 95% CI on success rate, a
  compromise between 50 at ±0.14 and 200 at ±0.07/~2x runtime; the paired,
  near-deterministic efficiency metrics need fewer.)

---

## 8. One experiment, any GPU (throughput/quality decoupling)

A user-level invariant of this codebase: **hardware tier must never water
down the experiment** (A100s are nearly unobtainable on Colab; most sessions
get T4s).

- `pointmaze_surrogate_a100.yaml` *defines* the experiment — model sizes,
  projection, CEM/GD budgets, episode count, planner set, and the run name
  `pointmaze_local_full_seed0`.
- `pointmaze_surrogate_t4.yaml` `extends:` it and overrides **throughput knobs
  only**: `global_model.rollout_batch_size` (32 vs 128) and
  `training.num_workers`. It inherits the run name, so a run started on one
  GPU tier resumes on another, into the same run directory.
- The two mechanisms that make this safe are the chunked candidate scoring
  (§5.2 — identical results at any chunk size) and the wall-clock caps with
  planner-level resume (§7.4). Training has its own `max_wall_minutes` stop.
- Notebook 03's GPU detection picks a *throughput profile*, and prints it as
  such; `LG_CONFIG` overrides.

---

## 9. Repository-integration techniques

### 9.1 Config system: reuse, then validate

The track reuses the DINO-WM track's plain-PyYAML helpers (no
OmegaConf/Hydra): single-parent `extends:` with recursive deep-merge, and
`${oc.env:VAR,default}` placeholders resolved in a separate non-mutating pass
(full-placeholder strings get scalar-parsed, so numbers stay numbers). On top:

- `validate_local_global_config` fails fast with one-line messages (track tag,
  required sections, enum membership for model types/projections/planners,
  bound-shape and elite/population sanity, `val_fraction ∈ (0,1)`);
- typed **frozen dataclass views** (`GlobalModelConfig`, `LocalModelConfig`,
  `PlannerConfig` with derived `step_action_*` properties) are built *from*
  the dict — the dict stays the source of truth (repo convention), the
  dataclasses give models/planners typo-proof attribute access;
- path derivation mirrors the DINO-WM cache layout
  (`<cache_root>/<task>/<encoder>_img<size>`) with explicit-override keys, so
  one env var (`DINO_WM_FEATURE_CACHE`) keeps both tracks pointed at the same
  cache;
- `resolve_run_dir` creates the spec §7 layout (`transition_data/`,
  `checkpoints/`, `metrics/`, `planning/`, `figures/`) and auto-names runs
  `local_global_<task>_<UTC stamp>` when unset; `save_resolved_config` writes
  `config_resolved.yaml` (atomic) into every run dir for provenance.

### 9.2 Execution-gating ladder

Mirroring `RUN_DINO_WM`: every heavy script is inert unless
`RUN_LOCAL_GLOBAL=1`, with two always-allowed escape hatches — `--dry-run`
(print the plan, touch nothing) and `--smoke` (tiny capped run, applies a
tuple of dot-path overrides like `cem_population→16`, `max_steps→200`). The
gate constant lives in one place (`configs.RUN_GATE_ENV`). Scripts are
runnable from the repo root, bootstrap `sys.path` themselves
(`parents[2]/src`), use `pathlib` and argparse, and print resolved paths
before acting (AGENTS.md code style).

### 9.3 Artifact placement discipline

Durable artifacts (run dirs, checkpoints, metrics, planning outputs) go to
Drive (`LG_RUN_ROOT`, default `<Drive>/logs/local_global`); the latent cache
stays on session-local disk (Drive FUSE random reads are too slow to train
from); smoke artifacts go under a dedicated `LG_SMOKE_ROOT`. Nothing large is
ever committed: the repo's `.gitignore` already blocks `*.pt/*.pth/*.npy/
runs/` etc., and `_`-prefixed directories (`_summary`, `_synthetic`) are
excluded from run discovery by convention.

---

## 10. Notebook engineering

### 10.1 The two-notebook pattern

Notebook 03 is operational (setup → verification → guarded launches →
summary), notebook 08 is strictly post-hoc (CPU-only, read-only, plots into
`_summary/figures/`). 08's only subprocess is the summarizer with
`check=False`; every renderer degrades to a friendly message when artifacts
are missing, so it runs usefully against smoke-only state.

### 10.2 Self-gating, self-arming launch cells

Heavy cells follow the repo's established idiom (notebook 02): query
completion state, then **skip / resume / launch** with the reason printed.
The track's versions also check *prerequisites*, so a top-to-bottom run on a
fresh machine degrades to instructions instead of tracebacks:

- training skips when `local_best.pt` exists (`LG_FORCE_RETRAIN=1` to force),
  refuses to arm when the latent cache is missing;
- planning skips planners that already have `summary.json`, requires a
  *usable* global checkpoint and a trained surrogate, and re-launching
  evaluates only the missing planners;
- arming is `enable_local_global_runs()` (sets the env gate) immediately
  followed by the `run_lg(...)` dispatch — a bare `run_lg` without arming is
  inert.

### 10.3 Colab session bootstrap (learned the hard way)

Three Colab-specific behaviors are handled explicitly, each discovered from a
real failed session:

1. **Subprocess output is invisible**: children writing to the kernel's
   inherited stdout fd don't render in Colab cells. `run_if_safe` therefore
   uses `Popen(stdout=PIPE, stderr=STDOUT, text=True, bufsize=1)` and streams
   lines into the cell via `print`, ending with an explicit
   `[exit code N]` line.
2. **The latent cache dies with the session**: the latent-cache cell is live —
   when the manifest is missing on Colab it installs the upstream deps
   (`install_colab_deps.py --quiet`) and rebuilds the cache via the DINO-WM
   precompute script (idempotent: it short-circuits when the cache already
   covers the dataset), with `RUN_DINO_WM=1` scoped to that invocation.
3. **`python` doesn't exist everywhere**: all dispatch uses `sys.executable`.

### 10.4 Auto-resolving the global model reference

The config cell resolves which DINO-WM run drives the global model instead of
trusting a hardcoded default:

- a run is *usable* iff it has `hydra.yaml` **and** a loadable checkpoint,
  where loadable excludes `model_latest_step.pth` (the rolling intra-epoch
  state-dict checkpoint that upstream `plan.py`-style loading rejects);
- when the configured name is unusable and usable runs exist, the newest (by
  checkpoint mtime) is selected and exported as `LG_GLOBAL_MODEL_NAME` — env
  export is the propagation mechanism, because launched scripts re-resolve
  the YAML placeholders themselves;
- `model_epoch=latest` falls back to the highest numbered epoch via
  `LG_GLOBAL_MODEL_EPOCH` when `model_latest.pth` is absent;
- the dependency-verification cell prints a per-run table (hydra.yaml /
  loadable checkpoints / checkpoints parked in
  `checkpoints_fresh_start_backup_*` dirs by forced restarts) and the exact
  `mv` command to restore a backup.

This logic was validated locally by executing the notebook against a **fake
checkpoint tree** (`runA` hydra-only, `runB` usable, `runC` no hydra, `runD`
step-checkpoint-only) and asserting the selection lands on `runB`.

---

## 11. Testing strategy

### 11.1 Torch-optional by construction

Repo convention: the committed suite must pass in an environment without
torch. Torch-dependent test modules start with
`torch = pytest.importorskip("torch")` followed by `# noqa: E402` imports
(ruff-clean); library modules guard their torch import with a
`try/except → RuntimeError` so config/visualization/summarizer code stays
importable anywhere. Result: ~130 tests run torch-free (configs, dataset
math, summarizer, plots via `importorskip("matplotlib")`), the full ~170 run
with torch.

### 11.2 Toy models encode planner contracts

`test_planners.py` builds exact linear toy models
(`x' = x + gain·a`, latent = broadcast state) so planner behavior is
verifiable analytically:

- CEM reduces a quadratic and respects bounds; same seed ⇒ identical result;
- gradient descent converges on a reachable goal and counts its backward
  steps;
- **the sign-flipped surrogate** (`ToyLocalModel(gain=−0.1)` against
  `ToyGlobalModel(gain=+0.1)`) makes local refinement *provably harmful*, and
  the tests assert the re-score gate rejects it (and that the non-rescore
  variant keeps it but logs the worse cost) — the safeguard is tested by
  constructing the exact failure it exists for;
- squash/unsquash round-trips within tolerance; the planner registry
  validates required models and rejects unknown names.

### 11.3 Integration tests on the synthetic task

`test_eval.py` drives the *real* `evaluate_planner` end-to-end on the
generated synthetic task (exact global model, untrained surrogate checkpoint
written by the fixture): artifacts exist, schemas hold, determinism of task
sampling, and the no-checkpoint error path. `test_global_models.py` uses the
`wm=` injection hook with a deterministic fake upstream model to prove
chunked scoring is exact and `advance` returns per-step latents.

### 11.4 Acceptance layers above pytest

- `scripts/local_global/run_smoke.sh`: the end-to-end acceptance check —
  fresh synthetic data → export → train (~200 steps) → all six planners →
  summary CSV, ~1–2 min on CPU. It wipes its own smoke-owned artifacts first
  so an interrupted attempt cannot wedge re-runs. The observed smoke ordering
  is itself a sanity signal (exact global model: `local_adam` reaches the
  lowest distances, hybrids improve on pure CEM, `local_gd` is weakest).
- **Local notebook execution**: both notebooks are executed top-to-bottom via
  `nbclient` with the Colab git-bootstrap cell stubbed and `LG_RUN_ROOT`
  pointed at a temp dir — asserting zero error outputs, and (for 08) that
  figures actually render from smoke artifacts. This caught every
  cell-ordering and environment bug before it reached Colab.
- Environment validation runs in *two* venvs (torch 2.2.2 + `numpy<2`, and a
  torch-free one), matching the repo's "validate against both worlds" rule.

---

## 12. Known limitations and deferred work

Deliberate v1 boundaries (spec §12 plus observed constraints):

- no real-environment (MuJoCo) evaluation in this track yet — the offline
  global-model-as-simulator protocol favors `global_cem` by construction;
  real-env confirmation goes through the DINO-WM track's `plan.py`;
- the DINO-WM adapter's upstream contract (`plan.load_model` importability,
  `rollout` signature/return) is exercised only on Colab; it fails loudly
  with instructions rather than silently;
- no PushT until PointMaze passes a real-latent run (latent cache pipeline is
  point_maze-only upstream);
- candidate later safeguards from the spec remain unimplemented by choice:
  surrogate ensembles, disagreement penalties inside planning costs,
  PCA projections, online surrogate fine-tuning on global rollouts (the FOG
  "model mini-epoch"), and dataset-magnitude-constrained CEM elites.

---

## 13. File map

| Area | Files |
| --- | --- |
| Config loading/validation | `src/wm_poc/local_global/configs.py` (reuses `wm_poc/dino_wm/configs.py`) |
| Data layer + synthetic task | `src/wm_poc/local_global/datasets.py` |
| Surrogate models + checkpoints | `src/wm_poc/local_global/models.py` |
| Losses | `src/wm_poc/local_global/losses.py` |
| Global model adapters | `src/wm_poc/local_global/global_models.py` |
| Planners | `src/wm_poc/local_global/planners.py` |
| MPC evaluation | `src/wm_poc/local_global/eval.py` |
| Post-hoc loading/plots | `src/wm_poc/local_global/visualization.py` |
| CLI scripts | `scripts/local_global/{export_transitions,train_local_surrogate,run_planning_eval,summarize_runs}.py`, `run_smoke.sh` |
| Configs | `configs/local_global/{base,smoke_synthetic,smoke_pointmaze,pointmaze_surrogate_a100,pointmaze_surrogate_t4}.yaml` |
| Notebooks | `notebooks/03_local_global_foundation.ipynb` (operational), `notebooks/08_local_global_results.ipynb` (read-only) |
| Tests | `tests/local_global/test_{configs,datasets,models,losses,planners,global_models,eval,summarize}.py` |
| Spec | `LOCAL_GLOBAL_DINO_WM_IMPLEMENTATION_SPEC.md` |
