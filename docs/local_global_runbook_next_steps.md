# Local/Global Track — Runbook: Finishing the First Real Run and What Comes After

Companion to `docs/local_global_techniques.md` (how it's built) and
`LOCAL_GLOBAL_DINO_WM_IMPLEMENTATION_SPEC.md` (what was asked). This document
is the *operational* guide: where you are in the pipeline, how to close out
the first real PointMaze run, how to read the results, and what the research
arc looks like afterwards. It is written to be usable mid-run (e.g. while
surrogate training is still going).

---

## 0. Where you are in the pipeline

```
[done]    DINO-WM global model trained (notebook 02)            -> Drive checkpoint
[done]    Latent cache rebuilt in this session (notebook 03)    -> /content/wm_poc_latent_cache
[done]    Transition export                                     -> <run>/transition_data/
[running] Surrogate training                                    -> <run>/checkpoints/local_{best,latest}.pt
[next]    Validation rollouts cell                              -> reads <run>/metrics/val_rollouts.jsonl
[next]    Planning evaluation                                   -> <run>/planning/<planner>/...
[next]    Summary cell + notebook 08                            -> _summary/summary.csv + figures
```

The run dir for the full experiment is
`$LG_RUN_ROOT/pointmaze_local_full_seed0` on Drive, shared by the A100 and T4
throughput profiles.

### While training is still running

- The training log prints `steps/s` and an **ETA** at every validation
  (every 500 steps). On a T4 expect roughly 0.2–0.5 s/step → 1–2.5 h for the
  full 20 000 steps; the run is I/O-bound, not GPU-bound.
- Each session is capped by `training.max_wall_minutes` (300). If the session
  ends — cap, disconnect, anything — **nothing is lost beyond the last 1000
  steps**: `local_latest.pt` (model + AdamW optimizer state + step counter)
  is written every `save_every=1000` steps and at every stop, onto Drive.
- To continue after an interruption: just re-run notebook 03 top-to-bottom in
  the new session. The cache cell rebuilds the session-local cache
  (~10–20 min), and the training cell prints
  `Partial run found: resuming from step N/20000` and continues. When the
  checkpoint reaches `max_steps`, the same cell switches to
  `Training complete ... skipping` forever after.
- `local_best.pt` (lowest validation loss so far) is what planning loads — it
  updates on every validation improvement, so it is always a sensible model
  even mid-training.

---

## 1. Closing out the experiment (after training completes)

1. **Validation rollouts cell** — sanity-check the surrogate before spending
   planning compute:
   - `one-step MSE` and the per-horizon `rollout step k: MSE ...` lines should
     be finite and *not* exploding across the 3 rollout steps;
   - in the training prints, `val vs-static` is the scale-free check: it is
     the rollout error divided by a "predict no change" baseline. **< 1 means
     the surrogate beats the trivial predictor**; the lower the better. If it
     hovers near 1.0, the surrogate learned almost nothing useful and the
     local planners will not be competitive — still worth running the
     comparison, but expect re-score rejections.
2. **Planning evaluation cell** — self-gating per planner: a planner is "done"
   only when its `summary.json` was produced at >= the configured
   `num_episodes` (so raising 50 -> 100 re-runs under-sampled planners). After
   a wall-clock cap (300 min) a planner leaves `summary_partial.json` and
   **resumes per episode** next time (continuing from the episodes already in
   `episodes.jsonl`, not from scratch). Six planners × 100 episodes is roughly
   1-2.5 h total on either GPU tier; with the 300-min cap it often fits in a
   single session, and per-episode resume covers disconnects. The heavy
   planners (`global_cem` and the two hybrids) dominate while the three local
   ones finish quickly.
3. **Summary cell** — aggregates everything under `$LG_RUN_ROOT` into
   `_summary/summary.csv`.
4. **Notebook 08** — run top-to-bottom (CPU-only, read-only). It renders
   training curves, per-horizon rollout error, the planner table and bar
   charts, per-planner optimization traces, hybrid refinement outcomes, and a
   ranked recommendation.
5. **Commit the executed notebooks** (03 and 08, with outputs) to `main` —
   repo convention: Colab outputs are part of the record.

Completing steps 1–4 on real PointMaze latents also closes the last open
acceptance criterion in the implementation spec ("at least one real-latent
PointMaze run can train a local surrogate and evaluate the three main
planners").

---

## 2. How to read the planner comparison

One structural fact colors everything: **the offline evaluation uses the
global DINO-WM model as both the MPC simulator and the judge.** Episodes are
latent goal-reaching tasks from held-out validation episodes; "where you end
up" is the global model's own prediction. This favors `global_cem` on
distance metrics by construction — it optimizes exactly the quantity it is
scored on. Read the table accordingly:

| Column(s) | What it tells you |
| --- | --- |
| `success_rate`, `mean_normalized_final_distance` | Goal-reaching quality *as judged by the global model*. Normalized distance < 1 means the planner got closer to the goal than the start was; success threshold is 0.5 ("closed at least half the gap"). |
| `mean_reference_final_distance_global` | What the *dataset's true actions* achieve under the same simulator — the calibration line. If planners are far above it, they underperform demonstrated behavior; if the reference itself is large, the global model's rollout error is the bottleneck. |
| `mean_planning_wall_time_sec` | The headline efficiency number per episode. |
| `total_global_forward_calls` vs `total_backward_steps` | The local/global trade at the heart of the experiment: global CEM pays thousands of expensive global rollouts; local planners pay cheap surrogate backward passes; hybrids pay a CEM budget plus a small refinement. This pair is the fair-comparison currency. |
| `accepted_refinement_rate` (hybrids) | The local-global agreement signal. High acceptance = surrogate gradients usually improve the trusted global cost; low acceptance = the surrogate's gradient directions disagree with the global model (look at `mean_local_global_disagreement` and the refinement-outcomes table in notebook 08). |
| `mean_local_global_disagreement` | Open-loop surrogate drift along the executed trajectory, in projected space. Grows with horizon by nature; what matters is relative magnitude across runs/ablations. |
| `action_bound_violation_count` | Must be 0 (bounds are enforced by construction; nonzero means a bug). |

Planner contrasts and what each isolates:

- `global_cem` vs `local_cem` — same optimizer, different model: pure
  **"small model" error**.
- `local_cem` vs `local_adam`/`local_gd` — same model, different optimizer:
  pure **"gradients vs sampling" effect**.
- `local_gd` vs `local_adam` — plain descent vs adaptive steps through the
  same surrogate.
- `global_cem` vs `hybrid_cem_local_refine` — does cheap local refinement
  improve the trusted global cost on top of the same CEM budget?
- hybrid vs `hybrid_..._global_rescore` — how often refinement *needs* to be
  rejected, i.e. how exploitable the surrogate's gradients are.

The interesting *positive* result for the local/global thesis is not "local
beats global on distance" (the judging setup nearly forbids it) but:
**hybrid matches or improves global CEM's cost at meaningfully lower wall
time / global-call count, with a high acceptance rate.**

---

## 3. The research arc after the first run

In rough order of value:

1. **Real-environment confirmation (the big one).** Offline rankings inherit
   the global model's biases. Before drawing conclusions, evaluate the
   winner(s) in the real MuJoCo PointMaze:
   - baseline: the DINO-WM track's `plan.py` (notebook 02) already does real
     env rollouts for global CEM;
   - follow-up implementation task: wire the local/hybrid planners into a
     real environment MPC loop (the planner interfaces are already
     env-agnostic; what's needed is an env-backed counterpart to the offline
     episode runner in `src/wm_poc/local_global/eval.py`).
2. **Seeds.** Everything so far is `seed0`. Surrogate retraining is cheap;
   planning evaluation is the cost. Two or three seeds give error bars on the
   planner table (`run_name` per seed keeps run dirs separate, e.g.
   `pointmaze_local_full_seed1` with `seed: 1`).
3. **Ablations, if the numbers warrant them** (all already implemented as
   config switches):
   - projection: `grid_pool_linear` (current) vs `mean_pool_linear` — how much
     does coarse spatial structure matter to the surrogate?
   - `model_type: gru_residual` vs `residual_mlp` — does recurrent context
     (velocity inference) reduce multi-step rollout error and disagreement?
   - `local_dim`, `rollout_steps`, `lambda_jacobian > 0` (gradient-smoothness
     regularization) if refinement acceptance is low.
4. **Deferred safeguards** (spec's "later" list) — only if hybrid acceptance
   is low or local planners exploit the surrogate: ensembles with disagreement
   penalties in the planning cost, dataset-magnitude-constrained CEM elites,
   or FOG-style online surrogate fine-tuning on global-model rollouts (the
   "model mini-epoch" that keeps the surrogate accurate under the current
   action distribution).
5. **PushT** — deliberately deferred. Prerequisite is a PushT latent cache,
   which is a DINO-WM-track item (the upstream latent pipeline currently
   supports `point_maze` only). Once the cache and a PushT global checkpoint
   exist, the local/global side needs only a config file (action dim/bounds,
   frameskip, run name) — contact-rich dynamics are exactly where the
   global re-score safeguard should earn its keep.

---

## 4. Why notebook 03 ends with the "Next Commands" cell

Three reasons, in decreasing order of weight:

1. **It is the escape hatch from the notebook.** The notebook is a thin
   dispatcher; every heavy operation is a plain script gated on
   `RUN_LOCAL_GLOBAL=1`. The cell prints the exact terminal commands with the
   session's resolved paths substituted in, so the same pipeline can run
   without the notebook: from a Colab terminal, over SSH, in `tmux` on any
   machine with the repo + Drive paths. It is the one place the notebook
   states precisely what it executes on your behalf.
2. **The implementation spec requires it** (notebook cell list, item 15:
   "print copy-paste commands for full PointMaze and PushT runs"), mirroring
   the repo's other operational notebooks.
3. **It records the PushT deferral where you would ask "what's next?"**

Honest caveat: now that the launch cells are fully self-gating and resumable,
this cell is the least load-bearing in the notebook — the gated cells already
print their commands as dry-runs. It stays as cheap documentation; folding the
PushT note into the summary cell and dropping the rest would also be
defensible.

---

## 5. Quick reference: interruption behavior

| Stage | Interrupted by cap/disconnect | On re-run |
| --- | --- | --- |
| Latent cache build | partial episode files | precompute skips complete episodes, finishes the rest (idempotent) |
| Surrogate training | loses ≤ `save_every`=1000 steps | resumes from `local_latest.pt` (model + optimizer + step) |
| Planning evaluation | capped planner writes `summary_partial.json` after the episodes it finished | resumes per episode from `episodes.jsonl`; planners with `summary.json` at >= the configured `num_episodes` are skipped |
| Summary / notebook 08 | stateless | just re-run |
