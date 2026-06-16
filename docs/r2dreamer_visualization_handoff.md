# R2-Dreamer Visualization Handoff

## Summary

Add a post-hoc visualization track for the existing R2-Dreamer experiments. This should not require a total rebuild of Dreamer/R2-Dreamer and should not modify the core training loop at first.

The goal is to visualize trained checkpoints beyond scalar loss/evaluation curves, starting with reliable policy rollout videos and then adding world-model-specific visualizations such as latent trajectories and, for vision runs, reconstruction or imagined-rollout views if the upstream R2-Dreamer model exposes the needed decoder/model APIs cleanly.

## Current Repository Context

- Repository: `wm-prediction`
- Python package: `wm_poc`
- Main Drive root: `/content/drive/MyDrive/wm_poc`
- Existing T4/proprio run folder:
  - `/content/drive/MyDrive/wm_poc/logs/r2dreamer/walker_walk_to_run_t4_r2_proprio_12m_seed0`
- Existing A100/vision preset:
  - `configs/r2dreamer/three_way_walker_walk_to_run_a100_r2_vision25m.yaml`
- Existing T4/proprio preset:
  - `configs/r2dreamer/three_way_walker_walk_to_run_t4_r2_proprio.yaml`
- Results notebook (metrics, rollout videos, latent analysis):
  - `notebooks/06_r2dreamer_results.ipynb`

## Completed T4 Track

The T4 `dmc_proprio` source/fine-tune/scratch track has completed and produced checkpoints/metrics. The source-base run completed cleanly, and checkpoint retention is expected to keep only the last 8 interval checkpoints due to:

```yaml
checkpoint_keep: 8
```

The dedicated results notebook should remain results-only. It reads metrics from Drive and creates:

- source evaluation curve,
- target fine-tune vs scratch evaluation curve,
- fine-tune advantage curve,
- rolling training episode score,
- optimization diagnostics,
- CSV summaries and metric inventory.

## Requested New Work

Create a separate visualization/evaluation track for trained R2-Dreamer checkpoints. Prefer post-hoc scripts/notebooks first, not training-loop integration.

This work now lives in the consolidated results notebook:

```text
notebooks/06_r2dreamer_results.ipynb
```

Suggested supporting scripts, if useful:

```text
scripts/r2dreamer/render_policy_rollouts.py
scripts/r2dreamer/extract_latent_trajectories.py
scripts/r2dreamer/plot_latent_trajectories.py
```

Do not add training jobs, dependency installs, large downloads, datasets, or checkpoints to Git.

## Visualization Priorities

1. Policy rollout videos
   - Load `latest.pt` or a chosen interval checkpoint.
   - Run the trained policy in the relevant DMC task.
   - Render videos to Drive.
   - This should work for both `dmc_proprio` and `dmc_vision`, because videos come from the environment renderer rather than the observation type.

2. Latent trajectory plots
   - Run one or more episodes.
   - Extract recurrent latent/world-model states if feasible from the upstream R2-Dreamer agent.
   - Produce 2D PCA plots first.
   - Optionally produce 3D PCA plots or simple animations.
   - Color trajectories by step, reward, velocity, run name, or checkpoint.

3. Fine-tune vs scratch latent comparison
   - Load `target_finetune/latest.pt` and `target_scratch/latest.pt`.
   - Collect latent trajectories on the same target task.
   - Compare projected latent paths in a shared PCA basis.

4. Vision-specific world-model views
   - For `dmc_vision`, investigate whether the model exposes decoded/reconstructed image predictions.
   - If available, produce grids/videos comparing:
     - real observed frames,
     - reconstructed frames,
     - imagined/predicted future frames.
   - If not available cleanly, document the limitation and keep the first vision visualization to policy videos plus latent trajectories.

## Recommended Implementation Shape

Keep this post-hoc and checkpoint-driven:

```text
checkpoint -> load model/agent -> run eval episodes -> save videos/latents/plots to Drive
```

Avoid changing the training loop initially. Training-loop integration can be a later phase if post-hoc visualization works:

```text
eval interval -> save checkpoint -> render short rollout -> save diagnostic plots
```

## Expected Inputs

The notebook/scripts should accept or define:

```bash
export WM_POC_DRIVE_ROOT=/content/drive/MyDrive/wm_poc
export R2DREAMER_REPO=/content/external_repos/r2dreamer
export R2_LOG_ROOT=/content/drive/MyDrive/wm_poc/logs/r2dreamer/walker_walk_to_run_t4_r2_proprio_12m_seed0
export R2_FIGURE_DIR=/content/drive/MyDrive/wm_poc/figures/r2dreamer/walker_walk_to_run_t4_r2_proprio_12m_seed0
```

Important checkpoint examples:

```text
$R2_LOG_ROOT/source_base/latest.pt
$R2_LOG_ROOT/target_finetune/latest.pt
$R2_LOG_ROOT/target_scratch/latest.pt
```

For interval checkpoints:

```text
$R2_LOG_ROOT/source_base/checkpoints/step_000500000.pt
$R2_LOG_ROOT/target_finetune/checkpoints/step_000250000.pt
$R2_LOG_ROOT/target_scratch/checkpoints/step_000250000.pt
```

## Expected Outputs

Save generated artifacts to Drive, not Git:

```text
/content/drive/MyDrive/wm_poc/videos/r2dreamer/<run-name>/rollouts/
/content/drive/MyDrive/wm_poc/figures/r2dreamer/<run-name>/visualizations/
/content/drive/MyDrive/wm_poc/reports/r2dreamer/<run-name>/visualizations/
```

Useful output names:

```text
source_base_policy_rollout.mp4
target_finetune_policy_rollout.mp4
target_scratch_policy_rollout.mp4
target_latent_pca_2d.png
target_latent_pca_3d.png
target_finetune_vs_scratch_latent_pca.png
vision_reconstruction_grid.png
vision_imagined_rollout_grid.png
```

## Constraints

- Do not run long training jobs.
- Do not install dependencies unless explicitly approved.
- Do not download large datasets.
- Do not commit videos, checkpoints, datasets, TensorBoard logs, or model weights.
- Save generated media and figures to Google Drive.
- Keep code PyTorch-based.
- Keep paths configurable through environment variables.
- Prefer post-hoc visualization from existing checkpoints before training-loop integration.
- Preserve existing notebooks and scripts unless changes are directly required.

## Validation

For repository-side changes, run lightweight checks only:

```bash
python scripts/verify_environment.py --cpu-only
python -m json.tool notebooks/06_r2dreamer_results.ipynb >/dev/null
pytest -q
```

If adding scripts:

```bash
python -m compileall -q scripts src tests
bash -n scripts/r2dreamer/*.sh
```

Do not run GPU visualization jobs as part of ordinary Codex validation unless explicitly requested. It is acceptable to make the notebook/scripts ready for Colab execution and validate syntax locally.

## Open Technical Questions

- How exactly does upstream `NM512/r2dreamer` restore agent/checkpoint state outside `train.py`?
- Can the policy be called cleanly in eval mode from a post-hoc script, or does the script need to reuse pieces of upstream `train.py`/trainer code?
- Are decoded image reconstructions available for `dmc_vision`, or does the model only expose latent/value/reward predictions cleanly?
- Which latent representation is most stable to extract: RSSM stochastic state, deterministic recurrent state, concatenated features, or encoder output?

## Preferred First Milestone

Implement policy rollout videos for the completed T4 `dmc_proprio` checkpoints:

```text
source_base/latest.pt
target_finetune/latest.pt
target_scratch/latest.pt
```

Then add latent PCA plots if checkpoint loading and agent inference are clean.

Only after that, investigate `dmc_vision` reconstructions or imagined rollouts.
