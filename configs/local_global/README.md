# Local/Global Planning Configs

Configs for the local/global track: the DINO-WM latent world model is the
trusted **global** forward model, a small differentiable surrogate trained on
cached DINO latents is the **local** model, and hybrid planners combine global
CEM search with local gradient refinement (plus global re-scoring). The full
implementation brief lives at `LOCAL_GLOBAL_DINO_WM_IMPLEMENTATION_SPEC.md`
in the repository root.

Conventions match `configs/dino_wm`: single-parent `extends:` inheritance and
`${oc.env:VAR,default}` placeholders resolved at load time. A detailed write-up
of every implementation technique (data layer, surrogate, planners, evaluation,
notebook/Colab engineering, testing) lives in
`docs/local_global_techniques.md`.

| File | Purpose |
| --- | --- |
| `base.yaml` | Full schema with PointMaze defaults; everything else extends it. |
| `smoke_synthetic.yaml` | CPU-only end-to-end smoke on a generated point-mass task (`scripts/local_global/run_smoke.sh`). |
| `smoke_pointmaze.yaml` | Tiny PointMaze run; uses the real latent cache when `LG_LATENT_CACHE_DIR`/`LG_ACTION_DATA_DIR` point at it, synthetic latents otherwise. |
| `pointmaze_surrogate_a100.yaml` | **The** full PointMaze experiment (grid-pool projection, full planner set), throughput-tuned for an A100; each stage stays well under 2h. |
| `pointmaze_surrogate_t4.yaml` | The *same* experiment on a 16 GB T4: extends the A100 file and overrides throughput knobs only (rollout chunk size, workers). Shares the run dir, so a run continues across GPU tiers; wall-clock caps + planner-level resume handle slower sessions. |

Key environment variables: `LG_RUN_ROOT` (run/artifact root, Drive on Colab),
`LG_GLOBAL_MODEL_NAME` / `LG_GLOBAL_MODEL_EPOCH` (which DINO-WM training run
provides the global checkpoint), `LG_LATENT_CACHE_DIR`, `LG_ACTION_DATA_DIR`,
`LG_SMOKE_ROOT`, plus the DINO-WM track's `DINO_WM_REPO`, `DINO_WM_DATA_ROOT`,
`DINO_WM_FEATURE_CACHE`, `DINO_CKPT_ROOT`. Heavy script runs are gated on
`RUN_LOCAL_GLOBAL=1`; `--smoke` and `--dry-run` are always allowed.

PushT is intentionally not configured yet: the latent cache pipeline currently
supports `point_maze` only, and PointMaze must pass a real-latent run first.
