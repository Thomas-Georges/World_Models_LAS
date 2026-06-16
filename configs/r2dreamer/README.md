# R2-Dreamer Config Templates

These YAML files configure this repository's wrapper scripts. They are not upstream Hydra config files.

Primary starting points:

```text
three_way_walker_walk_to_run.yaml
three_way_walker_walk_to_run_t4_r2_proprio.yaml
three_way_walker_walk_to_run_a100_r2_vision25m.yaml
debug_walker_walk_to_run.yaml
```

That config runs the controlled DMC Proprio comparison:

```text
source_base: dmc_walker_walk from scratch
target_finetune: dmc_walker_run initialized from source_base/latest.pt
target_scratch: dmc_walker_run from scratch with the same target budget
```

Run presets:

| Config | Intended GPU | Obs | Model | Rep loss | Source steps | Target steps | Train ratio | Env workers | Eval eps |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `debug_walker_walk_to_run.yaml` | any | `dmc_proprio` | `size12M` | `dreamer` | 100K | 50K | 16 | 4 | 2 |
| `three_way_walker_walk_to_run_t4_r2_proprio.yaml` | T4 | `dmc_proprio` | `size12M` | `r2dreamer` | 510K | 250K | 64 | 4 | 5 |
| `three_way_walker_walk_to_run_a100_r2_vision25m.yaml` | A100 | `dmc_vision` | `size25M` | `r2dreamer` | 800K | 400K | 224 | 8 | 5 |

The short 100K/50K run is retained only as `debug_walker_walk_to_run.yaml`. The canonical default `three_way_walker_walk_to_run.yaml` mirrors the T4 R2 Proprio preset.

The smoke settings intentionally use `env_num: 1` and `eval_episodes: 0` so Colab does not start the upstream default DMC worker pool. The T4 Proprio preset uses `env_num: 4`; the balanced A100 Vision preset uses `env_num: 8` while keeping the existing size25M config and run names.

For `dmc_proprio`, `disable_image_render: true` avoids unnecessary MuJoCo image rendering. Keep it `false` for `dmc_vision`, where image observations are the model input.

Use Colab Python 3.11 for R2-Dreamer training. The A100 vision configs keep `serial_envs: false` so Python 3.11 uses parallel DMC workers. If a runtime crashes in the multiprocessing render worker, override with `R2_SERIAL_ENVS=true` as a fallback; this preserves real pixels but can slow the run.

Full training configs print `[wm_poc] progress ...` heartbeats before eval/checkpoint boundaries. The default `progress_every: 100` can be overridden with `R2_PROGRESS_EVERY`; set it to `0` to disable progress output.

Smoke configs set `compile: false` so they bypass `torch.compile` and reach the progress heartbeat quickly. Full runs compile by default unless `R2_COMPILE=false` is set.

Full training configs save model-only interval checkpoints at eval boundaries. The T4 scaled preset keeps the most recent 8, the balanced A100 Vision preset keeps 6, the debug config keeps 12, and smoke configs disable interval checkpointing.
