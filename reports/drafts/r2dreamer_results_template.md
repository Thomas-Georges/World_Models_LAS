# DreamerV3/R2-Dreamer fine-tuning experiment

## Purpose

This experiment demonstrates checkpoint handling, GPU training, task adaptation, and fine-tuning versus scratch training for a PyTorch world-model agent.

## Hardware

- GPU:
- VRAM:
- PyTorch:
- CUDA:
- Runtime:
- Notes:

## External code

- r2dreamer remote:
- r2dreamer commit:
- Patch applied:
- Project repo commit:

## Experiment

Run preset:
Config path:
Source task:
Target task:
Observation mode:
Model size:
Representation objective:
Source training steps:
Target fine-tuning steps:
Target scratch steps:
Seed:

## Presets

| Config | Intended GPU | Obs | Model | Rep loss | Source steps | Target steps | Train ratio | Env workers | Eval eps |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `debug_walker_walk_to_run.yaml` | any | `dmc_proprio` | `size12M` | `dreamer` | 100K | 50K | 16 | 4 | 2 |
| `three_way_walker_walk_to_run_t4_r2_proprio.yaml` | T4 | `dmc_proprio` | `size12M` | `r2dreamer` | 510K | 250K | 64 | 4 | 5 |
| `three_way_walker_walk_to_run_a100_r2_vision25m.yaml` | A100 | `dmc_vision` | `size25M` | `r2dreamer` | 1.01M | 500K | 256 | 8 | 10 |

## Runs

| Run | Initialization | Task | Steps | Final eval score | Best eval score | Wall-clock | Peak VRAM |
|---|---|---|---:|---:|---:|---:|---:|
| source_base | random |  |  |  |  |  |  |
| target_finetune | source_base checkpoint |  |  |  |  |  |  |
| target_scratch | random |  |  |  |  |  |  |

## Main figure

Insert `figures/r2dreamer/finetune_vs_scratch.png`.

Primary figure: `target_finetune` versus `target_scratch` evaluation return over environment steps.

## Observations

- Did fine-tuning learn faster than scratch?
- Did the checkpoint load cleanly?
- Were observation/action spaces compatible?
- Any instability?
- Any GPU or Colab limitations?

## Limitations

This is not a full DreamerV3 benchmark reproduction. It is a controlled proof of competence in training, checkpointing, fine-tuning, and evaluating PyTorch world models.

## Next steps

- Repeat with `dmc_vision`.
- Increase model size on A100.
- Move to local/global predictive-control experiment on PointMaze or PushT.
