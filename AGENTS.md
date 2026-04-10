# AGENTS.md

## Project structure
- src/detection: training / evaluation pipeline
- src/models: model definitions
- scripts/gpu: GPU-side entry scripts
- configs/train: training configs
- configs/experiments: experiment-specific configs
- configs/system: machine-specific path configs

## Rules
- Do not commit runs/, logs/, outputs/, checkpoints/, data/, output_npy/
- Prefer config-driven paths over hardcoded absolute paths
- New GPU launchers go into scripts/gpu
- Keep training code under src/detection/training
- Keep dataloader code under src/detection/dataloader
- Keep dataset code under src/detection/dataset
- Use executable Python scripts when possible
