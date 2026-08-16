# MS_Detection

Research code for DAS microseismic event detection, reconstruction pretraining,
cross-site transfer learning, and latent-space distribution diagnostics.

This repository is being prepared as the code companion for:

> Preprocessing-Dependent Transferability of DAS Microseismic Detection Under Cross-Site Distribution Shift

## Data Availability

Raw and preprocessed DAS datasets are **not included** in this repository.
The following paths are intentionally ignored by Git:

- `data/`
- `runs/`
- `figures/`
- `temp/`
- `output_npy/`
- checkpoint files such as `*.pt`, `*.pth`, and `*.ckpt`
- raw acquisition files such as `*.tdms`, `*.sgy`, and `*.segy`

The code expects metadata CSV files and preprocessed `.npy` windows to be
available locally. Users should update the dataset paths in `configs/` before
running experiments.

## Repository Layout

```text
configs/
  experiments/        Experiment and HPO configuration files
  system/             Machine-specific path/device templates
  train/              Base, pretraining, fine-tuning, test, and analysis configs

scripts/gpu/          Shell and Python launchers for training, testing, HPO,
                      plotting, and result summarization

src/detection/
  analysis/           Evaluation and latent-analysis utilities
  dataloader/         DataLoader builders
  dataset/            Dataset definitions and preprocessing helpers
  training/           Pretraining, fine-tuning, and test entrypoints
  utils/              Config, device, and visualization helpers

src/models/           CNN encoder and reconstruction model definitions
```

## Main Workflow

Most experiments are config-driven. A typical run uses one base config and one
stage config.

```bash
export PYTHONPATH=.
export MPLBACKEND=Agg
```

Fine-tuning:

```bash
python -m src.detection.training.trainer_finetune \
  --base_cfg configs/train/base_pohang.yaml \
  --stage_cfg configs/train/train.yaml
```

Testing:

```bash
python -m src.detection.training.trainer_test \
  --base_cfg configs/train/base_pohang.yaml \
  --stage_cfg configs/train/test.yaml
```

Analysis:

```bash
python -m src.detection.analysis.analyze \
  --base_cfg configs/train/base_pohang.yaml \
  --stage_cfg configs/train/analyze.yaml
```

## Key Experiment Scripts

- `scripts/gpu/hpo_architecture.py`: architecture HPO
- `scripts/gpu/hpo_finetune.py`: fine-tuning HPO
- `scripts/gpu/run_metadata_v2_safe_rerun.sh`: final site-wise experiment launcher
- `scripts/gpu/run_metadata_v2_cross_reconst_swd.sh`: cross-site reconstruction transfer launcher
- `scripts/gpu/plot_controlled_latent_diagnosis_axes.py`: controlled latent diagnosis figure generation
- `scripts/gpu/plot_case1_fixed_filter_indomain_vs_cross.py`: representative site-shift diagnosis figure generation

## Reproducibility Notes

The final experiments used:

- three sites: Pohang, Utah FORGE 2019, and Utah FORGE 2023
- three input regimes: Raw, Low-pass + RMS, and Log-envelope
- label fractions: `0.05`, `0.10`, `0.25`, `0.50`, and `1.00`
- fixed random seed: `42`
- CNN encoder with 6 layers, 16 base channels, 512 latent dimensions, batch normalization, ReLU activation, dropout `0.1`, and average pooling

The repository includes code and configuration templates, but not the datasets
or generated experiment outputs. For exact reproduction, prepare local metadata
CSV files and preprocessed input windows following the expected `data/`
directory structure used in the configs.

## Citation

If this repository is used in a publication, cite the archived release DOI.
A `CITATION.cff` file can be added after creating the first public release.

## License

No license has been selected yet. Add a license before public release if reuse
by others should be permitted.
