# MS_Detection Project Summary

## 1. Project Goal

The goal of this project is to build a **microseismic detection model using DAS (Distributed Acoustic Sensing)** data.

The training pipeline consists of:

1. **Self-supervised pretraining**
2. **Supervised fine-tuning**
3. **Label efficiency experiments**

Two types of pretraining will be explored:

- **Reconstruction pretraining (Autoencoder-style)**
- **Contrastive learning pretraining**

---

# 2. Dataset

Two DAS datasets are used.

## Pohang Dataset

Label distribution:

| Label | Meaning | Count (approx) |
|------|------|------|
| 0 | Noise | ~750 |
| 1 | Event | ~407 |
| 2 | Unlabeled | ~2250 |

Directory:


data/pohang/
├─ 0_noise
├─ 1_event
└─ 2_unlabel


---

## UTAH Dataset

Label distribution:

| Label | Meaning | Count (approx) |
|------|------|------|
| 0 | Noise | ~900 |
| 1 | Event | ~135 |
| 2 | Unlabeled | ~3400 |

Directory:


data/utah/
├─ 0_noise
├─ 1_event
└─ 2_unlabel


---

# 3. Dataset Metadata

All dataset split information is stored in:


data/metadata/


Important folders:


data/metadata/
├─ input_segment_plans
└─ experiments


Example experiment folder:


data/metadata/experiments/stage1_pohang_only/
├─ pretrain.csv
├─ train.csv
├─ val.csv
└─ test.csv


Meaning:

| File | Purpose |
|-----|------|
| pretrain.csv | self-supervised training |
| train.csv | supervised training |
| val.csv | validation |
| test.csv | evaluation |

---

# 4. Project Structure

Current project directory:


MS_Detection/
├─ config
│ ├─ base.yaml
│ ├─ pretrain.yaml
│ ├─ train.yaml
│ └─ test.yaml
│
├─ scripts
│
├─ src
│ ├─ models
│ ├─ training
│ ├─ utils
│ ├─ dataset
│ └─ dataloader
│
└─ data
├─ pohang
├─ utah
└─ metadata


---

# 5. Configuration Files

Located in:


config/


Files:

| File | Purpose |
|----|----|
| base.yaml | common settings |
| pretrain.yaml | pretraining settings |
| train.yaml | fine-tuning settings |
| test.yaml | evaluation settings |

---

# 6. Training Pipeline

The full training pipeline will be:


Dataset CSV
↓
Dataset Loader
↓
Dataloader
↓
Pretraining
↓
Fine-tuning
↓
Evaluation


---

# 7. Pretraining Methods

Two pretraining approaches will be implemented.

## 1) Reconstruction Pretraining

Model:


Encoder + Decoder


Goal:


Reconstruct input DAS signal


Loss:


MSE / L1


---

## 2) Contrastive Pretraining

Model:


Encoder + Projection Head


Input:


Two augmented views


Loss:


NT-Xent / InfoNCE


---

# 8. Label Efficiency Experiment

Fine-tuning will evaluate performance using different labeled data fractions.

Example:


1%
5%
10%
25%
50%
100%


This will measure:


Label efficiency of the pretrained encoder


---

# 9. Next Tasks (Implementation Roadmap)

The following components must be implemented.

---

## Step 1 — Dataset Loader

Location:


src/dataset/


Files:


pretrain_dataset.py
finetune_dataset.py


Responsibilities:

- Load `.npy`
- Apply normalization
- Return tensors

---

## Step 2 — Dataloader

Location:


src/dataloader/


File:


build_dataloader.py


Responsibilities:

- Build PyTorch DataLoader
- Support:


batch size
num workers
label fraction
class balancing


---

## Step 3 — Pretraining Models

Location:


src/models/


Files:


encoder.py
pretrain_reconstruction.py
pretrain_contrastive.py


---

## Step 4 — Pretraining Trainer

Location:


src/training/


File:


trainer_pretrain.py


Responsibilities:

- training loop
- optimizer
- logging
- checkpoint saving

Supports:


reconstruction mode
contrastive mode


---

## Step 5 — Fine-tuning Trainer

Location:


src/training/


File:


trainer_finetune.py


Responsibilities:

- supervised training
- evaluation
- label efficiency experiment

---

# 10. Immediate Next Task

The next implementation step is:


src/training/trainer_pretrain.py


This trainer must support:


reconstruction pretraining
contrastive pretraining


using the dataset splits stored in:


data/metadata/experiments/


---

# End of Summary