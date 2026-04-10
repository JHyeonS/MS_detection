
# MS_Detection Project – Current Process Summary

## 1. Project Goal

Build a **DAS-based microseismic detection system** using a deep learning pipeline consisting of:

1. Self-supervised **CAE pretraining**
2. **Fine-tuning with Deep SAD + classifier (FCL)**
3. **Test evaluation using anomaly score and classifier output**

The system is designed for experiments on the **Pohang DAS dataset** and is extendable to other DAS datasets such as Utah FORGE.

---

# 2. Overall Pipeline

The training pipeline consists of three stages.

```
Pretrain  →  Finetune  →  Test
```

### Stage 1: Pretrain

Objective:
- Train **CNN encoder** using reconstruction CAE.
- Learn general DAS signal representation.

Model:
```
Input DAS segment
      ↓
CNN Encoder
      ↓
Latent Vector
      ↓
CNN Decoder
      ↓
Reconstruction
```

Loss:
```
Reconstruction Loss
```

Output:
```
runs/<run_root>/pretrain/<experiment>/
    best.pt
    last.pt
```

Checkpoint includes:
- model_state_dict
- optimizer_state_dict
- training loss
- hypersphere center `c`

---

### Stage 2: Finetune

Objective:
- Train anomaly detection model using pretrained encoder.

Model:
```
Input
  ↓
CNN Encoder (pretrained)
  ↓
Latent Vector z
  ↓
FC layer → classification logit
```

Loss structure:

```
loss_total = loss_fcl + λ * loss_anomaly
```

Where

```
loss_anomaly = ||z - c||²
```

Meaning:
- normal samples pulled toward center
- anomaly samples pushed away

Output:

```
runs/<run_root>/finetune/<experiment>/
    best.pt
    last.pt
```

Checkpoint contains:

```
model_state_dict
optimizer_state_dict
center_c
metrics
```

---

### Stage 3: Test

Objective:
- Evaluate final model on unseen dataset.

Process:

```
Input → Encoder → latent z
                    ↓
         anomaly_score = ||z - c||²
                    ↓
         classifier probability
```

Outputs:

```
runs/<run_root>/test/<experiment>/
    predictions.csv
    metrics.json
```

Example result:

```
accuracy  = 0.73
precision = 0.98
recall    = 0.38
f1        = 0.56
```

Interpretation:

Model is **very conservative**:
- almost no false positives
- but misses many events

Confusion matrix:

```
TP = 50
TN = 164
FP = 1
FN = 79
```

Meaning:
- high precision
- low recall

---

# 3. Dataset Structure

Metadata CSV files are organized under a split directory.

```
data/0312/metadata/experiments/stage1_pohang_only/

    pretrain.csv
    train.csv
    val.csv
    test.csv
```

Each CSV contains:

```
npy_path
label
(optional metadata fields)
```

The dataloaders automatically combine

```
split_dir + filename
```

to load the correct file.

---

# 4. Code Structure

Project directory:

```
MS_Detection

configs/train/
    base.yaml
    pretrain.yaml
    train.yaml
    test.yaml

scripts/
    pretrain.sh
    train.sh
    test.sh

src/

    dataset/
        finetune_dataset.py

    dataloader/
        pretrain_dataloader.py
        finetune_dataloader.py
        test_dataloader.py

    models/
        cnn_encoder.py
        pretrain_reconstruction.py

    training/
        trainer_pretrain.py
        trainer_finetune.py
        trainer_test.py

    utils/
        device.py
        config_io.py
```

---

# 5. Run Output Structure

All experiments are saved under:

```
runs/<run_root>/
```

Example:

```
runs/0313/

    pretrain/
        stage1_pohang_only/

    finetune/
        stage1_pohang_only/

    test/
        stage1_pohang_only/
```

Each experiment folder also stores config snapshots.

```
merged_config.yaml
base_config.yaml
stage_config.yaml
run_metadata.json
```

This ensures **experiment reproducibility**.

---

# 6. Device Management

GPU is configured via config:

```
device:
  type: cuda
  ids: [1]
```

Handled by:

```
src/utils/device.py
```

This sets

```
CUDA_VISIBLE_DEVICES
```

automatically.

---

# 7. Git Workflow

Development workflow:

```
local machine → github → server
```

Steps:

```
git add -A
git commit -m "update"
git push
```

Server:

```
git pull
```

---

# 8. Current Status

Completed:

✓ dataset structure finalized  
✓ pretrain pipeline implemented  
✓ finetune pipeline implemented  
✓ anomaly loss implemented  
✓ test pipeline implemented  
✓ dataloaders implemented  
✓ config snapshot system implemented  
✓ GPU device config system implemented  

Current working pipeline:

```
pretrain.sh
train.sh
test.sh
```

---

# 9. Next Possible Improvements

Potential research improvements:

1. Threshold tuning for anomaly score
2. ROC / PR curve evaluation
3. Better anomaly score calibration
4. Multi-dataset training (Utah + Pohang)
5. Few-shot learning for rare events
6. Hard negative mining
7. transformer-based encoder extension

---

# 10. Key Insight So Far

The current model behaves like:

```
High precision
Low recall
```

Meaning:

- event detection is reliable
- but sensitivity needs improvement

Possible fixes:

- adjust anomaly threshold
- tune λ in anomaly loss
- improve representation learning
