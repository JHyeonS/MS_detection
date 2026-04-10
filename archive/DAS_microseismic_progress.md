# DAS Microseismic Detection Project – Progress Summary

## 1. Dataset Construction
- Pohang: noise ~400 / event ~400 / unlabeled ~9800
- Utah: noise ~400 / event ~120 / unlabeled ~8000
- Severe class imbalance + few-shot scenario

## 2. Pipeline
- Bandpass + robust normalization
- CSV-based split
- Label efficiency sampling

## 3. Model
- CNN Encoder + FC + Anomaly branch
- Loss = classification + center-based anomaly

## 4. Core Problem
- Domain shift (Pohang ↔ Utah)
- Fixed center c → performance collapse

## 5. Experiments

### (1) pretrained + fixed center
- ❌ domain generalization 실패

### (2) no pretrain
- random init + dynamic center

### (3) pretrained + dynamic center
- pretrained encoder + target-based center
- center updated every epoch

## 6. Multi-base Setup
- base_1: pohang only
- base_2: utah only
- base_3: joint
- base_4: pohang→utah
- base_5: utah→pohang

## 7. Label Efficiency
fractions:
- 0.5 / 0.2 / 0.1 / 0.05

## 8. Key Insight
- pretraining alone is not enough
- center adaptation is critical

## 9. Next
- compare 3 setups
- generate plots
- write SEG abstract

## 10. Status
- dataset ✅
- pipeline ✅
- multi-base ✅
- experiments 진행중 🚧
