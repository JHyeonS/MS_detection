#!/bin/bash

# =========================================
# Usage:
# bash scripts/run_tsne.sh <GPU_ID> <EXP_NAME>
# =========================================

GPU_ID=$1
EXP_NAME=$2

if [ -z "$GPU_ID" ]; then
  GPU_ID=0
fi

if [ -z "$EXP_NAME" ]; then
  echo "[ERROR] Usage: bash scripts/run_tsne.sh <GPU_ID> <EXP_NAME>"
  exit 1
fi

# =========================================
# 환경 설정
# =========================================
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONPATH=.

echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] PYTHONPATH=$PYTHONPATH"

# =========================================
# 경로 설정
# =========================================
RUN_ROOT=runs/stage_label_efficiency_with_fixed_center/label_efficiency_stage1_utah_only/finetune

BASE_CFG=${RUN_ROOT}/${EXP_NAME}/config_snapshot/base_config.yaml
STAGE_CFG=${RUN_ROOT}/${EXP_NAME}/config_snapshot/stage_config.yaml

CKPT=${RUN_ROOT}/${EXP_NAME}/best.pt
OUT_DIR=${RUN_ROOT}/${EXP_NAME}/tsne

mkdir -p ${OUT_DIR}

echo "[INFO] EXP_NAME: $EXP_NAME"
echo "[INFO] CKPT: $CKPT"
echo "[INFO] OUT_DIR: $OUT_DIR"

# =========================================
# 실행
# =========================================
python src/analysis/tsne_splits.py \
  --base_cfg ${BASE_CFG} \
  --stage_cfg ${STAGE_CFG} \
  --ckpt ${CKPT} \
  --out_dir ${OUT_DIR} \
  --max_per_group 200 \
  --tsne_perplexity 30 \
  --seed 42

echo "[DONE] TSNE finished"