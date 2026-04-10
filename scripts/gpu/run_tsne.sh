#!/bin/bash

# Usage:
# bash scripts/run_tsne.sh <GPU_ID> <EXP_PATH>

GPU_ID=$1
EXP_PATH=$2

if [ -z "$GPU_ID" ]; then
  GPU_ID=0
fi

if [ -z "$EXP_PATH" ]; then
  echo "[ERROR] Usage: bash scripts/run_tsne.sh <GPU_ID> <EXP_PATH>"
  exit 1
fi

# TSNE는 CPU로 돌릴 거라 GPU 숨김
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=.

BASE_CFG=${EXP_PATH}/config_snapshot/base_config.yaml
STAGE_CFG=${EXP_PATH}/config_snapshot/stage_config.yaml
CKPT=${EXP_PATH}/best.pt
OUT_DIR=${EXP_PATH}/tsne

echo "[INFO] EXP_PATH=${EXP_PATH}"
echo "[INFO] BASE_CFG=${BASE_CFG}"
echo "[INFO] STAGE_CFG=${STAGE_CFG}"
echo "[INFO] CKPT=${CKPT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"

if [ ! -f "${BASE_CFG}" ]; then
  echo "[ERROR] base config not found: ${BASE_CFG}"
  exit 1
fi

if [ ! -f "${STAGE_CFG}" ]; then
  echo "[ERROR] stage config not found: ${STAGE_CFG}"
  exit 1
fi

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] checkpoint not found: ${CKPT}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

python src/analysis/tsne_splits.py \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg "${STAGE_CFG}" \
  --ckpt "${CKPT}" \
  --out_dir "${OUT_DIR}" \
  --max_per_group 200 \
  --tsne_perplexity 30 \
  --pca_dim 50 \
  --seed 42

echo "[DONE] TSNE finished"