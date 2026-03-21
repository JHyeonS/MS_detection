#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
MODE="${2:-full}"

BASE_CFG="${3:-config/base_contrast_4.yaml}"
PRETRAIN_CFG="config/pretrain_contrast.yaml"
TRAIN_CFG="config/train.yaml"
TEST_CFG="config/test.yaml"
ANALYZE_CFG="config/analyze.yaml"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODE=${MODE}"

if [[ "${MODE}" == "full" ]]; then
    bash scripts/pretrain_contrast.sh "${BASE_CFG}" "${PRETRAIN_CFG}"
    bash scripts/train.sh "${BASE_CFG}" "${TRAIN_CFG}"
    bash scripts/test.sh "${BASE_CFG}" "${TEST_CFG}"
    bash scripts/analyze.sh "${BASE_CFG}" "${ANALYZE_CFG}"
elif [[ "${MODE}" == "pretrain" ]]; then
    bash scripts/pretrain_contrast.sh "${BASE_CFG}" "${PRETRAIN_CFG}"
elif [[ "${MODE}" == "train" ]]; then
    bash scripts/train.sh "${BASE_CFG}" "${TRAIN_CFG}"
elif [[ "${MODE}" == "test" ]]; then
    bash scripts/test.sh "${BASE_CFG}" "${TEST_CFG}"
elif [[ "${MODE}" == "analyze" ]]; then
    bash scripts/analyze.sh "${BASE_CFG}" "${ANALYZE_CFG}"
else
    echo "[ERROR] Unknown mode: ${MODE}"
    exit 1
fi

echo "[DONE] run.sh finished."