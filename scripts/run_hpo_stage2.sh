#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
MODE="${2:-full}"
BASE_CFG="${3:-config/base_stage2_top1.yaml}"
TRAIN_CFG="${4:-config/train_stage2_unfreeze_lr3e-04_aw0p1.yaml}"

TEST_CFG="config/test.yaml"
ANALYZE_CFG="config/analyze.yaml"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODE=${MODE}"
echo "[INFO] BASE_CFG=${BASE_CFG}"
echo "[INFO] TRAIN_CFG=${TRAIN_CFG}"

if [[ "${MODE}" == "full" ]]; then
    bash scripts/train.sh "${BASE_CFG}" "${TRAIN_CFG}"
    bash scripts/test.sh "${BASE_CFG}" "${TEST_CFG}"
    bash scripts/analyze.sh "${BASE_CFG}" "${ANALYZE_CFG}"
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

echo "[DONE] run_hpo_stage2.sh finished."
