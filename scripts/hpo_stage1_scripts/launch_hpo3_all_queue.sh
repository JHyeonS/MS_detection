#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
GPU_LIST="${2:-0,1}"
LOG_ROOT="${3:-logs_hpo3_all}"

mkdir -p "${LOG_ROOT}"

bash scripts/launch_hpo3_contrast_queue.sh "${MODE}" "${GPU_LIST}" "${LOG_ROOT}/contrast"
bash scripts/launch_hpo3_reconst_queue.sh "${MODE}" "${GPU_LIST}" "${LOG_ROOT}/reconst"

echo "[DONE] All hpo3 queues finished."
