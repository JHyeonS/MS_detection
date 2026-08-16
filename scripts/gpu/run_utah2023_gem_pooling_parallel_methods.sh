#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${1:-logs/utah_2023_gem_pooling_parallel_methods}"
RUN_ROOT="${2:-runs/utah_2023_gem_pooling_parallel_methods}"

export PYTHONPATH=.

METHODS=("reconst" "contrast" "reconst_noanom" "scratch")
GPUS=("6" "7" "8" "9")

FRACTIONS="${SITE_STUDY_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
POOLING="${SITE_STUDY_POOLING:-gem}"
POOLING_P="${SITE_STUDY_POOLING_P:-3.0}"
POOLING_CHANNELWISE="${SITE_STUDY_POOLING_CHANNELWISE:-true}"
RUN_ANALYZE="${SITE_STUDY_RUN_ANALYZE:-true}"
RUN_TSNE="${SITE_STUDY_RUN_TSNE:-false}"
RUN_PRETRAIN="${SITE_STUDY_RUN_PRETRAIN:-true}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"

echo "[RUN] Utah 2023 GeM pooling parallel method study"
echo "run_root=${RUN_ROOT}"
echo "log_root=${LOG_ROOT}"
echo "fractions=${FRACTIONS}"
echo "pooling=${POOLING}"
echo "pooling_p=${POOLING_P}"
echo "run_pretrain=${RUN_PRETRAIN}"
echo "run_analyze=${RUN_ANALYZE}"
echo "run_tsne=${RUN_TSNE}"

pids=()

for idx in "${!METHODS[@]}"; do
  method="${METHODS[$idx]}"
  gpu="${GPUS[$idx]}"
  method_log_root="${LOG_ROOT}/${method}"
  method_run_root="${RUN_ROOT}/${method}_worker"

  mkdir -p "${method_log_root}" "${method_run_root}"

  echo "[LAUNCH] method=${method} gpu=${gpu} run_root=${method_run_root}"

  (
    SITE_STUDY_POOLING="${POOLING}" \
    SITE_STUDY_POOLING_P="${POOLING_P}" \
    SITE_STUDY_POOLING_CHANNELWISE="${POOLING_CHANNELWISE}" \
    SITE_STUDY_METHODS="${method}" \
    SITE_STUDY_FRACTIONS="${FRACTIONS}" \
    SITE_STUDY_RUN_PRETRAIN="${RUN_PRETRAIN}" \
    SITE_STUDY_RUN_ANALYZE="${RUN_ANALYZE}" \
    SITE_STUDY_RUN_TSNE="${RUN_TSNE}" \
    bash scripts/gpu/site_main_study.sh utah_2023 "${gpu}" "${method_log_root}" "${method_run_root}"
  ) > "${method_log_root}/launcher.log" 2>&1 &

  pids+=("$!")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  method="${METHODS[$idx]}"
  gpu="${GPUS[$idx]}"
  if wait "${pid}"; then
    echo "[DONE] method=${method} gpu=${gpu}"
  else
    echo "[ERROR] method=${method} gpu=${gpu} failed. See ${LOG_ROOT}/${method}/launcher.log"
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "[FAILED] one or more method workers failed"
  exit "${status}"
fi

echo "[DONE] all Utah 2023 GeM pooling method workers completed"
echo "[INFO] root: ${RUN_ROOT}"
