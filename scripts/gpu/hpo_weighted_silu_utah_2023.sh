#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-0}"
LOG_ROOT="${2:-logs}"
SPEC="${3:-configs/experiments/hpo/finetune_utah_2023_silu_bce_weight_targeted.yaml}"

export PYTHONPATH=.
export MPLBACKEND=Agg
export LD_LIBRARY_PATH=/home/anaconda3/lib:${LD_LIBRARY_PATH:-}
export MS_JOB_OWNER="${MS_JOB_OWNER:-${USER:-unknown}}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "/home/ted1204/.conda/envs/ms_detection/bin/python" ]]; then
    PYTHON_BIN="/home/ted1204/.conda/envs/ms_detection/bin/python"
  elif [[ -x "/home/anaconda3/bin/python3.9" ]]; then
    PYTHON_BIN="/home/anaconda3/bin/python3.9"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[ERROR] could not resolve PYTHON_BIN"
    exit 1
  fi
fi

mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/hpo_weighted_silu_utah_2023.log"
HPO_ROOT="./runs/hpo/finetune_utah_2023_silu_bce_weight_targeted"

{
  echo "============================================================"
  echo "[HPO WEIGHTED SILU UTAH 2023 START] $(date '+%F %T')"
  echo "python_bin=${PYTHON_BIN}"
  echo "gpu_list=${GPU_LIST}"
  echo "spec=${SPEC}"
  echo "hpo_root=${HPO_ROOT}"
  echo "============================================================"
} | tee -a "${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
PYTHON_BIN="${PYTHON_BIN}" \
"${PYTHON_BIN}" scripts/gpu/hpo_finetune.py \
  --spec "${SPEC}" >> "${LOG_FILE}" 2>&1

"${PYTHON_BIN}" scripts/gpu/rank_hpo_results.py \
  --root "${HPO_ROOT}" \
  --metric best_metric \
  --mode max >> "${LOG_FILE}" 2>&1

echo "[DONE] weighted SiLU Utah 2023 HPO completed" | tee -a "${LOG_FILE}"
echo "[INFO] leaderboard: ${HPO_ROOT}/leaderboard.csv" | tee -a "${LOG_FILE}"
echo "[INFO] ranked    : ${HPO_ROOT}/ranked_hpo_results.csv" | tee -a "${LOG_FILE}"
