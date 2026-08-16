#!/usr/bin/env bash
set -euo pipefail

SITE="${1:-utah_2019}"
GPU="${2:-0}"
LOG_ROOT="${3:-logs}"

export PYTHONPATH=.
export MPLBACKEND=Agg
export LD_LIBRARY_PATH=/home/anaconda3/lib:${LD_LIBRARY_PATH:-}

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

case "${SITE}" in
  pohang)
    BASE_CFG="configs/train/base_pohang_arch_best_silu.yaml"
    STAGE_CFG="configs/train/final_pohang_best_silu.yaml"
    FINETUNE_DIR="runs/final_silu/pohang_best/finetune/pohang"
    ;;
  utah_2019)
    BASE_CFG="configs/train/base_utah_2019_arch_best_silu.yaml"
    STAGE_CFG="configs/train/final_utah_2019_best_silu.yaml"
    FINETUNE_DIR="runs/final_silu/utah_2019_best/finetune/base_utah_2019"
    ;;
  utah_2023)
    BASE_CFG="configs/train/base_utah_2023_arch_best_silu.yaml"
    STAGE_CFG="configs/train/final_utah_2023_best_silu.yaml"
    FINETUNE_DIR="runs/final_silu/utah_2023_best/finetune/base_utah_2023"
    ;;
  *)
    echo "[ERROR] unsupported SITE: ${SITE}"
    echo "[ERROR] expected one of: pohang, utah_2019, utah_2023"
    exit 1
    ;;
esac

mkdir -p "${LOG_ROOT}"
LOG_FILE="${LOG_ROOT}/finetune_silu__${SITE}.log"

{
  echo "============================================================"
  echo "[SILU FINETUNE START] $(date '+%F %T') site=${SITE} gpu=${GPU}"
  echo "python_bin=${PYTHON_BIN}"
  echo "base_cfg=${BASE_CFG}"
  echo "finetune_stage_cfg=${STAGE_CFG}"
  echo "test_stage_cfg=configs/train/test.yaml"
  echo "analyze_stage_cfg=configs/train/analyze.yaml"
  echo "finetune_dir=${FINETUNE_DIR}"
  echo "============================================================"
} | tee -a "${LOG_FILE}"

echo "[SILU] finetune" | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg "${STAGE_CFG}" >> "${LOG_FILE}" 2>&1

echo "[SILU] test" | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg configs/train/test.yaml >> "${LOG_FILE}" 2>&1

echo "[SILU] analyze" | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -m src.detection.analysis.analyze \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg configs/train/analyze.yaml >> "${LOG_FILE}" 2>&1

echo "[SILU] tsne" | tee -a "${LOG_FILE}"
PYTHON_BIN="${PYTHON_BIN}" bash scripts/gpu/run_tsne.sh "${GPU}" "${FINETUNE_DIR}" >> "${LOG_FILE}" 2>&1

echo "[DONE] SiLU pipeline completed. site=${SITE}" | tee -a "${LOG_FILE}"
