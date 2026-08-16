#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-4}"

export PYTHONPATH=.
export MPLBACKEND=Agg
export LD_LIBRARY_PATH=/home/anaconda3/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/anaconda3/bin/python3.9}"
BASE_CFG=".tmp_logenv_cross_site_reconst_pohang_to_utah_2023/base.yaml"
FINETUNE_CFG=".tmp_logenv_cross_site_reconst_pohang_to_utah_2023/finetune_frac0p5.yaml"
EVAL_CFG=".tmp_logenv_cross_site_reconst_pohang_to_utah_2023/eval_frac0p5.yaml"
TEST_CFG=".tmp_logenv_cross_site_reconst_pohang_to_utah_2023/test.yaml"
LOG_ROOT="logs/filter_rms_cross_site_reconst_gpu024_v1"
LOG_FILE="${LOG_ROOT}/pohang_to_utah_2023__frac0p5_rerun.log"

mkdir -p "${LOG_ROOT}"

{
  echo "============================================================"
  echo "[RERUN] $(date '+%F %T') pohang->utah_2023 frac0p5 gpu=${GPU}"
  echo "base_cfg=${BASE_CFG}"
  echo "finetune_cfg=${FINETUNE_CFG}"
  echo "eval_cfg=${EVAL_CFG}"
  echo "test_cfg=${TEST_CFG}"
  echo "============================================================"
} | tee -a "${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg "${FINETUNE_CFG}" \
  --exp_suffix frac0p5 >> "${LOG_FILE}" 2>&1

echo "[TEST] pohang->utah_2023 frac0p5" | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
  --base_cfg "${EVAL_CFG}" \
  --stage_cfg "${TEST_CFG}" >> "${LOG_FILE}" 2>&1

echo "[DONE] pohang->utah_2023 frac0p5 rerun" | tee -a "${LOG_FILE}"
