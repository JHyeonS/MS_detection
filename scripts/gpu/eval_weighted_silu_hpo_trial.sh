#!/usr/bin/env bash
set -euo pipefail

TRIAL_SUFFIX="${1:?usage: bash scripts/gpu/eval_weighted_silu_hpo_trial.sh <trial_suffix> [gpu] [logs]}"
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

HPO_ROOT="./runs/hpo/finetune_utah_2023_silu_bce_weight_targeted"
SOURCE_BASE_CFG="configs/train/base_utah_2023_arch_best_silu.yaml"
TMP_DIR=".tmp_weighted_silu_hpo_eval"
mkdir -p "${LOG_ROOT}" "${TMP_DIR}"

BASE_CFG="${TMP_DIR}/utah_2023_${TRIAL_SUFFIX}_base.yaml"
TEST_CFG="${TMP_DIR}/utah_2023_${TRIAL_SUFFIX}_test.yaml"
FINETUNE_DIR="${HPO_ROOT}/finetune/base_utah_2023__${TRIAL_SUFFIX}"
LOG_FILE="${LOG_ROOT}/eval_weighted_silu_hpo__utah_2023__${TRIAL_SUFFIX}.log"

"${PYTHON_BIN}" - "${SOURCE_BASE_CFG}" "${HPO_ROOT}" "${TRIAL_SUFFIX}" "${BASE_CFG}" <<'PY'
import sys
import yaml

src, run_root, suffix, outp = sys.argv[1:5]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
cfg.setdefault("data", {})["experiment"] = f"{cfg['data']['experiment']}__{suffix}"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

"${PYTHON_BIN}" - "${TEST_CFG}" <<'PY'
import sys
import yaml

outp = sys.argv[1]
with open("configs/train/test.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("test", {})["threshold_metric"] = "balanced_acc"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

{
  echo "============================================================"
  echo "[EVAL WEIGHTED SILU HPO TRIAL START] $(date '+%F %T') trial=${TRIAL_SUFFIX} gpu=${GPU}"
  echo "python_bin=${PYTHON_BIN}"
  echo "base_cfg=${BASE_CFG}"
  echo "test_cfg=${TEST_CFG}"
  echo "finetune_dir=${FINETUNE_DIR}"
  echo "============================================================"
} | tee -a "${LOG_FILE}"

echo "[HPO TRIAL EVAL] test" | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg "${TEST_CFG}" >> "${LOG_FILE}" 2>&1

echo "[HPO TRIAL EVAL] analyze" | tee -a "${LOG_FILE}"
"${PYTHON_BIN}" -m src.detection.analysis.analyze \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg configs/train/analyze.yaml >> "${LOG_FILE}" 2>&1

echo "[HPO TRIAL EVAL] tsne" | tee -a "${LOG_FILE}"
PYTHON_BIN="${PYTHON_BIN}" bash scripts/gpu/run_tsne.sh "${GPU}" "${FINETUNE_DIR}" >> "${LOG_FILE}" 2>&1

echo "[DONE] weighted SiLU HPO trial eval completed. trial=${TRIAL_SUFFIX}" | tee -a "${LOG_FILE}"
