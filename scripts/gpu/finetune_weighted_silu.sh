#!/usr/bin/env bash
set -euo pipefail

SITE="${1:-utah_2023}"
BRANCH="${2:-both}"
GPU="${3:-0}"
LOG_ROOT="${4:-logs}"

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
  utah_2019)
    BASE_CFG="configs/train/base_utah_2019_arch_best_silu.yaml"
    TEMPLATE_STAGE_CFG="configs/train/final_utah_2019_best_silu.yaml"
    EXPERIMENT="base_utah_2019"
    ;;
  utah_2023)
    BASE_CFG="configs/train/base_utah_2023_arch_best_silu.yaml"
    TEMPLATE_STAGE_CFG="configs/train/final_utah_2023_best_silu.yaml"
    EXPERIMENT="base_utah_2023"
    ;;
  pohang)
    BASE_CFG="configs/train/base_pohang_arch_best_silu.yaml"
    TEMPLATE_STAGE_CFG="configs/train/final_pohang_best_silu.yaml"
    EXPERIMENT="pohang"
    ;;
  *)
    echo "[ERROR] unsupported SITE: ${SITE}"
    echo "[ERROR] expected one of: utah_2019, utah_2023, pohang"
    exit 1
    ;;
esac

case "${BRANCH}" in
  bce|anomaly|both|all) ;;
  *)
    echo "[ERROR] unsupported BRANCH: ${BRANCH}"
    echo "[ERROR] expected one of: bce, anomaly, both, all"
    exit 1
    ;;
esac

mkdir -p "${LOG_ROOT}"
mkdir -p .tmp_weighted_silu_cfg

make_weighted_stage_cfg() {
  local branch="$1"
  local out_cfg="$2"
  "${PYTHON_BIN}" - "${SITE}" "${branch}" "${TEMPLATE_STAGE_CFG}" "${out_cfg}" <<'PY'
import sys
import yaml

site, branch, template_path, out_path = sys.argv[1:5]
with open(template_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

weights = {
    "bce_pos_weight": 1.0,
    "bce_neg_weight": 1.0,
    "anomaly_pos_weight": 1.0,
    "anomaly_neg_weight": 1.0,
}

if site == "utah_2019":
    # Utah 2019 misses many events, so event-positive terms are emphasized.
    if branch in {"bce", "both"}:
        weights["bce_pos_weight"] = 2.0
    if branch in {"anomaly", "both"}:
        weights["anomaly_pos_weight"] = 1.5
elif site == "utah_2023":
    # Utah 2023 collapses toward all-event predictions, so noise terms are emphasized.
    if branch in {"bce", "both"}:
        weights["bce_neg_weight"] = 2.0
    if branch in {"anomaly", "both"}:
        weights["anomaly_neg_weight"] = 2.0
elif site == "pohang":
    # Pohang is already strong; this is only a control branch.
    if branch in {"bce", "both"}:
        weights["bce_neg_weight"] = 1.5
    if branch in {"anomaly", "both"}:
        weights["anomaly_neg_weight"] = 1.5
else:
    raise ValueError(f"Unsupported site: {site}")

cfg.setdefault("train", {}).update(weights)
cfg["train"]["weighting_branch"] = branch
if site == "utah_2023":
    cfg["train"]["monitor"] = "balanced_acc"
    cfg["train"]["monitor_mode"] = "max"
cfg.setdefault("paths", {})["run_root"] = f"./runs/weighted_silu/{site}_{branch}"

with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_weighted_base_cfg() {
  local run_root="$1"
  local out_cfg="$2"
  "${PYTHON_BIN}" - "${BASE_CFG}" "${run_root}" "${out_cfg}" <<'PY'
import sys
import yaml

base_path, run_root, out_path = sys.argv[1:4]
with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = f"./{run_root}"
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_weighted_test_cfg() {
  local out_cfg="$1"
  "${PYTHON_BIN}" - "${SITE}" "${out_cfg}" <<'PY'
import sys
import yaml

site, out_path = sys.argv[1:3]
with open("configs/train/test.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("test", {})["threshold_metric"] = "balanced_acc" if site == "utah_2023" else "f1"
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

run_one_branch() {
  local branch="$1"
  local run_root="runs/weighted_silu/${SITE}_${branch}"
  local stage_cfg=".tmp_weighted_silu_cfg/${SITE}_${branch}.yaml"
  local effective_base_cfg=".tmp_weighted_silu_cfg/${SITE}_${branch}_base.yaml"
  local test_stage_cfg=".tmp_weighted_silu_cfg/${SITE}_${branch}_test.yaml"
  local finetune_dir="${run_root}/finetune/${EXPERIMENT}"
  local log_file="${LOG_ROOT}/weighted_silu__${SITE}__${branch}.log"

  make_weighted_stage_cfg "${branch}" "${stage_cfg}"
  make_weighted_base_cfg "${run_root}" "${effective_base_cfg}"
  make_weighted_test_cfg "${test_stage_cfg}"

  {
    echo "============================================================"
    echo "[WEIGHTED SILU START] $(date '+%F %T') site=${SITE} branch=${branch} gpu=${GPU}"
    echo "python_bin=${PYTHON_BIN}"
    echo "base_cfg=${effective_base_cfg}"
    echo "source_base_cfg=${BASE_CFG}"
    echo "stage_cfg=${stage_cfg}"
    echo "test_stage_cfg=${test_stage_cfg}"
    echo "template_stage_cfg=${TEMPLATE_STAGE_CFG}"
    echo "run_root=${run_root}"
    echo "finetune_dir=${finetune_dir}"
    echo "============================================================"
  } | tee -a "${log_file}"

  echo "[WEIGHTED SILU] finetune" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
    --base_cfg "${effective_base_cfg}" \
    --stage_cfg "${stage_cfg}" >> "${log_file}" 2>&1

  echo "[WEIGHTED SILU] test" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
    --base_cfg "${effective_base_cfg}" \
    --stage_cfg "${test_stage_cfg}" >> "${log_file}" 2>&1

  echo "[WEIGHTED SILU] analyze" | tee -a "${log_file}"
  "${PYTHON_BIN}" -m src.detection.analysis.analyze \
    --base_cfg "${effective_base_cfg}" \
    --stage_cfg configs/train/analyze.yaml >> "${log_file}" 2>&1

  echo "[WEIGHTED SILU] tsne" | tee -a "${log_file}"
  PYTHON_BIN="${PYTHON_BIN}" bash scripts/gpu/run_tsne.sh "${GPU}" "${finetune_dir}" >> "${log_file}" 2>&1

  echo "[DONE] weighted SiLU pipeline completed. site=${SITE} branch=${branch}" | tee -a "${log_file}"
}

if [[ "${BRANCH}" == "all" ]]; then
  for branch in bce anomaly both; do
    run_one_branch "${branch}"
  done
else
  run_one_branch "${BRANCH}"
fi
