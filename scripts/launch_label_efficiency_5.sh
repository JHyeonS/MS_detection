#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/launch_label_efficiency_full.sh <mode> <gpu_list> <log_root>
#
# mode:
#   pretrain  : label-efficiency experiment용 base에 대해 pretrain만 실행
#   finetune  : fraction별 finetune만 실행
#   test      : fraction별 test만 실행
#   analyze   : fraction별 analyze만 실행
#   full      : pretrain -> finetune -> test -> analyze 전부 실행
#
# example:
# bash scripts/launch_label_efficiency_full.sh full 1,2,3,4 logs_label_efficiency_full

MODE="${1:-full}"
GPU_LIST="${2:-0}"
LOG_ROOT="${3:-logs_label_efficiency_full}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mkdir -p "${LOG_ROOT}"
mkdir -p .tmp_label_eff_test_cfg
mkdir -p .tmp_label_eff_analyze_cfg

export PYTHONPATH=.

if [[ "${MODE}" != "pretrain" && "${MODE}" != "finetune" && "${MODE}" != "test" && "${MODE}" != "analyze" && "${MODE}" != "full" ]]; then
  echo "[ERROR] invalid MODE: ${MODE}"
  exit 1
fi

# ------------------------------------------------------------
# Fixed best base / stage settings
# ------------------------------------------------------------
BASE_CFG="config/base_5.yaml"

TRAIN_CONFIGS=(
  "config/train_label_eff_frac1p0.yaml"
  "config/train_label_eff_frac0p5.yaml"
  "config/train_label_eff_frac0p2.yaml"
  "config/train_label_eff_frac0p1.yaml"
  "config/train_label_eff_frac0p05.yaml"
)

PRETRAIN_STAGE_CFG="config/pretrain_reconst.yaml"
TEST_STAGE_CFG="config/test.yaml"
ANALYZE_STAGE_CFG="config/analyze.yaml"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
extract_yaml_value() {
python - "$1" "$2" <<'PY'
import sys, yaml
path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cur = cfg
for part in key.split('.'):
    cur = cur[part]
print(cur)
PY
}

make_suffix_base_cfg() {
  local src_base_cfg="$1"
  local exp_suffix="$2"
  local out_cfg="$3"

  python - "${src_base_cfg}" "${exp_suffix}" "${out_cfg}" <<'PY'
import sys, yaml
src, suffix, outp = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
base_exp = cfg["data"]["experiment"]
cfg["data"]["experiment"] = f"{base_exp}__{suffix}"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(cfg["data"]["experiment"])
PY
}

cleanup() {
  echo
  echo "[WARN] interrupt received, killing child workers..."
  jobs -pr | xargs -r kill || true
  wait || true
  pkill -f "src/training/trainer_pretrain.py" || true
  pkill -f "src/training/trainer_finetune.py" || true
  pkill -f "src/training/trainer_test.py" || true
  pkill -f "src/analysis/analyze.py" || true
  exit 130
}
trap cleanup INT TERM

# ------------------------------------------------------------
# PRETRAIN
# ------------------------------------------------------------
run_pretrain_job() {
  local gpu="$1"
  local log_file="${LOG_ROOT}/pretrain.log"

  {
    echo "============================================================"
    echo "[PRETRAIN START] $(date '+%F %T') gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "base_cfg=${BASE_CFG}"
    echo "stage_cfg=${PRETRAIN_STAGE_CFG}"
    echo "============================================================"
  } | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" python src/training/trainer_pretrain.py \
    --base_cfg "${BASE_CFG}" \
    --stage_cfg "${PRETRAIN_STAGE_CFG}" >> "${log_file}" 2>&1

  local status=$?
  echo "[PRETRAIN DONE] status=${status}" | tee -a "${log_file}"
  return "${status}"
}

# ------------------------------------------------------------
# FINETUNE
# ------------------------------------------------------------
run_finetune_job() {
  local gpu="$1"
  local train_cfg="$2"

  local train_name exp_suffix log_file
  train_name="$(basename "${train_cfg}" .yaml)"
  exp_suffix="${train_name}"
  log_file="${LOG_ROOT}/finetune__${train_name}.log"

  {
    echo "============================================================"
    echo "[FINETUNE START] $(date '+%F %T') gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "base_cfg=${BASE_CFG}"
    echo "train_cfg=${train_cfg}"
    echo "exp_suffix=${exp_suffix}"
    echo "============================================================"
  } | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES="${gpu}" python src/training/trainer_finetune.py \
    --base_cfg "${BASE_CFG}" \
    --stage_cfg "${train_cfg}" \
    --exp_suffix "${exp_suffix}" >> "${log_file}" 2>&1

  local status=$?
  echo "[FINETUNE DONE] ${train_name} status=${status}" | tee -a "${log_file}"
  return "${status}"
}

launch_finetune_queue() {
  local pids=()
  local idx=0

  for train_cfg in "${TRAIN_CONFIGS[@]}"; do
    local gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
    run_finetune_job "${gpu}" "${train_cfg}" &
    pids+=($!)
    idx=$((idx + 1))
  done

  local exit_code=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
  done
  return "${exit_code}"
}

# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------
run_test_job() {
  local gpu="$1"
  local train_cfg="$2"

  local train_name exp_suffix log_file test_base_cfg
  train_name="$(basename "${train_cfg}" .yaml)"
  exp_suffix="${train_name}"
  test_base_cfg=".tmp_label_eff_test_cfg/${train_name}.yaml"
  log_file="${LOG_ROOT}/test__${train_name}.log"

  {
    echo "============================================================"
    echo "[TEST START] $(date '+%F %T') gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "base_cfg=${BASE_CFG}"
    echo "train_cfg=${train_cfg}"
    echo "exp_suffix=${exp_suffix}"
    echo "test_base_cfg=${test_base_cfg}"
    echo "============================================================"
  } | tee -a "${log_file}"

  make_suffix_base_cfg "${BASE_CFG}" "${exp_suffix}" "${test_base_cfg}" >> "${log_file}" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" python src/training/trainer_test.py \
    --base_cfg "${test_base_cfg}" \
    --stage_cfg "${TEST_STAGE_CFG}" >> "${log_file}" 2>&1

  local status=$?
  echo "[TEST DONE] ${train_name} status=${status}" | tee -a "${log_file}"
  return "${status}"
}

launch_test_queue() {
  local pids=()
  local idx=0

  for train_cfg in "${TRAIN_CONFIGS[@]}"; do
    local gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
    run_test_job "${gpu}" "${train_cfg}" &
    pids+=($!)
    idx=$((idx + 1))
  done

  local exit_code=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
  done
  return "${exit_code}"
}

# ------------------------------------------------------------
# ANALYZE
# ------------------------------------------------------------
run_analyze_all() {
  for train_cfg in "${TRAIN_CONFIGS[@]}"; do
    local train_name exp_suffix analyze_base_cfg log_file
    train_name="$(basename "${train_cfg}" .yaml)"
    exp_suffix="${train_name}"
    analyze_base_cfg=".tmp_label_eff_analyze_cfg/${train_name}.yaml"
    log_file="${LOG_ROOT}/analyze__${train_name}.log"

    {
      echo "============================================================"
      echo "[ANALYZE START] $(date '+%F %T')"
      echo "base_cfg=${BASE_CFG}"
      echo "train_cfg=${train_cfg}"
      echo "exp_suffix=${exp_suffix}"
      echo "analyze_base_cfg=${analyze_base_cfg}"
      echo "============================================================"
    } | tee -a "${log_file}"

    make_suffix_base_cfg "${BASE_CFG}" "${exp_suffix}" "${analyze_base_cfg}" >> "${log_file}" 2>&1

    python src/analysis/analyze.py \
      --base_cfg "${analyze_base_cfg}" \
      --stage_cfg "${ANALYZE_STAGE_CFG}" >> "${log_file}" 2>&1

    local status=$?
    echo "[ANALYZE DONE] ${train_name} status=${status}" | tee -a "${log_file}"

    if [[ "${status}" -ne 0 ]]; then
      echo "[ERROR] analyze failed for ${train_name}" | tee -a "${log_file}"
      return "${status}"
    fi
  done
  return 0
}

# ------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------
case "${MODE}" in
  pretrain)
    run_pretrain_job "${GPUS[0]}"
    ;;
  finetune)
    launch_finetune_queue
    ;;
  test)
    launch_test_queue
    ;;
  analyze)
    run_analyze_all
    ;;
  full)
    echo "[INFO] MODE=full : pretrain -> finetune -> test -> analyze"
    run_pretrain_job "${GPUS[0]}"
    launch_finetune_queue
    launch_test_queue
    run_analyze_all
    ;;
esac

echo "[DONE] Label-efficiency pipeline completed. mode=${MODE}"
