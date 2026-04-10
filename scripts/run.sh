#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
GPU_LIST="${2:-0}"
WORK_DIR="$(pwd)"
LOG_ROOT="${3:-${WORK_DIR}/logs}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mkdir -p "${LOG_ROOT}"
mkdir -p .tmp_stage_label_eff_pretrained_dynamic_center_test_cfg
mkdir -p .tmp_stage_label_eff_pretrained_dynamic_center_analyze_cfg

export PYTHONPATH=.
export LD_LIBRARY_PATH=/home/anaconda3/lib:${LD_LIBRARY_PATH:-}

BASE_CONFIGS=(
  "config/base_pohang.yaml"\
  "config/base_utah_2019.yaml"\
  "config/base_utah_2023.yaml"\
)

TRAIN_CONFIGS=(
  "config/train.yaml"\
)

PRETRAIN_STAGE_CFG="config/pretrain_reconst.yaml"
TEST_STAGE_CFG="config/test.yaml"
ANALYZE_STAGE_CFG="config/analyze.yaml"

if [[ "${MODE}" != "pretrain" && "${MODE}" != "finetune" && "${MODE}" != "test" && "${MODE}" != "analyze" && "${MODE}" != "full" ]]; then
  echo "[ERROR] invalid MODE: ${MODE}"
  exit 1
fi

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

run_pretrain_job() {
  local gpu="$1"
  local base_cfg="$2"
  local base_name log_file
  base_name="$(basename "${base_cfg}" .yaml)"
  log_file="${LOG_ROOT}/pretrain__${base_name}.log"
  {
    echo "============================================================"
    echo "[PRETRAIN START] $(date '+%F %T') gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "base_cfg=${base_cfg}"
    echo "stage_cfg=${PRETRAIN_STAGE_CFG}"
    echo "============================================================"
  } | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/training/trainer_pretrain.py \
    --base_cfg "${base_cfg}" \
    --stage_cfg "${PRETRAIN_STAGE_CFG}" >> "${log_file}" 2>&1
}

launch_pretrain_queue() {
  local pids=()
  local idx=0
  for base_cfg in "${BASE_CONFIGS[@]}"; do
    local gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
    run_pretrain_job "${gpu}" "${base_cfg}" &
    pids+=($!)
    idx=$((idx + 1))
  done
  local exit_code=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
  done
  return "${exit_code}"
}

run_finetune_job() {
  local gpu="$1"
  local base_cfg="$2"
  local train_cfg="$3"
  local base_name train_name exp_suffix log_file
  base_name="$(basename "${base_cfg}" .yaml)"
  train_name="$(basename "${train_cfg}" .yaml)"
  exp_suffix="${train_name}"
  log_file="${LOG_ROOT}/finetune__${base_name}__${train_name}.log"
  {
    echo "============================================================"
    echo "[FINETUNE START] $(date '+%F %T') gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "base_cfg=${base_cfg}"
    echo "train_cfg=${train_cfg}"
    echo "exp_suffix=${exp_suffix}"
    echo "============================================================"
  } | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/training/trainer_finetune.py \
    --base_cfg "${base_cfg}" \
    --stage_cfg "${train_cfg}" \
    --exp_suffix "${exp_suffix}" >> "${log_file}" 2>&1
}

launch_finetune_queue() {
  local pids=()
  local idx=0
  for base_cfg in "${BASE_CONFIGS[@]}"; do
    for train_cfg in "${TRAIN_CONFIGS[@]}"; do
      local gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
      run_finetune_job "${gpu}" "${base_cfg}" "${train_cfg}" &
      pids+=($!)
      idx=$((idx + 1))
    done
  done
  local exit_code=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
  done
  return "${exit_code}"
}

run_test_job() {
  local gpu="$1"
  local base_cfg="$2"
  local train_cfg="$3"
  local base_name train_name exp_suffix log_file test_base_cfg
  base_name="$(basename "${base_cfg}" .yaml)"
  train_name="$(basename "${train_cfg}" .yaml)"
  exp_suffix="${train_name}"
  test_base_cfg=".tmp_stage_label_eff_pretrained_dynamic_center_test_cfg/${base_name}__${train_name}.yaml"
  log_file="${LOG_ROOT}/test__${base_name}__${train_name}.log"
  {
    echo "============================================================"
    echo "[TEST START] $(date '+%F %T') gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "base_cfg=${base_cfg}"
    echo "train_cfg=${train_cfg}"
    echo "exp_suffix=${exp_suffix}"
    echo "============================================================"
  } | tee -a "${log_file}"
  make_suffix_base_cfg "${base_cfg}" "${exp_suffix}" "${test_base_cfg}" >> "${log_file}" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" python src/training/trainer_test.py \
    --base_cfg "${test_base_cfg}" \
    --stage_cfg "${TEST_STAGE_CFG}" >> "${log_file}" 2>&1
}

launch_test_queue() {
  local pids=()
  local idx=0
  for base_cfg in "${BASE_CONFIGS[@]}"; do
    for train_cfg in "${TRAIN_CONFIGS[@]}"; do
      local gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
      run_test_job "${gpu}" "${base_cfg}" "${train_cfg}" &
      pids+=($!)
      idx=$((idx + 1))
    done
  done
  local exit_code=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
  done
  return "${exit_code}"
}

run_analyze_all() {
  for base_cfg in "${BASE_CONFIGS[@]}"; do
    for train_cfg in "${TRAIN_CONFIGS[@]}"; do
      local base_name train_name exp_suffix analyze_base_cfg log_file
      base_name="$(basename "${base_cfg}" .yaml)"
      train_name="$(basename "${train_cfg}" .yaml)"
      exp_suffix="${train_name}"
      analyze_base_cfg=".tmp_stage_label_eff_pretrained_dynamic_center_analyze_cfg/${base_name}__${train_name}.yaml"
      log_file="${LOG_ROOT}/analyze__${base_name}__${train_name}.log"
      {
        echo "============================================================"
        echo "[ANALYZE START] $(date '+%F %T')"
        echo "base_cfg=${base_cfg}"
        echo "train_cfg=${train_cfg}"
        echo "exp_suffix=${exp_suffix}"
        echo "============================================================"
      } | tee -a "${log_file}"
      make_suffix_base_cfg "${base_cfg}" "${exp_suffix}" "${analyze_base_cfg}" >> "${log_file}" 2>&1
      python src/analysis/analyze.py \
        --base_cfg "${analyze_base_cfg}" \
        --stage_cfg "${ANALYZE_STAGE_CFG}" >> "${log_file}" 2>&1
    done
  done
}

case "${MODE}" in
  pretrain) launch_pretrain_queue ;;
  finetune) launch_finetune_queue ;;
  test) launch_test_queue ;;
  analyze) run_analyze_all ;;
  full)
    echo "[INFO] MODE=full : pretrain -> finetune -> test -> analyze"
    launch_pretrain_queue
    launch_finetune_queue
    launch_test_queue
    run_analyze_all
    ;;
esac

echo "[DONE] stage_label_efficiency_pretrained_dynamic_center completed. mode=${MODE}"