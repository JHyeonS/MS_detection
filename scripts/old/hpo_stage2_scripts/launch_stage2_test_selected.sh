#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-0}"
LOG_ROOT="${2:-logs_stage2_test}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mkdir -p "${LOG_ROOT}"
mkdir -p .tmp_stage2_test_cfg

export PYTHONPATH=.

BASE_CONFIGS=(
  "config/base_contrast_top1.yaml"
  "config/base_reconst_top1.yaml"
  "config/base_reconst_top2.yaml"
  "config/base_reconst_top3.yaml"
)

TRAIN_CONFIGS=(
  "config/train_stage2_freeze_lr1e-04_aw0p01.yaml"
  "config/train_stage2_freeze_lr3e-04_aw0p01.yaml"
  "config/train_stage2_unfreeze_lr1e-04_aw0p01.yaml"
  "config/train_stage2_unfreeze_lr3e-04_aw0p01.yaml"
  "config/train_stage2_freeze_lr1e-04_aw0p10.yaml"
  "config/train_stage2_unfreeze_lr1e-04_aw0p10.yaml"
  "config/train_stage2_freeze_lr3e-04_aw0p10.yaml"
  "config/train_stage2_unfreeze_lr3e-04_aw0p10.yaml"
)

make_test_base_cfg() {
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

worker() {
  local gpu="$1"
  local worker_id="$2"
  local idx=0

  echo "[WORKER-${worker_id}] start on GPU ${gpu}"

  for base_cfg in "${BASE_CONFIGS[@]}"; do
    for train_cfg in "${TRAIN_CONFIGS[@]}"; do
      if (( idx % ${#GPUS[@]} != worker_id )); then
        idx=$((idx + 1))
        continue
      fi

      local base_name train_name exp_suffix log_file test_base_cfg
      base_name="$(basename "${base_cfg}" .yaml)"
      train_name="$(basename "${train_cfg}" .yaml)"
      exp_suffix="${train_name}"
      test_base_cfg=".tmp_stage2_test_cfg/${base_name}__${train_name}.yaml"
      log_file="${LOG_ROOT}/${base_name}__${train_name}.log"

      {
        echo "============================================================"
        echo "[START] $(date '+%F %T') worker=${worker_id} gpu=${gpu}"
        echo "CUDA_VISIBLE_DEVICES=${gpu}"
        echo "base_cfg=${base_cfg}"
        echo "train_cfg=${train_cfg}"
        echo "exp_suffix=${exp_suffix}"
        echo "test_base_cfg=${test_base_cfg}"
        echo "============================================================"
      } | tee -a "${log_file}"

      make_test_base_cfg "${base_cfg}" "${exp_suffix}" "${test_base_cfg}" >> "${log_file}" 2>&1

      CUDA_VISIBLE_DEVICES="${gpu}" python src/training/trainer_test.py \
        --base_cfg "${test_base_cfg}" \
        --stage_cfg "config/test.yaml" >> "${log_file}" 2>&1

      local status=$?
      echo "[WORKER-${worker_id}] finished ${base_name} + ${train_name} status=${status}" | tee -a "${log_file}"

      if [[ "${status}" -ne 0 ]]; then
        echo "[WORKER-${worker_id}] stop because previous job failed." | tee -a "${log_file}"
        return "${status}"
      fi

      idx=$((idx + 1))
    done
  done

  echo "[WORKER-${worker_id}] all assigned jobs finished on GPU ${gpu}"
}

cleanup() {
  echo
  echo "[WARN] interrupt received, killing child workers..."
  jobs -pr | xargs -r kill || true
  wait || true
  pkill -f "src/training/trainer_test.py" || true
  exit 130
}
trap cleanup INT TERM

pids=()
for (( worker_id=0; worker_id<${#GPUS[@]}; worker_id++ )); do
  worker "${GPUS[$worker_id]}" "${worker_id}" &
  pids+=($!)
done

exit_code=0
for pid in "${pids[@]}"; do
  wait "${pid}" || exit_code=$?
done

if [[ "${exit_code}" -ne 0 ]]; then
  echo "[ERROR] one or more workers failed."
  exit "${exit_code}"
fi

echo "[DONE] all selected stage2 test jobs finished successfully."
