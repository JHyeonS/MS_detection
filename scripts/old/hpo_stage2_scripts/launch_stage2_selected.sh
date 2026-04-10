#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash scripts/launch_stage2_selected.sh <gpu_list> <log_root>
# example:
# bash scripts/launch_stage2_selected.sh 1,2,3,4 logs_stage2_selected

GPU_LIST="${1:-0}"
LOG_ROOT="${2:-logs_stage2_selected}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mkdir -p "${LOG_ROOT}"

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

sync_pretrain_for_base() {
  local base_cfg="$1"
  local exp_name stage2_root stage1_root src_dir dst_dir

  exp_name="$(extract_yaml_value "${base_cfg}" "data.experiment")"
  stage2_root="$(extract_yaml_value "${base_cfg}" "paths.run_root")"

  if [[ "${exp_name}" == *"_reconst_"* ]]; then
    stage1_root="runs/hpo_stage1c_reconst"
  elif [[ "${exp_name}" == *"_contrast_"* ]]; then
    stage1_root="runs/hpo_stage1c_contrast"
  else
    echo "[WARN] could not infer method from experiment: ${exp_name}"
    return 1
  fi

  src_dir="${stage1_root}/pretrain/${exp_name}"
  dst_dir="${stage2_root}/pretrain/${exp_name}"

  if [[ ! -d "${src_dir}" ]]; then
    echo "[ERROR] missing stage1 pretrain dir: ${src_dir}"
    return 1
  fi

  mkdir -p "${dst_dir}"
  rsync -a "${src_dir}/" "${dst_dir}/"
  echo "[SYNC] ${src_dir} -> ${dst_dir}"
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

      local base_name train_name exp_suffix log_file
      base_name="$(basename "${base_cfg}" .yaml)"
      train_name="$(basename "${train_cfg}" .yaml)"
      exp_suffix="${train_name}"
      log_file="${LOG_ROOT}/${base_name}__${train_name}.log"

      {
        echo "============================================================"
        echo "[START] $(date '+%F %T') worker=${worker_id} gpu=${gpu}"
        echo "CUDA_VISIBLE_DEVICES=${gpu}"
        echo "base_cfg=${base_cfg}"
        echo "train_cfg=${train_cfg}"
        echo "exp_suffix=${exp_suffix}"
        echo "============================================================"
      } | tee -a "${log_file}"

      sync_pretrain_for_base "${base_cfg}" >> "${log_file}" 2>&1

      CUDA_VISIBLE_DEVICES="${gpu}" python src/training/trainer_finetune.py         --base_cfg "${base_cfg}"         --stage_cfg "${train_cfg}"         --exp_suffix "${exp_suffix}" >> "${log_file}" 2>&1

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
  pkill -f "src/training/trainer_finetune.py" || true
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

echo "[DONE] all selected stage2 jobs finished successfully."
