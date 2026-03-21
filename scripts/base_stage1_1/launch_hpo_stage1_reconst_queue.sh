#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/launch_hpo_stage1_reconst_queue.sh full 8,9,10 logs_hpo_stage1_reconst
#
# Behavior:
#   - GPU 하나당 항상 job 1개만 실행
#   - 해당 job이 끝나면 같은 GPU에서 다음 config 실행
#   - 전체 config를 GPU 개수에 맞춰 라운드 분배

MODE="${1:-full}"
GPU_LIST="${2:-0,1}"
LOG_ROOT="${3:-logs_hpo_stage1_reconst}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mkdir -p "${LOG_ROOT}"

CONFIGS=(
  "config/base_hpo_stage1_reconst_28.yaml"
  "config/base_hpo_stage1_reconst_29.yaml"
  "config/base_hpo_stage1_reconst_30.yaml"
  "config/base_hpo_stage1_reconst_31.yaml"
  "config/base_hpo_stage1_reconst_32.yaml"
  "config/base_hpo_stage1_reconst_33.yaml"
  "config/base_hpo_stage1_reconst_34.yaml"
  "config/base_hpo_stage1_reconst_35.yaml"
  "config/base_hpo_stage1_reconst_36.yaml"
  "config/base_hpo_stage1_reconst_37.yaml"
  "config/base_hpo_stage1_reconst_38.yaml"
  "config/base_hpo_stage1_reconst_39.yaml"
  "config/base_hpo_stage1_reconst_40.yaml"
  "config/base_hpo_stage1_reconst_41.yaml"
  "config/base_hpo_stage1_reconst_42.yaml"
  "config/base_hpo_stage1_reconst_43.yaml"
  "config/base_hpo_stage1_reconst_44.yaml"
  "config/base_hpo_stage1_reconst_45.yaml"
  "config/base_hpo_stage1_reconst_46.yaml"
  "config/base_hpo_stage1_reconst_47.yaml"
  "config/base_hpo_stage1_reconst_48.yaml"
  "config/base_hpo_stage1_reconst_49.yaml"
  "config/base_hpo_stage1_reconst_50.yaml"
  "config/base_hpo_stage1_reconst_51.yaml"
  "config/base_hpo_stage1_reconst_52.yaml"
  "config/base_hpo_stage1_reconst_53.yaml"
  "config/base_hpo_stage1_reconst_54.yaml"
)

N_GPU="${#GPUS[@]}"
N_CFG="${#CONFIGS[@]}"

if [[ "${N_GPU}" -eq 0 ]]; then
    echo "[ERROR] No GPUs provided."
    exit 1
fi

worker() {
    local gpu="$1"
    local worker_id="$2"

    echo "[WORKER-${worker_id}] start on GPU ${gpu}"

    for (( idx=worker_id; idx<N_CFG; idx+=N_GPU )); do
        local cfg="${CONFIGS[$idx]}"
        local base_name
        base_name="$(basename "${cfg}" .yaml)"
        local log_file="${LOG_ROOT}/${base_name}.log"

        echo "[WORKER-${worker_id}] gpu=${gpu} mode=${MODE} cfg=${cfg}"
        echo "[WORKER-${worker_id}] log=${log_file}"

        bash scripts/run_hpo_stage1_reconst.sh "${gpu}" "${MODE}" "${cfg}" \
            > "${log_file}" 2>&1

        status=$?
        echo "[WORKER-${worker_id}] finished cfg=${cfg} status=${status}"

        if [[ "${status}" -ne 0 ]]; then
            echo "[WORKER-${worker_id}] stopped because previous job failed: ${cfg}"
            return "${status}"
        fi
    done

    echo "[WORKER-${worker_id}] all assigned jobs finished on GPU ${gpu}"
}

pids=()
for (( worker_id=0; worker_id<N_GPU; worker_id++ )); do
    gpu="${GPUS[$worker_id]}"
    worker "${gpu}" "${worker_id}" &
    pids+=($!)
done

exit_code=0
for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
done

if [[ "${exit_code}" -ne 0 ]]; then
    echo "[ERROR] One or more workers failed."
    exit "${exit_code}"
fi

echo "[DONE] All reconstruction queue jobs finished successfully."
