#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/launch_hpo_train_stage1_contrast_queue.sh full 1,8,9 logs_hpo_stage1_contrast
#
# Behavior:
#   - GPU 하나당 항상 job 1개만 실행
#   - 해당 job이 끝나면 같은 GPU에서 다음 config 실행
#   - pretrain 없이 train -> test -> analyze 만 수행

MODE="${1:-full}"
GPU_LIST="${2:-0,1}"
LOG_ROOT="${3:-logs_hpo_stage1_contrast}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mkdir -p "${LOG_ROOT}"

CONFIGS=(
  "config/base_hpo_stage1_contrast_01.yaml"
  "config/base_hpo_stage1_contrast_02.yaml"
  "config/base_hpo_stage1_contrast_03.yaml"
  "config/base_hpo_stage1_contrast_04.yaml"
  "config/base_hpo_stage1_contrast_05.yaml"
  "config/base_hpo_stage1_contrast_06.yaml"
  "config/base_hpo_stage1_contrast_07.yaml"
  "config/base_hpo_stage1_contrast_08.yaml"
  "config/base_hpo_stage1_contrast_09.yaml"
  "config/base_hpo_stage1_contrast_10.yaml"
  "config/base_hpo_stage1_contrast_11.yaml"
  "config/base_hpo_stage1_contrast_12.yaml"
  "config/base_hpo_stage1_contrast_13.yaml"
  "config/base_hpo_stage1_contrast_14.yaml"
  "config/base_hpo_stage1_contrast_15.yaml"
  "config/base_hpo_stage1_contrast_16.yaml"
  "config/base_hpo_stage1_contrast_17.yaml"
  "config/base_hpo_stage1_contrast_18.yaml"
  "config/base_hpo_stage1_contrast_19.yaml"
  "config/base_hpo_stage1_contrast_20.yaml"
  "config/base_hpo_stage1_contrast_21.yaml"
  "config/base_hpo_stage1_contrast_22.yaml"
  "config/base_hpo_stage1_contrast_23.yaml"
  "config/base_hpo_stage1_contrast_24.yaml"
  "config/base_hpo_stage1_contrast_25.yaml"
  "config/base_hpo_stage1_contrast_26.yaml"
  "config/base_hpo_stage1_contrast_27.yaml"
)

N_GPU="${#GPUS[@]}"
N_CFG="${#CONFIGS[@]}"

if [[ "${N_GPU}" -eq 0 ]]; then
    echo "[ERROR] No GPUs provided."
    exit 1
fi

if [[ "${MODE}" != "full" && "${MODE}" != "train" && "${MODE}" != "test" && "${MODE}" != "analyze" ]]; then
    echo "[ERROR] Invalid MODE: ${MODE}"
    echo "Usage: bash scripts/launch_hpo_train_stage1_contrast_queue.sh [full|train|test|analyze] [GPU_LIST] [LOG_ROOT]"
    exit 1
fi

run_one() {
    local gpu="$1"
    local cfg="$2"
    local log_file="$3"

    local train_cfg="config/train.yaml"
    local test_cfg="config/test.yaml"
    local analyze_cfg="config/analyze.yaml"

    export CUDA_VISIBLE_DEVICES="${gpu}"

    echo "[RUN] gpu=${gpu} mode=${MODE} cfg=${cfg}" | tee -a "${log_file}"

    if [[ "${MODE}" == "full" ]]; then
        bash scripts/train.sh "${cfg}" "${train_cfg}" >> "${log_file}" 2>&1
        bash scripts/test.sh "${cfg}" "${test_cfg}" >> "${log_file}" 2>&1
        bash scripts/analyze.sh "${cfg}" "${analyze_cfg}" >> "${log_file}" 2>&1
    elif [[ "${MODE}" == "train" ]]; then
        bash scripts/train.sh "${cfg}" "${train_cfg}" >> "${log_file}" 2>&1
    elif [[ "${MODE}" == "test" ]]; then
        bash scripts/test.sh "${cfg}" "${test_cfg}" >> "${log_file}" 2>&1
    elif [[ "${MODE}" == "analyze" ]]; then
        bash scripts/analyze.sh "${cfg}" "${analyze_cfg}" >> "${log_file}" 2>&1
    fi
}

worker() {
    local gpu="$1"
    local worker_id="$2"

    echo "[WORKER-${worker_id}] start on GPU ${gpu}"

    for (( idx=worker_id; idx<N_CFG; idx+=N_GPU )); do
        local cfg="${CONFIGS[$idx]}"
        local base_name
        base_name="$(basename "${cfg}" .yaml)"
        local log_file="${LOG_ROOT}/${base_name}.log"

        echo "[WORKER-${worker_id}] gpu=${gpu} mode=${MODE} cfg=${cfg} log=${log_file}"
        : > "${log_file}"

        run_one "${gpu}" "${cfg}" "${log_file}"
        status=$?

        echo "[WORKER-${worker_id}] finished cfg=${cfg} status=${status}" | tee -a "${log_file}"

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

echo "[DONE] All contrast queue jobs finished successfully."