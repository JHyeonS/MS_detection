#!/usr/bin/env bash
set -euo pipefail

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
if [[ "${N_GPU}" -eq 0 ]]; then
    echo "[ERROR] No GPUs provided."
    exit 1
fi

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    gpu="${GPUS[$((i % N_GPU))]}"
    log_file="${LOG_ROOT}/$(basename "${cfg}" .yaml).log"
    echo "[LAUNCH] gpu=${gpu} mode=${MODE} cfg=${cfg} log=${log_file}"
    nohup bash scripts/run_hpo_stage1_contrast.sh "${gpu}" "${MODE}" "${cfg}" > "${log_file}" 2>&1 &
done

echo "[DONE] All contrast jobs launched."
