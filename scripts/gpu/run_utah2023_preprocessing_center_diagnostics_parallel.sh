#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-0,1,2}"
LOG_ROOT="${2:-logs/preprocessing_center_utah2023_parallel}"
RUN_ROOT_BASE="${3:-runs/preprocessing_center_diagnostics_utah2023_v1}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "[ERROR] provide at least 3 GPUs, e.g. 0,1,2"
  exit 1
fi

export PYTHON_BIN="${PYTHON_BIN:-/home/ted1204/.conda/envs/ms_detection/bin/python}"
export CENTER_DIAG_FRACTIONS="${CENTER_DIAG_FRACTIONS:-0.10,0.50,1.00}"
export CENTER_DIAG_UPDATE="${CENTER_DIAG_UPDATE:-every_epoch}"
export CENTER_DIAG_RUN_TEST="${CENTER_DIAG_RUN_TEST:-true}"
export CENTER_DIAG_LOG_WASSERSTEIN="${CENTER_DIAG_LOG_WASSERSTEIN:-true}"
export CENTER_DIAG_WASSERSTEIN_PROJECTIONS="${CENTER_DIAG_WASSERSTEIN_PROJECTIONS:-32}"
export CENTER_DIAG_WASSERSTEIN_QUANTILES="${CENTER_DIAG_WASSERSTEIN_QUANTILES:-128}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT_BASE}"

echo "[INFO] launching Utah 2023 preprocessing center diagnostics in parallel"
echo "[INFO] run_root=${RUN_ROOT_BASE}"
echo "[INFO] fractions=${CENTER_DIAG_FRACTIONS}"
echo "[INFO] center_update=${CENTER_DIAG_UPDATE}"
echo "[INFO] log_wasserstein=${CENTER_DIAG_LOG_WASSERSTEIN}"

pids=()

bash scripts/gpu/preprocessing_center_diagnostics.sh \
  utah_2023 baseline "${GPUS[0]}" "${LOG_ROOT}" "${RUN_ROOT_BASE}" \
  > "${LOG_ROOT}/utah2023_baseline.stdout.log" 2>&1 &
pids+=($!)

bash scripts/gpu/preprocessing_center_diagnostics.sh \
  utah_2023 bandpass_agc_none "${GPUS[1]}" "${LOG_ROOT}" "${RUN_ROOT_BASE}" \
  > "${LOG_ROOT}/utah2023_bandpass_agc_none.stdout.log" 2>&1 &
pids+=($!)

bash scripts/gpu/preprocessing_center_diagnostics.sh \
  utah_2023 bandpass_agc_robust "${GPUS[2]}" "${LOG_ROOT}" "${RUN_ROOT_BASE}" \
  > "${LOG_ROOT}/utah2023_bandpass_agc_robust.stdout.log" 2>&1 &
pids+=($!)

exit_code=0
for pid in "${pids[@]}"; do
  wait "${pid}" || exit_code=$?
done

exit "${exit_code}"
