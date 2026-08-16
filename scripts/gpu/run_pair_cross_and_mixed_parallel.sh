#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-2,3,4}"
LOG_ROOT="${2:-logs}"
RUN_ROOT_BASE="${3:-runs/pair_pohang_utah2019_parallel}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt 3 ]]; then
  echo "[ERROR] provide at least 3 GPUs, e.g. 2,3,4"
  exit 1
fi

export PYTHON_BIN="${PYTHON_BIN:-/home/ted1204/.conda/envs/ms_detection/bin/python}"
export PAIR_METHODS="${PAIR_METHODS:-scratch,reconst}"
export PAIR_RUN_ANALYZE="${PAIR_RUN_ANALYZE:-true}"
export PAIR_RUN_TSNE="${PAIR_RUN_TSNE:-false}"
export PAIR_RUN_PRETRAIN="${PAIR_RUN_PRETRAIN:-true}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT_BASE}"

echo "[INFO] launching parallel pair study"
echo "[INFO] mixed GPU=${GPUS[0]}"
echo "[INFO] cross_p2u GPU=${GPUS[1]}"
echo "[INFO] cross_u2p GPU=${GPUS[2]}"
echo "[INFO] run_root=${RUN_ROOT_BASE}"

pids=()

bash scripts/gpu/run_pair_cross_and_mixed.sh \
  mixed "${GPUS[0]}" "${LOG_ROOT}" "${RUN_ROOT_BASE}" \
  > "${LOG_ROOT}/pair_parallel__mixed.stdout.log" 2>&1 &
pids+=($!)

bash scripts/gpu/run_pair_cross_and_mixed.sh \
  cross_p2u "${GPUS[1]}" "${LOG_ROOT}" "${RUN_ROOT_BASE}" \
  > "${LOG_ROOT}/pair_parallel__cross_p2u.stdout.log" 2>&1 &
pids+=($!)

bash scripts/gpu/run_pair_cross_and_mixed.sh \
  cross_u2p "${GPUS[2]}" "${LOG_ROOT}" "${RUN_ROOT_BASE}" \
  > "${LOG_ROOT}/pair_parallel__cross_u2p.stdout.log" 2>&1 &
pids+=($!)

exit_code=0
for pid in "${pids[@]}"; do
  wait "${pid}" || exit_code=$?
done

exit "${exit_code}"
