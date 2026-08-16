#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6}"
SEEDS="${FIGURE_V2_SEEDS:-41,42,43,44,45}"
LOG_ROOT="${1:-logs/figure_bundle_v2}"
RUN_ROOT="${2:-runs/figure_bundle_v2_seed_runs}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_LIST}"
if [[ "${#GPU_ARRAY[@]}" -lt 7 ]]; then
  echo "[ERROR] GPU_LIST must provide at least 7 GPUs; got: ${GPU_LIST}"
  exit 1
fi

launch_bg() {
  local log_file="$1"
  shift
  nohup "$@" > "${log_file}" 2>&1 &
  echo $!
}

PIDS=()
PIDS+=("$(launch_bg "${LOG_ROOT}/gpu0_pohang_main.log" bash scripts/gpu/run_figure_bundle_v2_worker.sh site "${GPU_ARRAY[0]}" pohang_main "${SEEDS}" "${LOG_ROOT}" "${RUN_ROOT}" pohang "scratch,reconst,contrast,reconst_noanom" true)")
PIDS+=("$(launch_bg "${LOG_ROOT}/gpu1_utah2019_main.log" bash scripts/gpu/run_figure_bundle_v2_worker.sh site "${GPU_ARRAY[1]}" utah_2019_main "${SEEDS}" "${LOG_ROOT}" "${RUN_ROOT}" utah_2019 "scratch,reconst,contrast,reconst_noanom" true)")
PIDS+=("$(launch_bg "${LOG_ROOT}/gpu2_utah2023_main.log" bash scripts/gpu/run_figure_bundle_v2_worker.sh site "${GPU_ARRAY[2]}" utah_2023_main "${SEEDS}" "${LOG_ROOT}" "${RUN_ROOT}" utah_2023 "scratch,reconst" true)")
PIDS+=("$(launch_bg "${LOG_ROOT}/gpu3_pohang_norm.log" bash scripts/gpu/run_figure_bundle_v2_worker.sh norm "${GPU_ARRAY[3]}" pohang_norm "${SEEDS}" "${LOG_ROOT}" "${RUN_ROOT}" pohang)")
PIDS+=("$(launch_bg "${LOG_ROOT}/gpu4_utah2019_norm.log" bash scripts/gpu/run_figure_bundle_v2_worker.sh norm "${GPU_ARRAY[4]}" utah_2019_norm "${SEEDS}" "${LOG_ROOT}" "${RUN_ROOT}" utah_2019)")
PIDS+=("$(launch_bg "${LOG_ROOT}/gpu5_utah2023_norm.log" bash scripts/gpu/run_figure_bundle_v2_worker.sh norm "${GPU_ARRAY[5]}" utah_2023_norm "${SEEDS}" "${LOG_ROOT}" "${RUN_ROOT}" utah_2023)")
PIDS+=("$(launch_bg "${LOG_ROOT}/gpu6_pair.log" bash scripts/gpu/run_figure_bundle_v2_worker.sh pair "${GPU_ARRAY[6]}" pair "${SEEDS}" "${LOG_ROOT}" "${RUN_ROOT}")")

WATCHER_LOG="${LOG_ROOT}/bundle_v2_plotter.log"
nohup bash -lc "while true; do alive=0; for pid in ${PIDS[*]}; do if kill -0 \$pid 2>/dev/null; then alive=1; fi; done; if [[ \$alive -eq 0 ]]; then break; fi; sleep 30; done; cd '${PWD}' && python scripts/gpu/plot_figure_bundle_v2.py" > "${WATCHER_LOG}" 2>&1 &
WATCHER_PID=$!

cat <<EOF
[V2] launched figure bundle v2 seed sweeps
log_root=${LOG_ROOT}
run_root=${RUN_ROOT}
seeds=${SEEDS}
pids=${PIDS[*]}
watcher_pid=${WATCHER_PID}
watcher_log=${WATCHER_LOG}
EOF
