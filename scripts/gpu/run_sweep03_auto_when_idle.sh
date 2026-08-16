#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLBACKEND=Agg

DATASET_NAME="${SWEEP03_DATASET_NAME:-sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site}"
EXPERIMENT_ROOT="${SWEEP03_EXPERIMENT_ROOT:-data/${DATASET_NAME}/metadata}"
RUN_ROOT_BASE="${SWEEP03_RUN_ROOT_BASE:-runs/${DATASET_NAME}}"
LOG_ROOT="${SWEEP03_LOG_ROOT:-logs/${DATASET_NAME}}"
GPUS_CSV="${SWEEP03_GPUS:-0,1,2,3,4,5,6,7,8,9}"
IDLE_MINUTES="${SWEEP03_IDLE_MINUTES:-20}"
POLL_SECONDS="${SWEEP03_POLL_SECONDS:-60}"
MEM_THRESHOLD_MB="${SWEEP03_MEM_THRESHOLD_MB:-1000}"
UTIL_THRESHOLD="${SWEEP03_UTIL_THRESHOLD:-10}"
FRACTIONS="${SWEEP03_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
RUN_ANALYZE="${SWEEP03_RUN_ANALYZE:-true}"
RUN_TSNE="${SWEEP03_RUN_TSNE:-false}"
RUN_PRETRAIN="${SWEEP03_RUN_PRETRAIN:-true}"
LOG_CENTER_DIAGNOSTICS="${SWEEP03_LOG_CENTER_DIAGNOSTICS:-false}"
LOG_WASSERSTEIN_DIAGNOSTICS="${SWEEP03_LOG_WASSERSTEIN_DIAGNOSTICS:-false}"
WASSERSTEIN_NUM_PROJECTIONS="${SWEEP03_WASSERSTEIN_NUM_PROJECTIONS:-32}"
WASSERSTEIN_NUM_QUANTILES="${SWEEP03_WASSERSTEIN_NUM_QUANTILES:-128}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT_BASE}"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IDLE_REQUIRED_SECONDS=$((IDLE_MINUTES * 60))
SCHEDULER_LOG="${LOG_ROOT}/sweep03_auto_when_idle.scheduler.log"

TASKS=(
  "pohang|scratch|stage1_pohang_only"
  "pohang|contrast|stage1_pohang_only"
  "pohang|reconst,reconst_noanom|stage1_pohang_only"
  "utah_2019|scratch|stage1_utah_2019_only"
  "utah_2019|contrast|stage1_utah_2019_only"
  "utah_2019|reconst,reconst_noanom|stage1_utah_2019_only"
  "utah_2023|scratch|stage1_utah_2023_only"
  "utah_2023|contrast|stage1_utah_2023_only"
  "utah_2023|reconst,reconst_noanom|stage1_utah_2023_only"
)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${SCHEDULER_LOG}"
}

task_slug() {
  printf '%s' "$1" | tr ',|' '__' | sed 's#[^A-Za-z0-9._-]#_#g'
}

gpu_is_idle() {
  local gpu="$1"
  local line
  line="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
  if [[ -z "${line}" ]]; then
    return 1
  fi

  local mem util
  IFS=',' read -r mem util <<< "${line}"
  [[ "${mem}" =~ ^[0-9]+$ ]] || return 1
  [[ "${util}" =~ ^[0-9]+$ ]] || return 1

  if (( mem <= MEM_THRESHOLD_MB && util <= UTIL_THRESHOLD )); then
    return 0
  fi
  return 1
}

launch_task() {
  local gpu="$1"
  local task="$2"
  local site methods split_name slug split_dir run_root log_file

  IFS='|' read -r site methods split_name <<< "${task}"
  slug="$(task_slug "${site}|${methods}")"
  split_dir="${EXPERIMENT_ROOT}/${split_name}"
  run_root="${RUN_ROOT_BASE}/${slug}"
  log_file="${LOG_ROOT}/${slug}.stdout.log"

  if [[ ! -f "${split_dir}/train.csv" || ! -f "${split_dir}/pretrain.csv" || ! -f "${split_dir}/val.csv" || ! -f "${split_dir}/test.csv" ]]; then
    log "[ERROR] missing split files for task=${task}: ${split_dir}"
    return 1
  fi

  log "[LAUNCH] gpu=${gpu} site=${site} methods=${methods} split=${split_name} run_root=${run_root}"
  (
    export SITE_STUDY_SPLIT_DIR="${split_dir}"
    export SITE_STUDY_PREPROCESS="load_only"
    export SITE_STUDY_NORMALIZE="none"
    export SITE_STUDY_METHODS="${methods}"
    export SITE_STUDY_FRACTIONS="${FRACTIONS}"
    export SITE_STUDY_RUN_ANALYZE="${RUN_ANALYZE}"
    export SITE_STUDY_RUN_TSNE="${RUN_TSNE}"
    export SITE_STUDY_RUN_PRETRAIN="${RUN_PRETRAIN}"
    export SITE_STUDY_LOG_CENTER_DIAGNOSTICS="${LOG_CENTER_DIAGNOSTICS}"
    export SITE_STUDY_LOG_WASSERSTEIN_DIAGNOSTICS="${LOG_WASSERSTEIN_DIAGNOSTICS}"
    export SITE_STUDY_WASSERSTEIN_NUM_PROJECTIONS="${WASSERSTEIN_NUM_PROJECTIONS}"
    export SITE_STUDY_WASSERSTEIN_NUM_QUANTILES="${WASSERSTEIN_NUM_QUANTILES}"
    export SITE_STUDY_LOG_SLUG="${DATASET_NAME}_${slug}"
    bash scripts/gpu/site_main_study.sh "${site}" "${gpu}" "${LOG_ROOT}" "${run_root}"
  ) > "${log_file}" 2>&1 &

  echo "$!" > "${LOG_ROOT}/${slug}.pid"
}

task_index=0
declare -A idle_seconds
declare -A gpu_busy
declare -A gpu_task_pid

for gpu in "${GPUS[@]}"; do
  idle_seconds["${gpu}"]=0
  gpu_busy["${gpu}"]=0
  gpu_task_pid["${gpu}"]=""
done

log "[START] dataset=${DATASET_NAME}"
log "[CONFIG] gpus=${GPUS_CSV} idle_minutes=${IDLE_MINUTES} mem_threshold_mb=${MEM_THRESHOLD_MB} util_threshold=${UTIL_THRESHOLD}"
log "[CONFIG] fractions=${FRACTIONS} analyze=${RUN_ANALYZE} tsne=${RUN_TSNE}"
log "[CONFIG] center_diagnostics=${LOG_CENTER_DIAGNOSTICS} wasserstein_diagnostics=${LOG_WASSERSTEIN_DIAGNOSTICS} wasserstein_projections=${WASSERSTEIN_NUM_PROJECTIONS} wasserstein_quantiles=${WASSERSTEIN_NUM_QUANTILES}"
log "[CONFIG] experiment_root=${EXPERIMENT_ROOT}"
log "[CONFIG] run_root_base=${RUN_ROOT_BASE}"
log "[CONFIG] queued_tasks=${#TASKS[@]}"

while true; do
  all_done=true

  for gpu in "${GPUS[@]}"; do
    pid="${gpu_task_pid[${gpu}]}"
    if [[ -n "${pid}" ]]; then
      if kill -0 "${pid}" 2>/dev/null; then
        all_done=false
        continue
      fi
      log "[DONE] gpu=${gpu} pid=${pid}"
      gpu_task_pid["${gpu}"]=""
      gpu_busy["${gpu}"]=0
      idle_seconds["${gpu}"]=0
    fi

    if (( task_index >= ${#TASKS[@]} )); then
      continue
    fi

    all_done=false
    if gpu_is_idle "${gpu}"; then
      idle_seconds["${gpu}"]=$((idle_seconds["${gpu}"] + POLL_SECONDS))
    else
      idle_seconds["${gpu}"]=0
    fi

    if (( idle_seconds["${gpu}"] >= IDLE_REQUIRED_SECONDS )); then
      task="${TASKS[${task_index}]}"
      task_index=$((task_index + 1))
      launch_task "${gpu}" "${task}"
      gpu_task_pid["${gpu}"]="$(cat "${LOG_ROOT}/$(task_slug "${task%|*}").pid" 2>/dev/null || true)"
      if [[ -z "${gpu_task_pid[${gpu}]}" ]]; then
        IFS='|' read -r task_site task_methods _ <<< "${task}"
        gpu_task_pid["${gpu}"]="$(cat "${LOG_ROOT}/$(task_slug "${task_site}|${task_methods}").pid")"
      fi
      gpu_busy["${gpu}"]=1
      idle_seconds["${gpu}"]=0
    fi
  done

  if [[ "${all_done}" == "true" && "${task_index}" -ge "${#TASKS[@]}" ]]; then
    log "[COMPLETE] all queued tasks finished"
    break
  fi

  sleep "${POLL_SECONDS}"
done
