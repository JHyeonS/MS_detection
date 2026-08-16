#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLBACKEND=Agg

GPUS_CSV="${RECONST_GEM_GPUS:-0,1,2,3,4,5,6}"
IDLE_MINUTES="${RECONST_GEM_IDLE_MINUTES:-0}"
POLL_SECONDS="${RECONST_GEM_POLL_SECONDS:-30}"
MEM_THRESHOLD_MB="${RECONST_GEM_MEM_THRESHOLD_MB:-1000}"
UTIL_THRESHOLD="${RECONST_GEM_UTIL_THRESHOLD:-10}"
FRACTIONS="${RECONST_GEM_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
RUN_ANALYZE="${RECONST_GEM_RUN_ANALYZE:-true}"
RUN_TSNE="${RECONST_GEM_RUN_TSNE:-false}"
POOLING_P="${RECONST_GEM_POOLING_P:-3.0}"
POOLING_CHANNELWISE="${RECONST_GEM_POOLING_CHANNELWISE:-true}"

RUN_ROOT_BASE="${RECONST_GEM_RUN_ROOT_BASE:-runs/reconst_gem_all_sweeps_v1}"
LOG_ROOT="${RECONST_GEM_LOG_ROOT:-logs/reconst_gem_all_sweeps_v1}"

mkdir -p "${RUN_ROOT_BASE}" "${LOG_ROOT}"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IDLE_REQUIRED_SECONDS=$((IDLE_MINUTES * 60))
SCHEDULER_LOG="${LOG_ROOT}/reconst_gem_all_sweeps.scheduler.log"

TASKS=(
  "sweep01|sweep01_current_fs1000_rms1p0_phlp50_ut19lp200_ut23lp500_3site|data/sweep01_current_fs1000_rms1p0_phlp50_ut19lp200_ut23lp500_3site/metadata/experiments_0406_mapped|pohang|stage1_pohang_only"
  "sweep01|sweep01_current_fs1000_rms1p0_phlp50_ut19lp200_ut23lp500_3site|data/sweep01_current_fs1000_rms1p0_phlp50_ut19lp200_ut23lp500_3site/metadata/experiments_0406_mapped|utah_2019|stage1_utah_2019_only"
  "sweep01|sweep01_current_fs1000_rms1p0_phlp50_ut19lp200_ut23lp500_3site|data/sweep01_current_fs1000_rms1p0_phlp50_ut19lp200_ut23lp500_3site/metadata/experiments_0406_mapped|utah_2023|stage1_utah_2023_only"
  "sweep02|sweep02_mid_fs1500_rms1p0_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275_3site|data/sweep02_mid_fs1500_rms1p0_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275_3site/metadata/experiments_0406_mapped|pohang|stage1_pohang_only"
  "sweep02|sweep02_mid_fs1500_rms1p0_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275_3site|data/sweep02_mid_fs1500_rms1p0_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275_3site/metadata/experiments_0406_mapped|utah_2019|stage1_utah_2019_only"
  "sweep02|sweep02_mid_fs1500_rms1p0_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275_3site|data/sweep02_mid_fs1500_rms1p0_phbp1p5-50_ut19bp1p5-125_ut23bp1p5-275_3site/metadata/experiments_0406_mapped|utah_2023|stage1_utah_2023_only"
  "sweep03|sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site|data/sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site/metadata|pohang|stage1_pohang_only"
  "sweep03|sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site|data/sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site/metadata|utah_2019|stage1_utah_2019_only"
  "sweep03|sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site|data/sweep03_oldfreq_fs2000_rms1p0_bp3-50_3site/metadata|utah_2023|stage1_utah_2023_only"
)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${SCHEDULER_LOG}"
}

task_slug() {
  printf '%s' "$1" | tr '|,' '__' | sed 's#[^A-Za-z0-9._-]#_#g'
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
  local sweep dataset experiment_root site split_name split_dir slug run_root log_file

  IFS='|' read -r sweep dataset experiment_root site split_name <<< "${task}"
  split_dir="${experiment_root}/${split_name}"
  slug="$(task_slug "${sweep}|${site}|reconst_gem")"
  run_root="${RUN_ROOT_BASE}/${slug}"
  log_file="${LOG_ROOT}/${slug}.stdout.log"

  if [[ ! -f "${split_dir}/train.csv" || ! -f "${split_dir}/pretrain.csv" || ! -f "${split_dir}/val.csv" || ! -f "${split_dir}/test.csv" ]]; then
    log "[ERROR] missing split files for task=${task}: ${split_dir}"
    return 1
  fi

  log "[LAUNCH] gpu=${gpu} sweep=${sweep} site=${site} methods=reconst pooling=gem split=${split_name} run_root=${run_root}"
  (
    export SITE_STUDY_SPLIT_DIR="${split_dir}"
    export SITE_STUDY_PREPROCESS="load_only"
    export SITE_STUDY_NORMALIZE="none"
    export SITE_STUDY_METHODS="reconst"
    export SITE_STUDY_FRACTIONS="${FRACTIONS}"
    export SITE_STUDY_RUN_ANALYZE="${RUN_ANALYZE}"
    export SITE_STUDY_RUN_TSNE="${RUN_TSNE}"
    export SITE_STUDY_RUN_PRETRAIN="true"
    export SITE_STUDY_POOLING="gem"
    export SITE_STUDY_POOLING_P="${POOLING_P}"
    export SITE_STUDY_POOLING_CHANNELWISE="${POOLING_CHANNELWISE}"
    export SITE_STUDY_LOG_SLUG="${dataset}_${sweep}_${site}_reconst_gem"
    bash scripts/gpu/site_main_study.sh "${site}" "${gpu}" "${LOG_ROOT}" "${run_root}"
  ) > "${log_file}" 2>&1 &

  echo "$!" > "${LOG_ROOT}/${slug}.pid"
}

task_index=0
declare -A idle_seconds
declare -A gpu_task_pid

for gpu in "${GPUS[@]}"; do
  idle_seconds["${gpu}"]=0
  gpu_task_pid["${gpu}"]=""
done

log "[START] reconst_gem_all_sweeps"
log "[CONFIG] gpus=${GPUS_CSV} idle_minutes=${IDLE_MINUTES} mem_threshold_mb=${MEM_THRESHOLD_MB} util_threshold=${UTIL_THRESHOLD}"
log "[CONFIG] fractions=${FRACTIONS} pooling=gem pooling_p=${POOLING_P} channelwise=${POOLING_CHANNELWISE}"
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
      IFS='|' read -r task_sweep _ _ task_site _ <<< "${task}"
      gpu_task_pid["${gpu}"]="$(cat "${LOG_ROOT}/$(task_slug "${task_sweep}|${task_site}|reconst_gem").pid")"
      idle_seconds["${gpu}"]=0
    fi
  done

  if [[ "${all_done}" == "true" && "${task_index}" -ge "${#TASKS[@]}" ]]; then
    log "[COMPLETE] all queued tasks finished"
    break
  fi

  sleep "${POLL_SECONDS}"
done
