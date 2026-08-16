#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLBACKEND=Agg

GPUS_CSV="${VISUALBEST_GPUS:-0,1,2,3,4,5,6,7}"
FRACTIONS="${VISUALBEST_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
RUN_ANALYZE="${VISUALBEST_RUN_ANALYZE:-true}"
RUN_TSNE="${VISUALBEST_RUN_TSNE:-false}"
RUN_PRETRAIN="${VISUALBEST_RUN_PRETRAIN:-true}"
LOG_CENTER_DIAGNOSTICS="${VISUALBEST_LOG_CENTER_DIAGNOSTICS:-true}"
LOG_WASSERSTEIN_DIAGNOSTICS="${VISUALBEST_LOG_WASSERSTEIN_DIAGNOSTICS:-true}"
WASSERSTEIN_NUM_PROJECTIONS="${VISUALBEST_WASSERSTEIN_NUM_PROJECTIONS:-32}"
WASSERSTEIN_NUM_QUANTILES="${VISUALBEST_WASSERSTEIN_NUM_QUANTILES:-128}"
NUM_WORKERS="${VISUALBEST_NUM_WORKERS:-0}"
LOG_ROOT="${VISUALBEST_PARALLEL_LOG_ROOT:-logs/visualbest_remaining_parallel_v1}"

mkdir -p "${LOG_ROOT}"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if (( ${#GPUS[@]} == 0 )); then
  echo "[ERROR] VISUALBEST_GPUS is empty"
  exit 1
fi

TASKS=(
  "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5|runs/visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5_site_main_v1|utah_2019|scratch|stage1_utah_2019_only"
  "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5|runs/visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5_site_main_v1|utah_2019|contrast|stage1_utah_2019_only"
  "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5|runs/visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5_site_main_v1|utah_2019|reconst+reconst_noanom|stage1_utah_2019_only"
  "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5|runs/visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5_site_main_v1|utah_2023|scratch|stage1_utah_2023_only"
  "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5|runs/visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5_site_main_v1|utah_2023|contrast|stage1_utah_2023_only"
  "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5|runs/visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5_site_main_v1|utah_2023|reconst+reconst_noanom|stage1_utah_2023_only"
  "visualbest_filter_rms_fs1000_rms0p15_lp50|runs/visualbest_filter_rms_fs1000_rms0p15_lp50_site_main_v1|utah_2019|scratch|stage1_utah_2019_only"
  "visualbest_filter_rms_fs1000_rms0p15_lp50|runs/visualbest_filter_rms_fs1000_rms0p15_lp50_site_main_v1|utah_2019|contrast|stage1_utah_2019_only"
  "visualbest_filter_rms_fs1000_rms0p15_lp50|runs/visualbest_filter_rms_fs1000_rms0p15_lp50_site_main_v1|utah_2019|reconst+reconst_noanom|stage1_utah_2019_only"
  "visualbest_filter_rms_fs1000_rms0p15_lp50|runs/visualbest_filter_rms_fs1000_rms0p15_lp50_site_main_v1|utah_2023|scratch|stage1_utah_2023_only"
  "visualbest_filter_rms_fs1000_rms0p15_lp50|runs/visualbest_filter_rms_fs1000_rms0p15_lp50_site_main_v1|utah_2023|contrast|stage1_utah_2023_only"
  "visualbest_filter_rms_fs1000_rms0p15_lp50|runs/visualbest_filter_rms_fs1000_rms0p15_lp50_site_main_v1|utah_2023|reconst+reconst_noanom|stage1_utah_2023_only"
)

task_slug() {
  printf '%s' "$1" | tr ',|' '__' | sed 's#[^A-Za-z0-9._-]#_#g'
}

experiment_name() {
  case "$1" in
    pohang) echo "pohang" ;;
    utah_2019) echo "base_utah_2019" ;;
    utah_2023) echo "base_utah_2023" ;;
    *) echo "[ERROR] unsupported site: $1" >&2; return 1 ;;
  esac
}

is_group_complete() {
  local run_root="$1"
  local site="$2"
  local methods_csv="$3"
  local exp
  exp="$(experiment_name "${site}")"

  local method_array
  IFS=',' read -r -a method_array <<< "${methods_csv}"
  for method in "${method_array[@]}"; do
    method="$(echo "${method}" | xargs)"
    for frac in 0p05 0p1 0p25 0p5 1; do
      if [[ ! -f "${run_root}/${method}/test/${exp}__frac${frac}/test_metrics_fixed_threshold.json" ]]; then
        return 1
      fi
    done
  done
  return 0
}

launch_task() {
  local gpu="$1"
  local task="$2"
  local dataset run_root_base site methods split_name split_dir run_root slug log_file lock_dir
  IFS='|' read -r dataset run_root_base site methods split_name <<< "${task}"
  methods="${methods//+/,}"

  slug="$(task_slug "${site}|${methods}")"
  split_dir="data/${dataset}/metadata/experiments/${split_name}"
  run_root="${run_root_base}/${slug}"
  log_file="${LOG_ROOT}/${dataset}__${slug}.stdout.log"
  lock_dir="${run_root}/.distributed_lock"
  mkdir -p "${run_root}"

  if is_group_complete "${run_root}" "${site}" "${methods}"; then
    echo "[$(date '+%F %T')] [SKIP_COMPLETE] ${dataset} ${slug}" | tee -a "${LOG_ROOT}/scheduler.log"
    return 0
  fi

  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "[$(date '+%F %T')] [SKIP_LOCKED] ${dataset} ${slug} lock=${lock_dir}" | tee -a "${LOG_ROOT}/scheduler.log"
    return 0
  fi

  echo "[$(date '+%F %T')] [LAUNCH] gpu=${gpu} dataset=${dataset} site=${site} methods=${methods} run_root=${run_root}" | tee -a "${LOG_ROOT}/scheduler.log"
  (
    trap 'rmdir "'"${lock_dir}"'" 2>/dev/null || true' EXIT
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
    export SITE_STUDY_NUM_WORKERS="${NUM_WORKERS}"
    export SITE_STUDY_LOG_SLUG="${dataset}_${slug}_distributed"
    bash scripts/gpu/site_main_study.sh "${site}" "${gpu}" "${LOG_ROOT}" "${run_root}"
  ) > "${log_file}" 2>&1 &

  echo "$!"
}

declare -A gpu_pid
declare -A gpu_task
for gpu in "${GPUS[@]}"; do
  gpu_pid["${gpu}"]=""
  gpu_task["${gpu}"]=""
done

task_index=0
echo "[$(date '+%F %T')] [START] gpus=${GPUS_CSV} tasks=${#TASKS[@]} swd=${LOG_WASSERSTEIN_DIAGNOSTICS} num_workers=${NUM_WORKERS}" | tee -a "${LOG_ROOT}/scheduler.log"

while true; do
  all_idle=true
  for gpu in "${GPUS[@]}"; do
    pid="${gpu_pid[${gpu}]}"
    if [[ -n "${pid}" ]]; then
      if kill -0 "${pid}" 2>/dev/null; then
        all_idle=false
        continue
      fi
      echo "[$(date '+%F %T')] [DONE] gpu=${gpu} pid=${pid} task=${gpu_task[${gpu}]}" | tee -a "${LOG_ROOT}/scheduler.log"
      gpu_pid["${gpu}"]=""
      gpu_task["${gpu}"]=""
    fi

    if (( task_index < ${#TASKS[@]} )); then
      task="${TASKS[${task_index}]}"
      task_index=$((task_index + 1))
      pid="$(launch_task "${gpu}" "${task}" | tail -n 1)"
      if [[ "${pid}" =~ ^[0-9]+$ ]]; then
        gpu_pid["${gpu}"]="${pid}"
        gpu_task["${gpu}"]="${task}"
        all_idle=false
      fi
    fi
  done

  if [[ "${all_idle}" == "true" && "${task_index}" -ge "${#TASKS[@]}" ]]; then
    echo "[$(date '+%F %T')] [COMPLETE] all distributed visualbest tasks finished" | tee -a "${LOG_ROOT}/scheduler.log"
    break
  fi
  sleep 60
done
