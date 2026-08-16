#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-ms_detection}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

GPUS_CSV="${METADATA_V2_MISSING_GPUS:-0,1,2,3}"
LOG_ROOT="${METADATA_V2_MISSING_LOG_ROOT:-logs/metadata_v2_missing_only_gpu0123_v1}"
RUN_ROOT_PREFIX="${METADATA_V2_RUN_ROOT_PREFIX:-runs/metadata_v2_safe_rerun_v1}"
CACHE_MODE="${METADATA_V2_MISSING_CACHE_MODE:-ram}"
NUM_WORKERS="${METADATA_V2_MISSING_NUM_WORKERS:-0}"

mkdir -p "${LOG_ROOT}" "${MPLCONFIGDIR}"

OLD_IFS="${IFS}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IFS="${OLD_IFS}"

TASKS=(
  "logenv|visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5|utah_2023|reconst_noanom|1.00|stage1_utah_2023_only|utah_2023_reconst_reconst_noanom"
  "filter_rms|visualbest_filter_rms_fs1000_rms0p15_lp50|utah_2019|contrast|1.00|stage1_utah_2019_only|utah_2019_contrast"
  "filter_rms|visualbest_filter_rms_fs1000_rms0p15_lp50|utah_2019|reconst|1.00|stage1_utah_2019_only|utah_2019_reconst_reconst_noanom"
  "filter_rms|visualbest_filter_rms_fs1000_rms0p15_lp50|utah_2019|reconst_noanom|1.00|stage1_utah_2019_only|utah_2019_reconst_reconst_noanom"
  "filter_rms|visualbest_filter_rms_fs1000_rms0p15_lp50|utah_2023|scratch|0.25,0.50,1.00|stage1_utah_2023_only|utah_2023_scratch"
  "filter_rms|visualbest_filter_rms_fs1000_rms0p15_lp50|utah_2023|contrast|0.05,0.10,0.25,0.50,1.00|stage1_utah_2023_only|utah_2023_contrast"
  "filter_rms|visualbest_filter_rms_fs1000_rms0p15_lp50|utah_2023|reconst|0.05,0.10,0.25,0.50,1.00|stage1_utah_2023_only|utah_2023_reconst_reconst_noanom"
  "filter_rms|visualbest_filter_rms_fs1000_rms0p15_lp50|utah_2023|reconst_noanom|0.05,0.10,0.25,0.50,1.00|stage1_utah_2023_only|utah_2023_reconst_reconst_noanom"
)

slugify() {
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

frac_tag() {
  python - "$1" <<'PY'
import sys
x = float(sys.argv[1])
print(("{:.6f}".format(x)).rstrip("0").rstrip(".").replace(".", "p"))
PY
}

is_task_complete() {
  local run_root_base="$1"
  local site="$2"
  local method="$3"
  local fractions_csv="$4"
  local exp fraction tag old_ifs
  local -a fraction_array
  exp="$(experiment_name "${site}")"
  old_ifs="${IFS}"
  IFS=',' read -r -a fraction_array <<< "${fractions_csv}"
  IFS="${old_ifs}"
  for fraction in "${fraction_array[@]}"; do
    fraction="$(echo "${fraction}" | xargs)"
    tag="$(frac_tag "${fraction}")"
    if [[ ! -f "${run_root_base}/${method}/test/${exp}__frac${tag}/test_metrics_fixed_threshold.json" ]]; then
      return 1
    fi
  done
  return 0
}

launch_task() {
  local gpu="$1"
  local task="$2"
  local dataset_key dataset_dir site method fractions split_name group_slug split_dir run_root_base run_root log_slug log_file lock_dir old_ifs
  old_ifs="${IFS}"
  IFS='|' read -r dataset_key dataset_dir site method fractions split_name group_slug <<< "${task}"
  IFS="${old_ifs}"

  split_dir="data/${dataset_dir}/metadata/experiments/${split_name}"
  run_root_base="${RUN_ROOT_PREFIX}/${dataset_key}_site_main_pre50_v2/${group_slug}"
  run_root="${run_root_base}"
  log_slug="$(slugify "${dataset_key}|${site}|${method}|${fractions}")"
  log_file="${LOG_ROOT}/${log_slug}.stdout.log"
  lock_dir="${run_root}/.metadata_v2_missing_lock_${log_slug}"
  mkdir -p "${run_root}"

  if is_task_complete "${run_root_base}" "${site}" "${method}" "${fractions}"; then
    echo "[$(date '+%F %T')] [SKIP_COMPLETE] ${task}" | tee -a "${LOG_ROOT}/scheduler.log"
    return 0
  fi

  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "[$(date '+%F %T')] [SKIP_LOCKED] ${task}" | tee -a "${LOG_ROOT}/scheduler.log"
    return 0
  fi

  echo "[$(date '+%F %T')] [LAUNCH] gpu=${gpu} dataset=${dataset_key} site=${site} method=${method} fractions=${fractions} run_root=${run_root}" | tee -a "${LOG_ROOT}/scheduler.log"
  (
    trap 'rmdir "'"${lock_dir}"'" 2>/dev/null || true' EXIT
    export SITE_STUDY_SPLIT_DIR="${split_dir}"
    export SITE_STUDY_PREPROCESS="load_only"
    export SITE_STUDY_NORMALIZE="none"
    export SITE_STUDY_METHODS="${method}"
    export SITE_STUDY_FRACTIONS="${fractions}"
    export SITE_STUDY_RUN_ANALYZE="true"
    export SITE_STUDY_RUN_TSNE="false"
    export SITE_STUDY_RUN_PRETRAIN="true"
    export SITE_STUDY_LOG_CENTER_DIAGNOSTICS="false"
    export SITE_STUDY_LOG_WASSERSTEIN_DIAGNOSTICS="false"
    export SITE_STUDY_NUM_WORKERS="${NUM_WORKERS}"
    export SITE_STUDY_CACHE_MODE="${CACHE_MODE}"
    export SITE_STUDY_PRETRAIN_EPOCHS="50"
    export SITE_STUDY_PREFETCH_FACTOR="2"
    export SITE_STUDY_PERSISTENT_WORKERS="false"
    export SITE_STUDY_LOG_SLUG="metadata_v2_missing_${dataset_key}_${site}_${method}"
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
echo "[$(date '+%F %T')] [START] gpus=${GPUS_CSV} tasks=${#TASKS[@]} cache_mode=${CACHE_MODE} num_workers=${NUM_WORKERS}" | tee -a "${LOG_ROOT}/scheduler.log"

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
    echo "[$(date '+%F %T')] [COMPLETE] metadata v2 missing-only tasks finished" | tee -a "${LOG_ROOT}/scheduler.log"
    break
  fi
  sleep 60
done
