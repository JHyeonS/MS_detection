#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-ms_detection}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

GPUS_CSV="${METADATA_V2_GPUS:-0,1}"
LOG_ROOT="${METADATA_V2_LOG_ROOT:-logs/metadata_v2_safe_rerun_v1}"
RUN_ROOT_PREFIX="${METADATA_V2_RUN_ROOT_PREFIX:-runs/metadata_v2_safe_rerun_v1}"
FRACTIONS="${METADATA_V2_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
RUN_CROSS="${METADATA_V2_RUN_CROSS:-true}"

DATASETS=(
  "logenv|visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5"
  "filter_rms|visualbest_filter_rms_fs1000_rms0p15_lp50"
)

MAIN_TASKS=(
  "pohang|scratch|stage1_pohang_only|pohang_scratch"
  "pohang|contrast|stage1_pohang_only|pohang_contrast"
  "pohang|reconst,reconst_noanom|stage1_pohang_only|pohang_reconst_reconst_noanom"
  "utah_2019|scratch|stage1_utah_2019_only|utah_2019_scratch"
  "utah_2019|contrast|stage1_utah_2019_only|utah_2019_contrast"
  "utah_2019|reconst,reconst_noanom|stage1_utah_2019_only|utah_2019_reconst_reconst_noanom"
  "utah_2023|scratch|stage1_utah_2023_only|utah_2023_scratch"
  "utah_2023|contrast|stage1_utah_2023_only|utah_2023_contrast"
  "utah_2023|reconst,reconst_noanom|stage1_utah_2023_only|utah_2023_reconst_reconst_noanom"
)

mkdir -p "${LOG_ROOT}" "${RUN_ROOT_PREFIX}" "${MPLCONFIGDIR}"

OLD_IFS="${IFS}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IFS="${OLD_IFS}"

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

is_main_task_complete() {
  local run_root_base="$1"
  local site="$2"
  local methods_csv="$3"
  local group_slug="$4"
  local exp method fraction tag old_ifs
  local -a method_array
  local -a fraction_array
  exp="$(experiment_name "${site}")"
  old_ifs="${IFS}"
  IFS=',' read -r -a method_array <<< "${methods_csv}"
  IFS=',' read -r -a fraction_array <<< "${FRACTIONS}"
  IFS="${old_ifs}"
  for method in "${method_array[@]}"; do
    method="$(echo "${method}" | xargs)"
    for fraction in "${fraction_array[@]}"; do
      tag="$(frac_tag "${fraction}")"
      if [[ ! -f "${run_root_base}/${group_slug}/${method}/test/${exp}__frac${tag}/test_metrics_fixed_threshold.json" ]]; then
        return 1
      fi
    done
  done
  return 0
}

launch_main_task() {
  local gpu="$1"
  local dataset_key="$2"
  local dataset_dir="$3"
  local task="$4"
  local site methods split_name group_slug split_dir run_root_base run_root log_slug log_file lock_dir old_ifs
  old_ifs="${IFS}"
  IFS='|' read -r site methods split_name group_slug <<< "${task}"
  IFS="${old_ifs}"

  split_dir="data/${dataset_dir}/metadata/experiments/${split_name}"
  run_root_base="${RUN_ROOT_PREFIX}/${dataset_key}_site_main_pre50_v2"
  run_root="${run_root_base}/${group_slug}"
  log_slug="$(slugify "${dataset_key}|${site}|${methods}|${FRACTIONS}")"
  log_file="${LOG_ROOT}/${log_slug}.stdout.log"
  lock_dir="${run_root}/.metadata_v2_lock_${log_slug}"
  mkdir -p "${run_root}"

  if is_main_task_complete "${run_root_base}" "${site}" "${methods}" "${group_slug}"; then
    echo "[$(date '+%F %T')] [SKIP_COMPLETE] main ${dataset_key} ${task}" | tee -a "${LOG_ROOT}/scheduler.log"
    return 0
  fi

  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "[$(date '+%F %T')] [SKIP_LOCKED] main ${dataset_key} ${task}" | tee -a "${LOG_ROOT}/scheduler.log"
    return 0
  fi

  echo "[$(date '+%F %T')] [LAUNCH_MAIN] gpu=${gpu} dataset=${dataset_key} site=${site} methods=${methods} run_root=${run_root}" | tee -a "${LOG_ROOT}/scheduler.log"
  (
    trap 'rmdir "'"${lock_dir}"'" 2>/dev/null || true' EXIT
    export SITE_STUDY_SPLIT_DIR="${split_dir}"
    export SITE_STUDY_PREPROCESS="load_only"
    export SITE_STUDY_NORMALIZE="none"
    export SITE_STUDY_METHODS="${methods}"
    export SITE_STUDY_FRACTIONS="${FRACTIONS}"
    export SITE_STUDY_RUN_ANALYZE="true"
    export SITE_STUDY_RUN_TSNE="false"
    export SITE_STUDY_RUN_PRETRAIN="true"
    export SITE_STUDY_LOG_CENTER_DIAGNOSTICS="false"
    export SITE_STUDY_LOG_WASSERSTEIN_DIAGNOSTICS="false"
    export SITE_STUDY_NUM_WORKERS="1"
    export SITE_STUDY_CACHE_MODE="ram"
    export SITE_STUDY_PRETRAIN_EPOCHS="50"
    export SITE_STUDY_PREFETCH_FACTOR="2"
    export SITE_STUDY_PERSISTENT_WORKERS="false"
    export SITE_STUDY_LOG_SLUG="metadata_v2_${dataset_key}_${site}_${group_slug}"
    bash scripts/gpu/site_main_study.sh "${site}" "${gpu}" "${LOG_ROOT}" "${run_root}"
  ) > "${log_file}" 2>&1 &
  echo "$!"
}

run_scheduler() {
  local phase="$1"
  shift
  local -a tasks=("$@")
  declare -A gpu_pid
  declare -A gpu_task
  local gpu pid task task_index all_idle

  for gpu in "${GPUS[@]}"; do
    gpu_pid["${gpu}"]=""
    gpu_task["${gpu}"]=""
  done

  task_index=0
  echo "[$(date '+%F %T')] [PHASE_START] ${phase} gpus=${GPUS_CSV} tasks=${#tasks[@]}" | tee -a "${LOG_ROOT}/scheduler.log"
  while true; do
    all_idle=true
    for gpu in "${GPUS[@]}"; do
      pid="${gpu_pid[${gpu}]}"
      if [[ -n "${pid}" ]]; then
        if kill -0 "${pid}" 2>/dev/null; then
          all_idle=false
          continue
        fi
        echo "[$(date '+%F %T')] [DONE] phase=${phase} gpu=${gpu} pid=${pid} task=${gpu_task[${gpu}]}" | tee -a "${LOG_ROOT}/scheduler.log"
        gpu_pid["${gpu}"]=""
        gpu_task["${gpu}"]=""
      fi

      if (( task_index < ${#tasks[@]} )); then
        task="${tasks[${task_index}]}"
        task_index=$((task_index + 1))
        pid="$(eval "${task}" | tail -n 1)"
        if [[ "${pid}" =~ ^[0-9]+$ ]]; then
          gpu_pid["${gpu}"]="${pid}"
          gpu_task["${gpu}"]="${task}"
          all_idle=false
        fi
      fi
    done

    if [[ "${all_idle}" == "true" && "${task_index}" -ge "${#tasks[@]}" ]]; then
      echo "[$(date '+%F %T')] [PHASE_COMPLETE] ${phase}" | tee -a "${LOG_ROOT}/scheduler.log"
      break
    fi
    sleep 60
  done
}

build_main_commands() {
  local commands=()
  local dataset_spec dataset_key dataset_dir task old_ifs
  for dataset_spec in "${DATASETS[@]}"; do
    old_ifs="${IFS}"
    IFS='|' read -r dataset_key dataset_dir <<< "${dataset_spec}"
    IFS="${old_ifs}"
    for task in "${MAIN_TASKS[@]}"; do
      commands+=("launch_main_task \"\${gpu}\" \"${dataset_key}\" \"${dataset_dir}\" \"${task}\"")
    done
  done
  printf '%s\n' "${commands[@]}"
}

run_cross_dataset() {
  local dataset_key="$1"
  local dataset_dir="$2"
  local gpu_csv="$3"
  local source_root="${RUN_ROOT_PREFIX}/${dataset_key}_site_main_pre50_v2"
  local cross_root="${RUN_ROOT_PREFIX}/${dataset_key}_cross_site_reconst_pre50_v2"
  local cross_logs="${LOG_ROOT}/${dataset_key}_cross"
  echo "[$(date '+%F %T')] [CROSS_START] dataset=${dataset_key} gpus=${gpu_csv}" | tee -a "${LOG_ROOT}/scheduler.log"
  CROSS_RECONST_GPUS="${gpu_csv}" \
  CROSS_RECONST_DATASET="${dataset_dir}" \
  CROSS_RECONST_SOURCE_PRETRAIN_ROOT="${source_root}" \
  CROSS_RECONST_RUN_ROOT="${cross_root}" \
  CROSS_RECONST_LOG_ROOT="${cross_logs}" \
  CROSS_RECONST_FRACTIONS="${FRACTIONS}" \
  CROSS_RECONST_CACHE_MODE="ram" \
  CROSS_RECONST_NUM_WORKERS="1" \
  CROSS_RECONST_PERSISTENT_WORKERS="false" \
  CROSS_RECONST_PREFETCH_FACTOR="2" \
    bash scripts/gpu/run_logenv_cross_site_reconst_gpu24.sh
  echo "[$(date '+%F %T')] [CROSS_COMPLETE] dataset=${dataset_key}" | tee -a "${LOG_ROOT}/scheduler.log"
}

main() {
  local dataset_spec dataset_key dataset_dir old_ifs
  mapfile -t main_commands < <(build_main_commands)
  run_scheduler "main_in_domain" "${main_commands[@]}"

  if [[ "${RUN_CROSS}" != "true" ]]; then
    echo "[$(date '+%F %T')] [COMPLETE] metadata v2 safe rerun main-only finished" | tee -a "${LOG_ROOT}/scheduler.log"
    return 0
  fi

  for dataset_spec in "${DATASETS[@]}"; do
    old_ifs="${IFS}"
    IFS='|' read -r dataset_key dataset_dir <<< "${dataset_spec}"
    IFS="${old_ifs}"
    run_cross_dataset "${dataset_key}" "${dataset_dir}" "${GPUS_CSV}"
  done

  echo "[$(date '+%F %T')] [COMPLETE] metadata v2 safe rerun finished" | tee -a "${LOG_ROOT}/scheduler.log"
}

main "$@"
