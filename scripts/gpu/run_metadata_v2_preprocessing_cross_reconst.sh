#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLBACKEND=Agg
export LD_LIBRARY_PATH=/home/anaconda3/lib:${LD_LIBRARY_PATH:-}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "/home/ted1204/.conda/envs/ms_detection/bin/python" ]]; then
    PYTHON_BIN="/home/ted1204/.conda/envs/ms_detection/bin/python"
  elif [[ -x "/home/anaconda3/bin/python3.9" ]]; then
    PYTHON_BIN="/home/anaconda3/bin/python3.9"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[ERROR] could not resolve PYTHON_BIN" >&2
    exit 1
  fi
fi

GPUS_CSV="${PREPROC_CROSS_GPUS:-0}"
FRACTIONS="${PREPROC_CROSS_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
RUN_ROOT_PREFIX="${METADATA_V2_RUN_ROOT_PREFIX:-runs/metadata_v2_safe_rerun_v1}"
RUN_ROOT_BASE="${PREPROC_CROSS_RUN_ROOT:-${RUN_ROOT_PREFIX}/preprocessing_cross_reconst_pre50_v1}"
LOG_ROOT="${PREPROC_CROSS_LOG_ROOT:-logs/metadata_v2_preprocessing_cross_reconst_pre50_v1}"
TASKS_CSV="${PREPROC_CROSS_TASKS:-}"
DATA_CACHE_MODE="${PREPROC_CROSS_CACHE_MODE:-ram}"
DATA_NUM_WORKERS="${PREPROC_CROSS_NUM_WORKERS:-1}"
DATA_PERSISTENT_WORKERS="${PREPROC_CROSS_PERSISTENT_WORKERS:-false}"
DATA_PREFETCH_FACTOR="${PREPROC_CROSS_PREFETCH_FACTOR:-2}"
LOG_CENTER_DIAGNOSTICS="${PREPROC_CROSS_LOG_CENTER_DIAGNOSTICS:-false}"
LOG_WASSERSTEIN_DIAGNOSTICS="${PREPROC_CROSS_LOG_WASSERSTEIN_DIAGNOSTICS:-false}"
CENTER_DIAGNOSTICS_INTERVAL="${PREPROC_CROSS_CENTER_DIAGNOSTICS_INTERVAL:-10}"
WASSERSTEIN_NUM_PROJECTIONS="${PREPROC_CROSS_WASSERSTEIN_NUM_PROJECTIONS:-32}"
WASSERSTEIN_NUM_QUANTILES="${PREPROC_CROSS_WASSERSTEIN_NUM_QUANTILES:-128}"

mkdir -p "${RUN_ROOT_BASE}" "${LOG_ROOT}"
LOCK_ROOT="${RUN_ROOT_BASE}/.locks"
mkdir -p "${LOCK_ROOT}"

OLD_IFS="${IFS}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IFS="${OLD_IFS}"

TASKS=(
  "pohang|filter_rms|logenv"
  "pohang|logenv|filter_rms"
  "utah_2019|filter_rms|logenv"
  "utah_2019|logenv|filter_rms"
  "utah_2023|filter_rms|logenv"
  "utah_2023|logenv|filter_rms"
)

if [[ -n "${TASKS_CSV}" ]]; then
  OLD_IFS="${IFS}"
  IFS=',' read -r -a TASKS <<< "${TASKS_CSV}"
  IFS="${OLD_IFS}"
fi

dataset_dir() {
  case "$1" in
    logenv) echo "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5" ;;
    filter_rms) echo "visualbest_filter_rms_fs1000_rms0p15_lp50" ;;
    raw) echo "visualbest_raw_rms_fs1000_rms0p15_nofilter" ;;
    *) echo "[ERROR] unsupported preprocessing: $1" >&2; return 1 ;;
  esac
}

site_experiment() {
  case "$1" in
    pohang) echo "pohang" ;;
    utah_2019) echo "base_utah_2019" ;;
    utah_2023) echo "base_utah_2023" ;;
    *) echo "[ERROR] unsupported site: $1" >&2; return 1 ;;
  esac
}

site_split_name() {
  case "$1" in
    pohang) echo "stage1_pohang_only" ;;
    utah_2019) echo "stage1_utah_2019_only" ;;
    utah_2023) echo "stage1_utah_2023_only" ;;
    *) echo "[ERROR] unsupported site: $1" >&2; return 1 ;;
  esac
}

site_base_template() {
  case "$1" in
    pohang) echo "configs/train/base_pohang_arch_best.yaml" ;;
    utah_2019) echo "configs/train/base_utah_2019_arch_best.yaml" ;;
    utah_2023) echo "configs/train/base_utah_2023_arch_best.yaml" ;;
    *) echo "[ERROR] unsupported site: $1" >&2; return 1 ;;
  esac
}

site_finetune_template() {
  case "$1" in
    pohang) echo "configs/train/final_pohang_best.yaml" ;;
    utah_2019) echo "configs/train/final_utah_2019_best.yaml" ;;
    utah_2023) echo "configs/train/final_utah_2023_best.yaml" ;;
    *) echo "[ERROR] unsupported site: $1" >&2; return 1 ;;
  esac
}

site_anomaly_weight() {
  case "$1" in
    pohang) echo "0.05" ;;
    utah_2019) echo "0.01" ;;
    utah_2023) echo "0.3" ;;
    *) echo "[ERROR] unsupported site: $1" >&2; return 1 ;;
  esac
}

preproc_main_root() {
  case "$1" in
    logenv) echo "${RUN_ROOT_PREFIX}/logenv_site_main_pre50_v2" ;;
    filter_rms) echo "${RUN_ROOT_PREFIX}/filter_rms_site_main_pre50_v2" ;;
    raw) echo "${RUN_ROOT_PREFIX}/raw_site_main_pre50_v1" ;;
    *) echo "[ERROR] unsupported preprocessing: $1" >&2; return 1 ;;
  esac
}

source_encoder() {
  local site="$1"
  local preproc="$2"
  local root
  root="$(preproc_main_root "${preproc}")"
  if [[ "${preproc}" == "raw" ]]; then
    case "${site}" in
      pohang)
        echo "${root}/pohang/reconst/pretrain/pohang/best_encoder.pt"
        ;;
      utah_2019)
        echo "${root}/utah_2019/reconst/pretrain/base_utah_2019/best_encoder.pt"
        ;;
      utah_2023)
        echo "${root}/utah_2023/reconst/pretrain/base_utah_2023/best_encoder.pt"
        ;;
      *) echo "[ERROR] unsupported site: ${site}" >&2; return 1 ;;
    esac
    return
  fi
  case "${site}" in
    pohang)
      echo "${root}/pohang_reconst_reconst_noanom/reconst/pretrain/pohang/best_encoder.pt"
      ;;
    utah_2019)
      echo "${root}/utah_2019_reconst_reconst_noanom/reconst/pretrain/base_utah_2019/best_encoder.pt"
      ;;
    utah_2023)
      echo "${root}/utah_2023_reconst_reconst_noanom/reconst/pretrain/base_utah_2023/best_encoder.pt"
      ;;
    *) echo "[ERROR] unsupported site: ${site}" >&2; return 1 ;;
  esac
}

frac_tag() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
x = float(sys.argv[1])
print(("{:.6f}".format(x)).rstrip("0").rstrip(".").replace(".", "p"))
PY
}

all_tests_complete() {
  local site="$1"
  local source_preproc="$2"
  local target_preproc="$3"
  local target_exp fraction tag test_path old_ifs
  local -a fraction_array
  target_exp="$(site_experiment "${site}")"
  old_ifs="${IFS}"
  IFS=',' read -r -a fraction_array <<< "${FRACTIONS}"
  IFS="${old_ifs}"
  for fraction in "${fraction_array[@]}"; do
    fraction="$(echo "${fraction}" | xargs)"
    tag="$(frac_tag "${fraction}")"
    test_path="${RUN_ROOT_BASE}/${site}/${source_preproc}_to_${target_preproc}/reconst/test/${target_exp}__frac${tag}/test_metrics_fixed_threshold.json"
    if [[ ! -f "${test_path}" ]]; then
      return 1
    fi
  done
  return 0
}

make_base_cfg() {
  local site="$1"
  local target_preproc="$2"
  local run_root="$3"
  local out_cfg="$4"
  local template split_name split_dir target_dataset
  template="$(site_base_template "${site}")"
  split_name="$(site_split_name "${site}")"
  target_dataset="$(dataset_dir "${target_preproc}")"
  split_dir="data/${target_dataset}/metadata/experiments/${split_name}"
  "${PYTHON_BIN}" - "${template}" "${run_root}" "${split_dir}" "${out_cfg}" <<'PY'
import os
import sys
import yaml

src, run_root, split_dir, outp = sys.argv[1:5]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
data = cfg.setdefault("data", {})
data["split_dir"] = split_dir
data["num_workers"] = int(os.environ.get("PREPROC_CROSS_NUM_WORKERS", "1"))
data["cache_mode"] = os.environ.get("PREPROC_CROSS_CACHE_MODE", "ram")
data["persistent_workers"] = os.environ.get("PREPROC_CROSS_PERSISTENT_WORKERS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
data["prefetch_factor"] = int(os.environ.get("PREPROC_CROSS_PREFETCH_FACTOR", "2"))
data["normalize"] = "none"
pre = data.setdefault("preprocess", {})
pre["load_only"] = True
pre["detrend"] = False
pre["bandpass"] = False
pre["agc"] = False
data["preprocess_variant"] = "load_only"
enc = cfg.setdefault("model", {}).setdefault("encoder", {})
enc["pooling"] = "avg"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_finetune_cfg() {
  local site="$1"
  local run_root="$2"
  local fraction="$3"
  local encoder="$4"
  local out_cfg="$5"
  local template anomaly_weight
  template="$(site_finetune_template "${site}")"
  anomaly_weight="$(site_anomaly_weight "${site}")"
  "${PYTHON_BIN}" - "${template}" "${run_root}" "${fraction}" "${encoder}" "${anomaly_weight}" "${out_cfg}" <<'PY'
import os
import sys
import yaml

src, run_root, fraction, encoder, anomaly_weight, outp = sys.argv[1:7]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
data = cfg.setdefault("data", {})
data["num_workers"] = int(os.environ.get("PREPROC_CROSS_NUM_WORKERS", "1"))
data["cache_mode"] = os.environ.get("PREPROC_CROSS_CACHE_MODE", "ram")
data["persistent_workers"] = os.environ.get("PREPROC_CROSS_PERSISTENT_WORKERS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
data["prefetch_factor"] = int(os.environ.get("PREPROC_CROSS_PREFETCH_FACTOR", "2"))
data["normalize"] = "none"
pre = data.setdefault("preprocess", {})
pre["load_only"] = True
pre["detrend"] = False
pre["bandpass"] = False
pre["agc"] = False
data["preprocess_variant"] = "load_only"
train = cfg.setdefault("train", {})
fraction_f = float(fraction)
train["use_labeled_fraction"] = fraction_f < 1.0
train["labeled_fraction"] = fraction_f
train["balance_fraction_by_class"] = True
train["min_samples_per_class"] = 1
train["seed"] = 42
train["fraction_seed"] = 42
train["drop_last"] = False
train["use_pretrained_encoder"] = True
train["pretrained_encoder_path"] = encoder
train["anomaly_loss_weight"] = float(anomaly_weight)
train["log_center_diagnostics"] = os.environ.get("PREPROC_CROSS_LOG_CENTER_DIAGNOSTICS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
train["log_wasserstein_diagnostics"] = os.environ.get("PREPROC_CROSS_LOG_WASSERSTEIN_DIAGNOSTICS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
train["center_diagnostics_interval"] = int(os.environ.get("PREPROC_CROSS_CENTER_DIAGNOSTICS_INTERVAL", "10"))
train["wasserstein_num_projections"] = int(os.environ.get("PREPROC_CROSS_WASSERSTEIN_NUM_PROJECTIONS", "32"))
train["wasserstein_num_quantiles"] = int(os.environ.get("PREPROC_CROSS_WASSERSTEIN_NUM_QUANTILES", "128"))
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_eval_base_cfg() {
  local base_cfg="$1"
  local suffix="$2"
  local out_cfg="$3"
  "${PYTHON_BIN}" - "${base_cfg}" "${suffix}" "${out_cfg}" <<'PY'
import sys
import yaml
src, suffix, outp = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg["data"]["experiment"] = f"{cfg['data']['experiment']}__{suffix}"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_test_cfg() {
  local out_cfg="$1"
  "${PYTHON_BIN}" - "configs/train/test.yaml" "${out_cfg}" <<'PY'
import os
import sys
import yaml
src, outp = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
data = cfg.setdefault("data", {})
data["num_workers"] = int(os.environ.get("PREPROC_CROSS_NUM_WORKERS", "1"))
data["cache_mode"] = os.environ.get("PREPROC_CROSS_CACHE_MODE", "ram")
data["persistent_workers"] = os.environ.get("PREPROC_CROSS_PERSISTENT_WORKERS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
data["prefetch_factor"] = int(os.environ.get("PREPROC_CROSS_PREFETCH_FACTOR", "2"))
data["normalize"] = "none"
pre = data.setdefault("preprocess", {})
pre["load_only"] = True
pre["detrend"] = False
pre["bandpass"] = False
pre["agc"] = False
data["preprocess_variant"] = "load_only"
cfg.setdefault("test", {})["num_workers"] = int(os.environ.get("PREPROC_CROSS_NUM_WORKERS", "1"))
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

run_task() {
  local gpu="$1"
  local task="$2"
  local site source_preproc target_preproc encoder target_exp run_root tmp_dir log_slug task_log base_cfg test_cfg old_ifs lock_dir
  local -a fraction_array
  old_ifs="${IFS}"
  IFS='|' read -r site source_preproc target_preproc <<< "${task}"
  IFS="${old_ifs}"

  if [[ "${source_preproc}" == "${target_preproc}" ]]; then
    echo "[ERROR] source and target preprocessing must differ: ${task}" >&2
    return 1
  fi

  run_root="${RUN_ROOT_BASE}/${site}/${source_preproc}_to_${target_preproc}/reconst"
  tmp_dir=".tmp_preprocessing_cross_reconst_${site}_${source_preproc}_to_${target_preproc}"
  log_slug="${site}__${source_preproc}_to_${target_preproc}"
  task_log="${LOG_ROOT}/${log_slug}.log"
  base_cfg="${tmp_dir}/base.yaml"
  test_cfg="${tmp_dir}/test.yaml"
  target_exp="$(site_experiment "${site}")"
  encoder="$(source_encoder "${site}" "${source_preproc}")"
  mkdir -p "${run_root}" "${tmp_dir}"

  if [[ ! -f "${encoder}" ]]; then
    echo "[ERROR] missing ${source_preproc} encoder for ${site}: ${encoder}" | tee -a "${task_log}"
    return 1
  fi

  if all_tests_complete "${site}" "${source_preproc}" "${target_preproc}"; then
    echo "[$(date '+%F %T')] [SKIP_COMPLETE] ${site} ${source_preproc}->${target_preproc}" | tee -a "${LOG_ROOT}/scheduler.log"
    return 0
  fi

  lock_dir="${LOCK_ROOT}/${site}__${source_preproc}_to_${target_preproc}.lock"
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "[$(date '+%F %T')] [SKIP_LOCKED] ${site} ${source_preproc}->${target_preproc}" | tee -a "${LOG_ROOT}/scheduler.log" | tee -a "${task_log}"
    return 0
  fi
  trap 'rmdir "'"${lock_dir}"'" 2>/dev/null || true' EXIT

  echo "[$(date '+%F %T')] [TASK_START] gpu=${gpu} site=${site} preprocessing=${source_preproc}->${target_preproc} encoder=${encoder}" | tee -a "${LOG_ROOT}/scheduler.log" | tee -a "${task_log}"
  make_base_cfg "${site}" "${target_preproc}" "${run_root}" "${base_cfg}"
  make_test_cfg "${test_cfg}"

  old_ifs="${IFS}"
  IFS=',' read -r -a fraction_array <<< "${FRACTIONS}"
  IFS="${old_ifs}"

  for fraction in "${fraction_array[@]}"; do
    fraction="$(echo "${fraction}" | xargs)"
    tag="frac$(frac_tag "${fraction}")"
    local test_json="${run_root}/test/${target_exp}__${tag}/test_metrics_fixed_threshold.json"
    local ft_cfg="${tmp_dir}/finetune_${tag}.yaml"
    local eval_cfg="${tmp_dir}/eval_${tag}.yaml"
    local run_log="${LOG_ROOT}/${log_slug}__${tag}.log"

    if [[ -f "${test_json}" ]]; then
      echo "[SKIP] existing test ${site} ${source_preproc}->${target_preproc} ${tag}" | tee -a "${task_log}" | tee -a "${run_log}"
      continue
    fi

    make_finetune_cfg "${site}" "${run_root}" "${fraction}" "${encoder}" "${ft_cfg}"
    make_eval_base_cfg "${base_cfg}" "${tag}" "${eval_cfg}"

    {
      echo "============================================================"
      echo "[PREPROCESSING CROSS RECONST] $(date '+%F %T') site=${site} source_preproc=${source_preproc} target_preproc=${target_preproc} fraction=${fraction} gpu=${gpu}"
      echo "run_root=${run_root}"
      echo "encoder=${encoder}"
      echo "base_cfg=${base_cfg}"
      echo "finetune_cfg=${ft_cfg}"
      echo "eval_cfg=${eval_cfg}"
      echo "============================================================"
    } | tee -a "${task_log}" | tee -a "${run_log}"

    echo "[FINETUNE] ${site} ${source_preproc}->${target_preproc} ${tag}" | tee -a "${task_log}" | tee -a "${run_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
      --base_cfg "${base_cfg}" \
      --stage_cfg "${ft_cfg}" \
      --exp_suffix "${tag}" >> "${run_log}" 2>&1

    echo "[TEST] ${site} ${source_preproc}->${target_preproc} ${tag}" | tee -a "${task_log}" | tee -a "${run_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
      --base_cfg "${eval_cfg}" \
      --stage_cfg "${test_cfg}" >> "${run_log}" 2>&1

    echo "[DONE] ${site} ${source_preproc}->${target_preproc} ${tag}" | tee -a "${task_log}" | tee -a "${run_log}"
  done

  "${PYTHON_BIN}" scripts/gpu/summarize_pohang_main_study.py \
    --root "${run_root}" \
    --out "${RUN_ROOT_BASE}/${site}/${source_preproc}_to_${target_preproc}/summary.csv" >> "${task_log}" 2>&1
  echo "[$(date '+%F %T')] [TASK_DONE] gpu=${gpu} site=${site} preprocessing=${source_preproc}->${target_preproc}" | tee -a "${LOG_ROOT}/scheduler.log" | tee -a "${task_log}"
}

declare -A gpu_pid
declare -A gpu_task
for gpu in "${GPUS[@]}"; do
  gpu_pid["${gpu}"]=""
  gpu_task["${gpu}"]=""
done

task_index=0
echo "[$(date '+%F %T')] [START] gpus=${GPUS_CSV} tasks=${#TASKS[@]} fractions=${FRACTIONS} center=${LOG_CENTER_DIAGNOSTICS} swd=${LOG_WASSERSTEIN_DIAGNOSTICS} interval=${CENTER_DIAGNOSTICS_INTERVAL}" | tee -a "${LOG_ROOT}/scheduler.log"

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
      run_task "${gpu}" "${task}" &
      pid="$!"
      gpu_pid["${gpu}"]="${pid}"
      gpu_task["${gpu}"]="${task}"
      all_idle=false
    fi
  done

  if [[ "${all_idle}" == "true" && "${task_index}" -ge "${#TASKS[@]}" ]]; then
    echo "[$(date '+%F %T')] [COMPLETE] preprocessing-cross reconst tasks finished" | tee -a "${LOG_ROOT}/scheduler.log"
    break
  fi
  sleep 60
done
