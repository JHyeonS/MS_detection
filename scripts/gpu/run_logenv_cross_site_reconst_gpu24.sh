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
  if [[ -x "/home/anaconda3/bin/python3.9" ]]; then
    PYTHON_BIN="/home/anaconda3/bin/python3.9"
  elif [[ -x "/home/anaconda3/bin/python" ]]; then
    PYTHON_BIN="/home/anaconda3/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

GPUS_CSV="${CROSS_RECONST_GPUS:-${LOGENV_CROSS_GPUS:-2,4}}"
DATASET="${CROSS_RECONST_DATASET:-visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5}"
SOURCE_PRETRAIN_ROOT="${CROSS_RECONST_SOURCE_PRETRAIN_ROOT:-runs/visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5_site_main_pre50_v1}"
RUN_ROOT_BASE="${CROSS_RECONST_RUN_ROOT:-${LOGENV_CROSS_RUN_ROOT:-runs/logenv_cross_site_reconst_pre50_v1}}"
LOG_ROOT="${CROSS_RECONST_LOG_ROOT:-${LOGENV_CROSS_LOG_ROOT:-logs/logenv_cross_site_reconst_gpu24_v1}}"
FRACTIONS="${CROSS_RECONST_FRACTIONS:-${LOGENV_CROSS_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}}"
DATA_CACHE_MODE="${CROSS_RECONST_CACHE_MODE:-ram}"
DATA_NUM_WORKERS="${CROSS_RECONST_NUM_WORKERS:-1}"
DATA_PERSISTENT_WORKERS="${CROSS_RECONST_PERSISTENT_WORKERS:-false}"
DATA_PREFETCH_FACTOR="${CROSS_RECONST_PREFETCH_FACTOR:-2}"
LOG_CENTER_DIAGNOSTICS="${CROSS_RECONST_LOG_CENTER_DIAGNOSTICS:-false}"
LOG_WASSERSTEIN_DIAGNOSTICS="${CROSS_RECONST_LOG_WASSERSTEIN_DIAGNOSTICS:-false}"
CENTER_DIAGNOSTICS_INTERVAL="${CROSS_RECONST_CENTER_DIAGNOSTICS_INTERVAL:-1}"
WASSERSTEIN_NUM_PROJECTIONS="${CROSS_RECONST_WASSERSTEIN_NUM_PROJECTIONS:-32}"
WASSERSTEIN_NUM_QUANTILES="${CROSS_RECONST_WASSERSTEIN_NUM_QUANTILES:-128}"

mkdir -p "${RUN_ROOT_BASE}" "${LOG_ROOT}"
LOCK_ROOT="${RUN_ROOT_BASE}/.locks"
mkdir -p "${LOCK_ROOT}"

OLD_IFS="${IFS}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
IFS="${OLD_IFS}"

TASKS=(
  "pohang|utah_2019"
  "utah_2019|pohang"
  "pohang|utah_2023"
  "utah_2023|pohang"
  "utah_2019|utah_2023"
  "utah_2023|utah_2019"
)

if [[ -n "${CROSS_RECONST_TASKS:-}" ]]; then
  OLD_IFS="${IFS}"
  IFS=',' read -r -a TASKS <<< "${CROSS_RECONST_TASKS}"
  IFS="${OLD_IFS}"
fi

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

source_encoder() {
  local site="$1"
  local preferred=""
  local fallback=""
  case "${site}" in
    pohang)
      preferred="${SOURCE_PRETRAIN_ROOT}/pohang_reconst_reconst_noanom/reconst/pretrain/pohang/best_encoder.pt"
      fallback="${SOURCE_PRETRAIN_ROOT}/pohang/reconst/pretrain/pohang/best_encoder.pt"
      ;;
    utah_2019)
      preferred="${SOURCE_PRETRAIN_ROOT}/utah_2019_reconst_reconst_noanom/reconst/pretrain/base_utah_2019/best_encoder.pt"
      fallback="${SOURCE_PRETRAIN_ROOT}/utah_2019/reconst/pretrain/base_utah_2019/best_encoder.pt"
      ;;
    utah_2023)
      preferred="${SOURCE_PRETRAIN_ROOT}/utah_2023_reconst_reconst_noanom/reconst/pretrain/base_utah_2023/best_encoder.pt"
      fallback="${SOURCE_PRETRAIN_ROOT}/utah_2023/reconst/pretrain/base_utah_2023/best_encoder.pt"
      ;;
    *) echo "[ERROR] unsupported source site: ${site}" >&2; return 1 ;;
  esac
  if [[ -f "${preferred}" ]]; then
    echo "${preferred}"
  else
    echo "${fallback}"
  fi
}

frac_tag() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
x = float(sys.argv[1])
print(("{:.6f}".format(x)).rstrip("0").rstrip(".").replace(".", "p"))
PY
}

all_tests_complete() {
  local source="$1"
  local target="$2"
  local target_exp fraction tag test_path old_ifs
  local -a fraction_array
  target_exp="$(site_experiment "${target}")"
  old_ifs="${IFS}"
  IFS=',' read -r -a fraction_array <<< "${FRACTIONS}"
  IFS="${old_ifs}"
  for fraction in "${fraction_array[@]}"; do
    fraction="$(echo "${fraction}" | xargs)"
    tag="$(frac_tag "${fraction}")"
    test_path="${RUN_ROOT_BASE}/${source}_to_${target}/reconst/test/${target_exp}__frac${tag}/test_metrics_fixed_threshold.json"
    if [[ ! -f "${test_path}" ]]; then
      return 1
    fi
  done
  return 0
}

make_base_cfg() {
  local target="$1"
  local run_root="$2"
  local out_cfg="$3"
  local template split_name split_dir
  template="$(site_base_template "${target}")"
  split_name="$(site_split_name "${target}")"
  split_dir="data/${DATASET}/metadata/experiments/${split_name}"
  "${PYTHON_BIN}" - "${template}" "${run_root}" "${split_dir}" "${out_cfg}" <<'PY'
import sys
import os
import yaml

src, run_root, split_dir, outp = sys.argv[1:5]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
data = cfg.setdefault("data", {})
data["split_dir"] = split_dir
data["num_workers"] = int(os.environ.get("CROSS_RECONST_NUM_WORKERS", "1"))
data["cache_mode"] = os.environ.get("CROSS_RECONST_CACHE_MODE", "ram")
data["persistent_workers"] = os.environ.get("CROSS_RECONST_PERSISTENT_WORKERS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
data["prefetch_factor"] = int(os.environ.get("CROSS_RECONST_PREFETCH_FACTOR", "2"))
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
  local target="$1"
  local run_root="$2"
  local fraction="$3"
  local encoder="$4"
  local out_cfg="$5"
  local template anomaly_weight
  template="$(site_finetune_template "${target}")"
  anomaly_weight="$(site_anomaly_weight "${target}")"
  "${PYTHON_BIN}" - "${template}" "${run_root}" "${fraction}" "${encoder}" "${anomaly_weight}" "${out_cfg}" <<'PY'
import sys
import os
import yaml

src, run_root, fraction, encoder, anomaly_weight, outp = sys.argv[1:7]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
data = cfg.setdefault("data", {})
data["num_workers"] = int(os.environ.get("CROSS_RECONST_NUM_WORKERS", "1"))
data["cache_mode"] = os.environ.get("CROSS_RECONST_CACHE_MODE", "ram")
data["persistent_workers"] = os.environ.get("CROSS_RECONST_PERSISTENT_WORKERS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
data["prefetch_factor"] = int(os.environ.get("CROSS_RECONST_PREFETCH_FACTOR", "2"))
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
train["log_center_diagnostics"] = os.environ.get("CROSS_RECONST_LOG_CENTER_DIAGNOSTICS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
train["log_wasserstein_diagnostics"] = os.environ.get("CROSS_RECONST_LOG_WASSERSTEIN_DIAGNOSTICS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
train["center_diagnostics_interval"] = int(os.environ.get("CROSS_RECONST_CENTER_DIAGNOSTICS_INTERVAL", "1"))
train["wasserstein_num_projections"] = int(os.environ.get("CROSS_RECONST_WASSERSTEIN_NUM_PROJECTIONS", "32"))
train["wasserstein_num_quantiles"] = int(os.environ.get("CROSS_RECONST_WASSERSTEIN_NUM_QUANTILES", "128"))
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
import sys
import os
import yaml
src, outp = sys.argv[1:3]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
data = cfg.setdefault("data", {})
data["num_workers"] = int(os.environ.get("CROSS_RECONST_NUM_WORKERS", "1"))
data["cache_mode"] = os.environ.get("CROSS_RECONST_CACHE_MODE", "ram")
data["persistent_workers"] = os.environ.get("CROSS_RECONST_PERSISTENT_WORKERS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
data["prefetch_factor"] = int(os.environ.get("CROSS_RECONST_PREFETCH_FACTOR", "2"))
data["normalize"] = "none"
pre = data.setdefault("preprocess", {})
pre["load_only"] = True
pre["detrend"] = False
pre["bandpass"] = False
pre["agc"] = False
data["preprocess_variant"] = "load_only"
cfg.setdefault("test", {})["num_workers"] = int(os.environ.get("CROSS_RECONST_NUM_WORKERS", "1"))
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

run_task() {
  local gpu="$1"
  local task="$2"
  local source target encoder target_exp run_root tmp_dir log_slug task_log base_cfg test_cfg old_ifs lock_dir
  local -a fraction_array
  old_ifs="${IFS}"
  IFS='|' read -r source target <<< "${task}"
  IFS="${old_ifs}"

  run_root="${RUN_ROOT_BASE}/${source}_to_${target}/reconst"
  tmp_dir=".tmp_logenv_cross_site_reconst_${source}_to_${target}"
  log_slug="${source}_to_${target}"
  task_log="${LOG_ROOT}/${log_slug}.log"
  base_cfg="${tmp_dir}/base.yaml"
  test_cfg="${tmp_dir}/test.yaml"
  target_exp="$(site_experiment "${target}")"
  encoder="$(source_encoder "${source}")"
  mkdir -p "${run_root}" "${tmp_dir}"

  if [[ ! -f "${encoder}" ]]; then
    echo "[ERROR] missing encoder for ${source}: ${encoder}" | tee -a "${task_log}"
    return 1
  fi

  if all_tests_complete "${source}" "${target}"; then
    echo "[$(date '+%F %T')] [SKIP_COMPLETE] ${source}->${target}" | tee -a "${LOG_ROOT}/scheduler.log"
    return 0
  fi

  lock_dir="${LOCK_ROOT}/${source}_to_${target}.lock"
  if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "[$(date '+%F %T')] [SKIP_LOCKED] ${source}->${target} lock=${lock_dir}" | tee -a "${LOG_ROOT}/scheduler.log" | tee -a "${task_log}"
    return 0
  fi
  trap 'rmdir "'"${lock_dir}"'" 2>/dev/null || true' EXIT

  echo "[$(date '+%F %T')] [TASK_START] gpu=${gpu} source=${source} target=${target} encoder=${encoder}" | tee -a "${LOG_ROOT}/scheduler.log" | tee -a "${task_log}"
  make_base_cfg "${target}" "${run_root}" "${base_cfg}"
  make_test_cfg "${test_cfg}"

  old_ifs="${IFS}"
  IFS=',' read -r -a fraction_array <<< "${FRACTIONS}"
  IFS="${old_ifs}"

  for fraction in "${fraction_array[@]}"; do
    fraction="$(echo "${fraction}" | xargs)"
    tag="frac$(frac_tag "${fraction}")"
    local ft_dir="${run_root}/finetune/${target_exp}__${tag}"
    local test_json="${run_root}/test/${target_exp}__${tag}/test_metrics_fixed_threshold.json"
    local ft_cfg="${tmp_dir}/finetune_${tag}.yaml"
    local eval_cfg="${tmp_dir}/eval_${tag}.yaml"
    local run_log="${LOG_ROOT}/${log_slug}__${tag}.log"

    if [[ -f "${test_json}" ]]; then
      echo "[SKIP] existing test ${source}->${target} ${tag}" | tee -a "${task_log}" | tee -a "${run_log}"
      continue
    fi

    make_finetune_cfg "${target}" "${run_root}" "${fraction}" "${encoder}" "${ft_cfg}"
    make_eval_base_cfg "${base_cfg}" "${tag}" "${eval_cfg}"

    {
      echo "============================================================"
      echo "[LOGENV CROSS RECONST] $(date '+%F %T') source=${source} target=${target} fraction=${fraction} gpu=${gpu}"
      echo "run_root=${run_root}"
      echo "encoder=${encoder}"
      echo "base_cfg=${base_cfg}"
      echo "finetune_cfg=${ft_cfg}"
      echo "eval_cfg=${eval_cfg}"
      echo "finetune_dir=${ft_dir}"
      echo "============================================================"
    } | tee -a "${task_log}" | tee -a "${run_log}"

    echo "[FINETUNE] ${source}->${target} ${tag}" | tee -a "${task_log}" | tee -a "${run_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
      --base_cfg "${base_cfg}" \
      --stage_cfg "${ft_cfg}" \
      --exp_suffix "${tag}" >> "${run_log}" 2>&1

    echo "[TEST] ${source}->${target} ${tag}" | tee -a "${task_log}" | tee -a "${run_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
      --base_cfg "${eval_cfg}" \
      --stage_cfg "${test_cfg}" >> "${run_log}" 2>&1

    echo "[DONE] ${source}->${target} ${tag}" | tee -a "${task_log}" | tee -a "${run_log}"
  done

  "${PYTHON_BIN}" scripts/gpu/summarize_pohang_main_study.py \
    --root "${run_root}" \
    --out "${RUN_ROOT_BASE}/${source}_to_${target}/summary.csv" >> "${task_log}" 2>&1
  echo "[$(date '+%F %T')] [TASK_DONE] gpu=${gpu} source=${source} target=${target}" | tee -a "${LOG_ROOT}/scheduler.log" | tee -a "${task_log}"
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
    echo "[$(date '+%F %T')] [COMPLETE] logenv cross-site reconst tasks finished" | tee -a "${LOG_ROOT}/scheduler.log"
    break
  fi
  sleep 60
done
