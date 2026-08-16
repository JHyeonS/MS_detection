#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="$(pwd)"
MODE="${1:-full}"

if [[ "${MODE}" == "final" || "${MODE}" == "final_test" || "${MODE}" == "final_full" ]]; then
  FINAL_SITE="${2:-pohang}"
  GPU_LIST="${3:-0}"
  LOG_ROOT="${4:-${WORK_DIR}/logs}"
  HPO_SPEC="${5:-configs/experiments/hpo/finetune_stage1_pohang_best_arch_refined.yaml}"
  ARCH_HPO_SPEC="${5:-configs/experiments/hpo/architecture_stage1_pohang.yaml}"
else
  FINAL_SITE="${FINAL_SITE:-pohang}"
  GPU_LIST="${2:-0}"
  LOG_ROOT="${3:-${WORK_DIR}/logs}"
  HPO_SPEC="${4:-configs/experiments/hpo/finetune_stage1_pohang_best_arch_refined.yaml}"
  ARCH_HPO_SPEC="${4:-configs/experiments/hpo/architecture_stage1_pohang.yaml}"
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
mkdir -p "${LOG_ROOT}"
mkdir -p .tmp_stage_label_eff_pretrained_dynamic_center_test_cfg
mkdir -p .tmp_stage_label_eff_pretrained_dynamic_center_analyze_cfg
mkdir -p .tmp_final_pipeline_cfg

export PYTHONPATH=.
export LD_LIBRARY_PATH=/home/anaconda3/lib:${LD_LIBRARY_PATH:-}
export MS_JOB_OWNER="${MS_JOB_OWNER:-${USER:-unknown}}"

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
    echo "[ERROR] could not resolve PYTHON_BIN"
    exit 1
  fi
fi

echo "[INFO] resolved PYTHON_BIN=${PYTHON_BIN}"

BASE_CONFIGS=(
  "configs/train/base_pohang.yaml"\
  "configs/train/base_utah_2019.yaml"\
  "configs/train/base_utah_2023.yaml"\
)

TRAIN_CONFIGS=(
  "configs/train/train.yaml"\
)

PRETRAIN_STAGE_CFG="configs/train/pretrain_reconst.yaml"
TEST_STAGE_CFG="configs/train/test.yaml"
ANALYZE_STAGE_CFG="configs/train/analyze.yaml"
FINAL_BASE_CFG="configs/train/base_pohang_arch_best.yaml"
FINAL_TRAIN_CFG="configs/train/final_pohang_best.yaml"
FINAL_RUN_ROOT="./runs/final/pohang_best"
FINAL_EXPERIMENT="pohang"

if [[ "${MODE}" != "pretrain" && "${MODE}" != "finetune" && "${MODE}" != "test" && "${MODE}" != "analyze" && "${MODE}" != "hpo" && "${MODE}" != "hpo_arch" && "${MODE}" != "full" && "${MODE}" != "final" && "${MODE}" != "final_test" && "${MODE}" != "final_full" ]]; then
  echo "[ERROR] invalid MODE: ${MODE}"
  exit 1
fi

make_suffix_base_cfg() {
  local src_base_cfg="$1"
  local exp_suffix="$2"
  local out_cfg="$3"
  "${PYTHON_BIN}" - "${src_base_cfg}" "${exp_suffix}" "${out_cfg}" <<'PY'
import sys, yaml
src, suffix, outp = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
base_exp = cfg["data"]["experiment"]
cfg["data"]["experiment"] = f"{base_exp}__{suffix}"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_run_root_base_cfg() {
  local src_base_cfg="$1"
  local run_root="$2"
  local out_cfg="$3"
  "${PYTHON_BIN}" - "${src_base_cfg}" "${run_root}" "${out_cfg}" <<'PY'
import sys, yaml
src, run_root, outp = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

cleanup() {
  echo
  echo "[WARN] interrupt received, killing child workers..."
  jobs -pr | xargs -r kill || true
  wait || true
  pkill -f "src/detection/training/trainer_pretrain.py" || true
  pkill -f "src/detection/training/trainer_finetune.py" || true
  pkill -f "src/detection/training/trainer_test.py" || true
  pkill -f "src/detection/analysis/analyze.py" || true
  exit 130
}
trap cleanup INT TERM

run_pretrain_job() {
  local gpu="$1"
  local base_cfg="$2"
  local base_name log_file
  base_name="$(basename "${base_cfg}" .yaml)"
  log_file="${LOG_ROOT}/pretrain__${base_name}.log"
  {
    echo "============================================================"
    echo "[PRETRAIN START] $(date '+%F %T') gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "base_cfg=${base_cfg}"
    echo "stage_cfg=${PRETRAIN_STAGE_CFG}"
    echo "============================================================"
  } | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_pretrain \
    --base_cfg "${base_cfg}" \
    --stage_cfg "${PRETRAIN_STAGE_CFG}" >> "${log_file}" 2>&1
}

launch_pretrain_queue() {
  local pids=()
  local idx=0
  for base_cfg in "${BASE_CONFIGS[@]}"; do
    local gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
    run_pretrain_job "${gpu}" "${base_cfg}" &
    pids+=($!)
    idx=$((idx + 1))
  done
  local exit_code=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
  done
  return "${exit_code}"
}

run_finetune_job() {
  local gpu="$1"
  local base_cfg="$2"
  local train_cfg="$3"
  local base_name train_name exp_suffix log_file
  base_name="$(basename "${base_cfg}" .yaml)"
  train_name="$(basename "${train_cfg}" .yaml)"
  exp_suffix="${train_name}"
  log_file="${LOG_ROOT}/finetune__${base_name}__${train_name}.log"
  {
    echo "============================================================"
    echo "[FINETUNE START] $(date '+%F %T') gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "base_cfg=${base_cfg}"
    echo "train_cfg=${train_cfg}"
    echo "exp_suffix=${exp_suffix}"
    echo "============================================================"
  } | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
    --base_cfg "${base_cfg}" \
    --stage_cfg "${train_cfg}" \
    --exp_suffix "${exp_suffix}" >> "${log_file}" 2>&1
}

launch_finetune_queue() {
  local pids=()
  local idx=0
  for base_cfg in "${BASE_CONFIGS[@]}"; do
    for train_cfg in "${TRAIN_CONFIGS[@]}"; do
      local gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
      run_finetune_job "${gpu}" "${base_cfg}" "${train_cfg}" &
      pids+=($!)
      idx=$((idx + 1))
    done
  done
  local exit_code=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
  done
  return "${exit_code}"
}

run_test_job() {
  local gpu="$1"
  local base_cfg="$2"
  local train_cfg="$3"
  local base_name train_name exp_suffix log_file test_base_cfg
  base_name="$(basename "${base_cfg}" .yaml)"
  train_name="$(basename "${train_cfg}" .yaml)"
  exp_suffix="${train_name}"
  test_base_cfg=".tmp_stage_label_eff_pretrained_dynamic_center_test_cfg/${base_name}__${train_name}.yaml"
  log_file="${LOG_ROOT}/test__${base_name}__${train_name}.log"
  {
    echo "============================================================"
    echo "[TEST START] $(date '+%F %T') gpu=${gpu}"
    echo "CUDA_VISIBLE_DEVICES=${gpu}"
    echo "base_cfg=${base_cfg}"
    echo "train_cfg=${train_cfg}"
    echo "exp_suffix=${exp_suffix}"
    echo "============================================================"
  } | tee -a "${log_file}"
  make_suffix_base_cfg "${base_cfg}" "${exp_suffix}" "${test_base_cfg}" >> "${log_file}" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
    --base_cfg "${test_base_cfg}" \
    --stage_cfg "${TEST_STAGE_CFG}" >> "${log_file}" 2>&1
}

launch_test_queue() {
  local pids=()
  local idx=0
  for base_cfg in "${BASE_CONFIGS[@]}"; do
    for train_cfg in "${TRAIN_CONFIGS[@]}"; do
      local gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
      run_test_job "${gpu}" "${base_cfg}" "${train_cfg}" &
      pids+=($!)
      idx=$((idx + 1))
    done
  done
  local exit_code=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
  done
  return "${exit_code}"
}

run_analyze_all() {
  for base_cfg in "${BASE_CONFIGS[@]}"; do
    for train_cfg in "${TRAIN_CONFIGS[@]}"; do
      local base_name train_name exp_suffix analyze_base_cfg log_file
      base_name="$(basename "${base_cfg}" .yaml)"
      train_name="$(basename "${train_cfg}" .yaml)"
      exp_suffix="${train_name}"
      analyze_base_cfg=".tmp_stage_label_eff_pretrained_dynamic_center_analyze_cfg/${base_name}__${train_name}.yaml"
      log_file="${LOG_ROOT}/analyze__${base_name}__${train_name}.log"
      {
        echo "============================================================"
        echo "[ANALYZE START] $(date '+%F %T')"
        echo "base_cfg=${base_cfg}"
        echo "train_cfg=${train_cfg}"
        echo "exp_suffix=${exp_suffix}"
        echo "============================================================"
      } | tee -a "${log_file}"
      make_suffix_base_cfg "${base_cfg}" "${exp_suffix}" "${analyze_base_cfg}" >> "${log_file}" 2>&1
      "${PYTHON_BIN}" -m src.detection.analysis.analyze \
        --base_cfg "${analyze_base_cfg}" \
        --stage_cfg "${ANALYZE_STAGE_CFG}" >> "${log_file}" 2>&1
    done
  done
}

run_hpo() {
  local log_file
  log_file="${LOG_ROOT}/hpo__$(basename "${HPO_SPEC}" .yaml).log"
  {
    echo "============================================================"
    echo "[HPO START] $(date '+%F %T')"
    echo "spec=${HPO_SPEC}"
    echo "python_bin=${PYTHON_BIN}"
    echo "job_owner=${MS_JOB_OWNER}"
    echo "CUDA_VISIBLE_DEVICES=${GPU_LIST}"
    echo "============================================================"
  } | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  "${PYTHON_BIN}" scripts/gpu/hpo_finetune.py \
    --spec "${HPO_SPEC}" >> "${log_file}" 2>&1
}

run_hpo_arch() {
  local log_file
  log_file="${LOG_ROOT}/hpo_arch__$(basename "${ARCH_HPO_SPEC}" .yaml).log"
  {
    echo "============================================================"
    echo "[HPO ARCH START] $(date '+%F %T')"
    echo "spec=${ARCH_HPO_SPEC}"
    echo "python_bin=${PYTHON_BIN}"
    echo "job_owner=${MS_JOB_OWNER}"
    echo "CUDA_VISIBLE_DEVICES=${GPU_LIST}"
    echo "============================================================"
  } | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  "${PYTHON_BIN}" scripts/gpu/hpo_architecture.py \
    --spec "${ARCH_HPO_SPEC}" >> "${log_file}" 2>&1
}

resolve_final_configs() {
  case "${FINAL_SITE}" in
    pohang)
      FINAL_BASE_CFG="configs/train/base_pohang_arch_best.yaml"
      FINAL_TRAIN_CFG="configs/train/final_pohang_best.yaml"
      FINAL_RUN_ROOT="./runs/final/pohang_best"
      FINAL_EXPERIMENT="pohang"
      ;;
    utah_2019)
      FINAL_BASE_CFG="configs/train/base_utah_2019_arch_best.yaml"
      FINAL_TRAIN_CFG="configs/train/final_utah_2019_best.yaml"
      FINAL_RUN_ROOT="./runs/final/utah_2019_best"
      FINAL_EXPERIMENT="base_utah_2019"
      ;;
    utah_2023)
      FINAL_BASE_CFG="configs/train/base_utah_2023_arch_best.yaml"
      FINAL_TRAIN_CFG="configs/train/final_utah_2023_best.yaml"
      FINAL_RUN_ROOT="./runs/final/utah_2023_best"
      FINAL_EXPERIMENT="base_utah_2023"
      ;;
    *)
      echo "[ERROR] unsupported FINAL_SITE: ${FINAL_SITE}"
      echo "[ERROR] expected one of: pohang, utah_2019, utah_2023"
      exit 1
      ;;
  esac
}

prepare_final_base_cfg() {
  resolve_final_configs
  FINAL_EFFECTIVE_BASE_CFG=".tmp_final_pipeline_cfg/${FINAL_SITE}_base.yaml"
  make_run_root_base_cfg "${FINAL_BASE_CFG}" "${FINAL_RUN_ROOT}" "${FINAL_EFFECTIVE_BASE_CFG}"
}

run_final_pretrain() {
  local gpu="$1"
  local log_file="$2"
  echo "[FINAL] pretrain" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_pretrain \
    --base_cfg "${FINAL_EFFECTIVE_BASE_CFG}" \
    --stage_cfg "${PRETRAIN_STAGE_CFG}" >> "${log_file}" 2>&1
}

run_final_finetune() {
  local gpu="$1"
  local log_file="$2"
  echo "[FINAL] finetune" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
    --base_cfg "${FINAL_EFFECTIVE_BASE_CFG}" \
    --stage_cfg "${FINAL_TRAIN_CFG}" >> "${log_file}" 2>&1
}

run_final_test() {
  local gpu="$1"
  local log_file="$2"
  echo "[FINAL] test" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
    --base_cfg "${FINAL_EFFECTIVE_BASE_CFG}" \
    --stage_cfg "${TEST_STAGE_CFG}" >> "${log_file}" 2>&1
}

run_final_analyze() {
  local log_file="$1"
  echo "[FINAL] analyze" | tee -a "${log_file}"
  "${PYTHON_BIN}" -m src.detection.analysis.analyze \
    --base_cfg "${FINAL_EFFECTIVE_BASE_CFG}" \
    --stage_cfg "${ANALYZE_STAGE_CFG}" >> "${log_file}" 2>&1
}

run_final_tsne() {
  local gpu="$1"
  local log_file="$2"
  local finetune_dir="${FINAL_RUN_ROOT}/finetune/${FINAL_EXPERIMENT}"
  echo "[FINAL] tsne" | tee -a "${log_file}"
  PYTHON_BIN="${PYTHON_BIN}" bash scripts/gpu/run_tsne.sh "${gpu}" "${finetune_dir}" >> "${log_file}" 2>&1
}

run_final_pipeline() {
  local gpu="$1"
  local run_pretrain="$2"
  local run_downstream="$3"
  local log_file
  prepare_final_base_cfg
  log_file="${LOG_ROOT}/final__${FINAL_SITE}.log"

  {
    echo "============================================================"
    echo "[FINAL START] $(date '+%F %T') site=${FINAL_SITE} gpu=${gpu}"
    echo "base_cfg=${FINAL_EFFECTIVE_BASE_CFG}"
    echo "source_base_cfg=${FINAL_BASE_CFG}"
    echo "stage_cfg=${FINAL_TRAIN_CFG}"
    echo "run_root=${FINAL_RUN_ROOT}"
    echo "experiment=${FINAL_EXPERIMENT}"
    echo "run_pretrain=${run_pretrain}"
    echo "run_downstream=${run_downstream}"
    echo "============================================================"
  } | tee -a "${log_file}"

  if [[ "${run_pretrain}" == "true" ]]; then
    run_final_pretrain "${gpu}" "${log_file}"
  fi

  if [[ "${run_downstream}" == "true" ]]; then
    run_final_finetune "${gpu}" "${log_file}"
  fi

  run_final_test "${gpu}" "${log_file}"
  run_final_analyze "${log_file}"
  run_final_tsne "${gpu}" "${log_file}"

  echo "[DONE] final pipeline completed. site=${FINAL_SITE}" | tee -a "${log_file}"
}

case "${MODE}" in
  pretrain) launch_pretrain_queue ;;
  finetune) launch_finetune_queue ;;
  test) launch_test_queue ;;
  analyze) run_analyze_all ;;
  hpo) run_hpo ;;
  hpo_arch) run_hpo_arch ;;
  final) run_final_pipeline "${GPUS[0]}" true true ;;
  final_test) run_final_pipeline "${GPUS[0]}" false false ;;
  final_full)
    echo "[INFO] MODE=final_full is deprecated; use MODE=final. Running full final pipeline."
    run_final_pipeline "${GPUS[0]}" true true
    ;;
  full)
    echo "[INFO] MODE=full : pretrain -> finetune -> test -> analyze"
    launch_pretrain_queue
    launch_finetune_queue
    launch_test_queue
    run_analyze_all
    ;;
esac

echo "[DONE] stage_label_efficiency_pretrained_dynamic_center completed. mode=${MODE}"
