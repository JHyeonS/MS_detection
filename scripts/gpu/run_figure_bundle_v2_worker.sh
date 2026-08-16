#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?mode required}"
GPU="${2:?gpu required}"
TASK_NAME="${3:?task name required}"
SEEDS="${4:?seeds required}"
LOG_ROOT="${5:?log root required}"
RUN_ROOT="${6:?run root required}"
SITE="${7:-}"
METHODS="${8:-}"
RUN_PRETRAIN="${9:-true}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/ted1204/.conda/envs/ms_detection/bin/python" ]]; then
    export PYTHON_BIN="/home/ted1204/.conda/envs/ms_detection/bin/python"
  fi
fi

run_site_sweep() {
  local site="$1"
  local gpu="$2"
  local task_name="$3"
  local methods="$4"
  local seeds="$5"
  local run_pretrain="${6:-true}"
  IFS=',' read -r -a SEED_ARRAY <<< "${seeds}"
  for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | xargs)"
    local seed_root="${RUN_ROOT}/${task_name}/seed_${seed}"
    echo "[V2] site task=${task_name} seed=${seed} gpu=${gpu} run_root=${seed_root}"
    SITE_STUDY_METHODS="${methods}" \
    SITE_STUDY_FRACTIONS="0.05,0.10,0.25,0.50,1.00" \
    SITE_STUDY_RUN_ANALYZE="false" \
    SITE_STUDY_RUN_TSNE="false" \
    SITE_STUDY_RUN_PRETRAIN="${run_pretrain}" \
    SITE_STUDY_SEED="${seed}" \
    SITE_STUDY_FRACTION_SEED="${seed}" \
    bash scripts/gpu/site_main_study.sh "${site}" "${gpu}" "${LOG_ROOT}" "${seed_root}"
  done
}

run_norm_sweep() {
  local site="$1"
  local gpu="$2"
  local task_name="$3"
  local seeds="$4"
  IFS=',' read -r -a SEED_ARRAY <<< "${seeds}"
  for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | xargs)"
    local seed_root="${RUN_ROOT}/${task_name}/seed_${seed}"
    echo "[V2] norm task=${task_name} seed=${seed} gpu=${gpu} run_root=${seed_root}"
    NORM_ABLATION_METHODS="reconst" \
    NORM_ABLATION_FRACTIONS="0.05,0.10,0.25,0.50,1.00" \
    NORM_ABLATION_NORMALIZATIONS="none,robust" \
    NORM_ABLATION_PREPROCESS="bandpass_agc" \
    NORM_ABLATION_RUN_ANALYZE="false" \
    NORM_ABLATION_RUN_TSNE="false" \
    NORM_ABLATION_RUN_PRETRAIN="true" \
    SITE_STUDY_SEED="${seed}" \
    SITE_STUDY_FRACTION_SEED="${seed}" \
    bash scripts/gpu/preprocess_normalization_ablation.sh "${site}" "${gpu}" "${LOG_ROOT}" "${seed_root}"
  done
}

run_pair_sweep() {
  local gpu="$1"
  local task_name="$2"
  local seeds="$3"
  IFS=',' read -r -a SEED_ARRAY <<< "${seeds}"
  for seed in "${SEED_ARRAY[@]}"; do
    seed="$(echo "${seed}" | xargs)"
    local seed_root="${RUN_ROOT}/${task_name}/seed_${seed}"
    echo "[V2] pair task=${task_name} seed=${seed} gpu=${gpu} run_root=${seed_root}"
    PAIR_METHODS="scratch,reconst" \
    PAIR_RUN_ANALYZE="false" \
    PAIR_RUN_TSNE="false" \
    PAIR_RUN_PRETRAIN="true" \
    PAIR_STUDY_SEED="${seed}" \
    PAIR_STUDY_FRACTION_SEED="${seed}" \
    bash scripts/gpu/pair_cross_and_mixed_study.sh "${gpu}" "${LOG_ROOT}" "${seed_root}"
  done
}

case "${MODE}" in
  site)
    run_site_sweep "${SITE}" "${GPU}" "${TASK_NAME}" "${METHODS}" "${SEEDS}" "${RUN_PRETRAIN}"
    ;;
  norm)
    run_norm_sweep "${SITE}" "${GPU}" "${TASK_NAME}" "${SEEDS}"
    ;;
  pair)
    run_pair_sweep "${GPU}" "${TASK_NAME}" "${SEEDS}"
    ;;
  *)
    echo "[ERROR] unsupported mode: ${MODE}"
    exit 1
    ;;
esac
