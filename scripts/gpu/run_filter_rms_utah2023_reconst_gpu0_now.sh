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

GPU="${GPU:-0}"
DATASET="visualbest_filter_rms_fs1000_rms0p15_lp50"
SITE="utah_2023"
RUN_ROOT_BASE="runs/metadata_v2_safe_rerun_v1/filter_rms_site_main_pre50_v2/utah_2023_reconst_reconst_noanom"
LOG_ROOT="${LOG_ROOT:-logs/metadata_v2_missing_gpu0_reconst_now_v1}"
SPLIT_DIR="data/${DATASET}/metadata/experiments/stage1_utah_2023_only"
FRACTIONS="0.05,0.10,0.25,0.50,1.00"

mkdir -p "${RUN_ROOT_BASE}" "${LOG_ROOT}" "${MPLCONFIGDIR}"

slugify() {
  printf '%s' "$1" | tr ',|' '__' | sed 's#[^A-Za-z0-9._-]#_#g'
}

reconst_slug="$(slugify "filter_rms|utah_2023|reconst|${FRACTIONS}")"
noanom_slug="$(slugify "filter_rms|utah_2023|reconst_noanom|${FRACTIONS}")"
reconst_lock="${RUN_ROOT_BASE}/.metadata_v2_missing_lock_${reconst_slug}"
noanom_lock="${RUN_ROOT_BASE}/.metadata_v2_missing_lock_${noanom_slug}"

cleanup() {
  rmdir "${reconst_lock}" 2>/dev/null || true
  rmdir "${noanom_lock}" 2>/dev/null || true
}
trap cleanup EXIT

if ! mkdir "${reconst_lock}" 2>/dev/null; then
  echo "[SKIP_LOCKED] reconst lock exists: ${reconst_lock}" | tee -a "${LOG_ROOT}/scheduler.log"
  exit 0
fi
if ! mkdir "${noanom_lock}" 2>/dev/null; then
  echo "[SKIP_LOCKED] reconst_noanom lock exists: ${noanom_lock}" | tee -a "${LOG_ROOT}/scheduler.log"
  exit 0
fi

echo "[$(date '+%F %T')] [LAUNCH] gpu=${GPU} filter_rms utah_2023 methods=reconst,reconst_noanom fractions=${FRACTIONS}" | tee -a "${LOG_ROOT}/scheduler.log"

SITE_STUDY_SPLIT_DIR="${SPLIT_DIR}" \
SITE_STUDY_PREPROCESS="load_only" \
SITE_STUDY_NORMALIZE="none" \
SITE_STUDY_METHODS="reconst,reconst_noanom" \
SITE_STUDY_FRACTIONS="${FRACTIONS}" \
SITE_STUDY_RUN_ANALYZE="true" \
SITE_STUDY_RUN_TSNE="false" \
SITE_STUDY_RUN_PRETRAIN="true" \
SITE_STUDY_LOG_CENTER_DIAGNOSTICS="false" \
SITE_STUDY_LOG_WASSERSTEIN_DIAGNOSTICS="false" \
SITE_STUDY_NUM_WORKERS="1" \
SITE_STUDY_CACHE_MODE="ram" \
SITE_STUDY_PRETRAIN_EPOCHS="50" \
SITE_STUDY_PREFETCH_FACTOR="2" \
SITE_STUDY_PERSISTENT_WORKERS="false" \
SITE_STUDY_LOG_SLUG="metadata_v2_missing_filter_rms_utah_2023_reconst_gpu0_now" \
  bash scripts/gpu/site_main_study.sh "${SITE}" "${GPU}" "${LOG_ROOT}" "${RUN_ROOT_BASE}"

echo "[$(date '+%F %T')] [DONE] gpu=${GPU} filter_rms utah_2023 methods=reconst,reconst_noanom" | tee -a "${LOG_ROOT}/scheduler.log"
