#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-ms_detection}}"
export PYTHON_BIN="${PYTHON_BIN:-/home/ted1204/.conda/envs/hsenv/bin/python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

GPUS="${WRITING_TRANSFER_GPUS:-0,1,2,3}"
FRACTIONS="${WRITING_TRANSFER_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
RUN_ROOT_PREFIX="${METADATA_V2_RUN_ROOT_PREFIX:-runs/metadata_v2_safe_rerun_v1}"
LOG_ROOT="${WRITING_TRANSFER_LOG_ROOT:-logs/writing_remaining_transfer_v1}"

mkdir -p "${LOG_ROOT}" "${MPLCONFIGDIR}"

echo "[$(date '+%F %T')] [START] writing remaining transfer experiments" | tee -a "${LOG_ROOT}/scheduler.log"
echo "[$(date '+%F %T')] [CONFIG] gpus=${GPUS} fractions=${FRACTIONS}" | tee -a "${LOG_ROOT}/scheduler.log"

echo "[$(date '+%F %T')] [PHASE] raw site transfer" | tee -a "${LOG_ROOT}/scheduler.log"
CROSS_RECONST_GPUS="${GPUS}" \
CROSS_RECONST_DATASET="visualbest_raw_rms_fs1000_rms0p15_nofilter" \
CROSS_RECONST_SOURCE_PRETRAIN_ROOT="${RUN_ROOT_PREFIX}/raw_site_main_pre50_v1" \
CROSS_RECONST_RUN_ROOT="${RUN_ROOT_PREFIX}/raw_cross_site_reconst_pre50_v1" \
CROSS_RECONST_LOG_ROOT="${LOG_ROOT}/raw_cross_site" \
CROSS_RECONST_FRACTIONS="${FRACTIONS}" \
CROSS_RECONST_CACHE_MODE="ram" \
CROSS_RECONST_NUM_WORKERS="1" \
CROSS_RECONST_PERSISTENT_WORKERS="false" \
CROSS_RECONST_PREFETCH_FACTOR="2" \
CROSS_RECONST_LOG_CENTER_DIAGNOSTICS="false" \
CROSS_RECONST_LOG_WASSERSTEIN_DIAGNOSTICS="false" \
  bash scripts/gpu/run_logenv_cross_site_reconst_gpu24.sh

echo "[$(date '+%F %T')] [PHASE] preprocessing transfer 3P2" | tee -a "${LOG_ROOT}/scheduler.log"
PREPROC_CROSS_GPUS="${GPUS}" \
PREPROC_CROSS_FRACTIONS="${FRACTIONS}" \
PREPROC_CROSS_RUN_ROOT="${RUN_ROOT_PREFIX}/preprocessing_cross_reconst_pre50_v1" \
PREPROC_CROSS_LOG_ROOT="${LOG_ROOT}/preprocessing_cross_3p2" \
PREPROC_CROSS_CACHE_MODE="ram" \
PREPROC_CROSS_NUM_WORKERS="1" \
PREPROC_CROSS_PERSISTENT_WORKERS="false" \
PREPROC_CROSS_PREFETCH_FACTOR="2" \
PREPROC_CROSS_LOG_CENTER_DIAGNOSTICS="false" \
PREPROC_CROSS_LOG_WASSERSTEIN_DIAGNOSTICS="false" \
PREPROC_CROSS_TASKS="pohang|raw|filter_rms,pohang|raw|logenv,pohang|filter_rms|raw,pohang|filter_rms|logenv,pohang|logenv|raw,pohang|logenv|filter_rms,utah_2019|raw|filter_rms,utah_2019|raw|logenv,utah_2019|filter_rms|raw,utah_2019|filter_rms|logenv,utah_2019|logenv|raw,utah_2019|logenv|filter_rms,utah_2023|raw|filter_rms,utah_2023|raw|logenv,utah_2023|filter_rms|raw,utah_2023|filter_rms|logenv,utah_2023|logenv|raw,utah_2023|logenv|filter_rms" \
  bash scripts/gpu/run_metadata_v2_preprocessing_cross_reconst.sh

echo "[$(date '+%F %T')] [COMPLETE] writing remaining transfer experiments" | tee -a "${LOG_ROOT}/scheduler.log"
