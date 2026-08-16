#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LOG_ROOT="${2:-logs/utah_2023_gem_pooling_study}"
RUN_ROOT="${3:-runs/utah_2023_gem_pooling_study}"

export PYTHONPATH=.
export SITE_STUDY_POOLING="${SITE_STUDY_POOLING:-gem}"
export SITE_STUDY_POOLING_P="${SITE_STUDY_POOLING_P:-3.0}"
export SITE_STUDY_POOLING_CHANNELWISE="${SITE_STUDY_POOLING_CHANNELWISE:-true}"
export SITE_STUDY_METHODS="${SITE_STUDY_METHODS:-scratch,reconst}"
export SITE_STUDY_FRACTIONS="${SITE_STUDY_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
export SITE_STUDY_RUN_PRETRAIN="${SITE_STUDY_RUN_PRETRAIN:-true}"
export SITE_STUDY_RUN_ANALYZE="${SITE_STUDY_RUN_ANALYZE:-true}"
export SITE_STUDY_RUN_TSNE="${SITE_STUDY_RUN_TSNE:-false}"

mkdir -p "${LOG_ROOT}"

echo "[RUN] Utah 2023 GeM pooling study"
echo "gpu=${GPU}"
echo "run_root=${RUN_ROOT}"
echo "log_root=${LOG_ROOT}"
echo "methods=${SITE_STUDY_METHODS}"
echo "fractions=${SITE_STUDY_FRACTIONS}"
echo "pooling=${SITE_STUDY_POOLING}"
echo "pooling_p=${SITE_STUDY_POOLING_P}"

bash scripts/gpu/site_main_study.sh utah_2023 "${GPU}" "${LOG_ROOT}" "${RUN_ROOT}"
