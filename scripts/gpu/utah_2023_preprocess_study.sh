#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LOG_ROOT="${2:-logs}"
RUN_ROOT_BASE="${3:-runs/utah_2023_preprocess_study}"

VARIANTS="${UTAH_PREPROCESS_VARIANTS:-bandpass_agc,agc,bandpass}"
METHODS="${UTAH_PREPROCESS_METHODS:-reconst}"
FRACTIONS="${UTAH_PREPROCESS_FRACTIONS:-0.25,0.50,1.00}"
NORMALIZE="${UTAH_PREPROCESS_NORMALIZE:-robust}"
RUN_ANALYZE="${UTAH_PREPROCESS_RUN_ANALYZE:-true}"
RUN_TSNE="${UTAH_PREPROCESS_RUN_TSNE:-true}"
AGC_WINDOW_SEC="${UTAH_PREPROCESS_AGC_WINDOW_SEC:-0.2}"
AGC_CLIP="${UTAH_PREPROCESS_AGC_CLIP:-10.0}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT_BASE}"

IFS=',' read -r -a VARIANT_ARRAY <<< "${VARIANTS}"

for variant in "${VARIANT_ARRAY[@]}"; do
  variant="$(echo "${variant}" | xargs)"
  variant_root="${RUN_ROOT_BASE}/${variant}"

  echo "============================================================"
  echo "[UTAH 2023 PREPROCESS STUDY] $(date '+%F %T') variant=${variant} gpu=${GPU}"
  echo "methods=${METHODS}"
  echo "fractions=${FRACTIONS}"
  echo "normalize=${NORMALIZE}"
  echo "run_root=${variant_root}"
  echo "============================================================"

  SITE_STUDY_PREPROCESS="${variant}" \
  SITE_STUDY_NORMALIZE="${NORMALIZE}" \
  SITE_STUDY_AGC_WINDOW_SEC="${AGC_WINDOW_SEC}" \
  SITE_STUDY_AGC_CLIP="${AGC_CLIP}" \
  SITE_STUDY_METHODS="${METHODS}" \
  SITE_STUDY_FRACTIONS="${FRACTIONS}" \
  SITE_STUDY_RUN_PRETRAIN=true \
  SITE_STUDY_RUN_ANALYZE="${RUN_ANALYZE}" \
  SITE_STUDY_RUN_TSNE="${RUN_TSNE}" \
  bash scripts/gpu/site_main_study.sh utah_2023 "${GPU}" "${LOG_ROOT}" "${variant_root}"
done

echo "[DONE] Utah 2023 preprocess study completed"
echo "[INFO] root: ${RUN_ROOT_BASE}"
