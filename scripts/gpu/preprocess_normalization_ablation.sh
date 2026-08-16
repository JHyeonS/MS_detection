#!/usr/bin/env bash
set -euo pipefail

SITE="${1:-utah_2023}"
GPU="${2:-0}"
LOG_ROOT="${3:-logs}"
RUN_ROOT_BASE="${4:-runs/${SITE}_normalization_ablation}"

METHODS="${NORM_ABLATION_METHODS:-reconst}"
FRACTIONS="${NORM_ABLATION_FRACTIONS:-0.25,0.50,1.00}"
NORMALIZATIONS="${NORM_ABLATION_NORMALIZATIONS:-none,robust}"
PREPROCESS="${NORM_ABLATION_PREPROCESS:-bandpass_agc}"
AGC_WINDOW_SEC="${NORM_ABLATION_AGC_WINDOW_SEC:-0.2}"
AGC_CLIP="${NORM_ABLATION_AGC_CLIP:-10.0}"
RUN_ANALYZE="${NORM_ABLATION_RUN_ANALYZE:-true}"
RUN_TSNE="${NORM_ABLATION_RUN_TSNE:-true}"
RUN_PRETRAIN="${NORM_ABLATION_RUN_PRETRAIN:-true}"
POOLING="${NORM_ABLATION_POOLING:-${SITE_STUDY_POOLING:-avg}}"
POOLING_P="${NORM_ABLATION_POOLING_P:-${SITE_STUDY_POOLING_P:-3.0}}"
POOLING_CHANNELWISE="${NORM_ABLATION_POOLING_CHANNELWISE:-${SITE_STUDY_POOLING_CHANNELWISE:-true}}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT_BASE}"

IFS=',' read -r -a NORMALIZATION_ARRAY <<< "${NORMALIZATIONS}"

for normalize in "${NORMALIZATION_ARRAY[@]}"; do
  normalize="$(echo "${normalize}" | xargs)"
  variant_root="${RUN_ROOT_BASE}/${PREPROCESS}_${normalize}"

  echo "============================================================"
  echo "[NORMALIZATION ABLATION] $(date '+%F %T') site=${SITE} normalize=${normalize} gpu=${GPU}"
  echo "preprocess=${PREPROCESS}"
  echo "methods=${METHODS}"
  echo "fractions=${FRACTIONS}"
  echo "agc_window_sec=${AGC_WINDOW_SEC}"
  echo "agc_clip=${AGC_CLIP}"
  echo "pooling=${POOLING}"
  echo "run_root=${variant_root}"
  echo "============================================================"

  SITE_STUDY_PREPROCESS="${PREPROCESS}" \
  SITE_STUDY_NORMALIZE="${normalize}" \
  SITE_STUDY_AGC_WINDOW_SEC="${AGC_WINDOW_SEC}" \
  SITE_STUDY_AGC_CLIP="${AGC_CLIP}" \
  SITE_STUDY_POOLING="${POOLING}" \
  SITE_STUDY_POOLING_P="${POOLING_P}" \
  SITE_STUDY_POOLING_CHANNELWISE="${POOLING_CHANNELWISE}" \
  SITE_STUDY_METHODS="${METHODS}" \
  SITE_STUDY_FRACTIONS="${FRACTIONS}" \
  SITE_STUDY_RUN_PRETRAIN="${RUN_PRETRAIN}" \
  SITE_STUDY_RUN_ANALYZE="${RUN_ANALYZE}" \
  SITE_STUDY_RUN_TSNE="${RUN_TSNE}" \
  bash scripts/gpu/site_main_study.sh "${SITE}" "${GPU}" "${LOG_ROOT}" "${variant_root}"
done

echo "[DONE] normalization ablation completed"
echo "[INFO] root: ${RUN_ROOT_BASE}"
