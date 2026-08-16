#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RAW_DONE_JSON="${RAW_DONE_JSON:-runs/metadata_v2_safe_rerun_v1/raw_site_main_pre50_v1_gpu0_frac1/utah_2023/reconst/test/base_utah_2023__frac1/test_metrics_fixed_threshold.json}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"
LOG_ROOT="${LOG_ROOT:-logs/metadata_v2_filter_rms_cross_site_reconst_swd_interval10_v1}"
WATCH_LOG="${WATCH_LOG:-${LOG_ROOT}/after_raw_frac1_launcher.log}"

mkdir -p "${LOG_ROOT}"

echo "[$(date '+%F %T')] [WATCH_START] waiting_for=${RAW_DONE_JSON}" | tee -a "${WATCH_LOG}"

elapsed=0
while [[ ! -f "${RAW_DONE_JSON}" ]]; do
  if [[ "${MAX_WAIT_SECONDS}" != "0" && "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]]; then
    echo "[$(date '+%F %T')] [WATCH_TIMEOUT] elapsed=${elapsed}s missing=${RAW_DONE_JSON}" | tee -a "${WATCH_LOG}"
    exit 1
  fi
  echo "[$(date '+%F %T')] [WAIT] elapsed=${elapsed}s missing=${RAW_DONE_JSON}" | tee -a "${WATCH_LOG}"
  sleep "${WAIT_SECONDS}"
  elapsed=$((elapsed + WAIT_SECONDS))
done

echo "[$(date '+%F %T')] [RAW_DONE] found=${RAW_DONE_JSON}" | tee -a "${WATCH_LOG}"
echo "[$(date '+%F %T')] [LAUNCH] filter_rms cross-site reconst frac1.00" | tee -a "${WATCH_LOG}"

METADATA_V2_CROSS_KIND=filter_rms \
METADATA_V2_CROSS_GPUS="${METADATA_V2_CROSS_GPUS:-0,1,3,4}" \
METADATA_V2_CROSS_FRACTIONS=1.00 \
METADATA_V2_CROSS_TASKS="${METADATA_V2_CROSS_TASKS:-pohang|utah_2019,utah_2019|pohang,pohang|utah_2023,utah_2023|pohang,utah_2019|utah_2023,utah_2023|utah_2019}" \
METADATA_V2_CROSS_CACHE_MODE="${METADATA_V2_CROSS_CACHE_MODE:-ram}" \
METADATA_V2_CROSS_NUM_WORKERS="${METADATA_V2_CROSS_NUM_WORKERS:-1}" \
METADATA_V2_CROSS_PERSISTENT_WORKERS="${METADATA_V2_CROSS_PERSISTENT_WORKERS:-false}" \
METADATA_V2_CROSS_PREFETCH_FACTOR="${METADATA_V2_CROSS_PREFETCH_FACTOR:-2}" \
METADATA_V2_CROSS_LOG_CENTER_DIAGNOSTICS="${METADATA_V2_CROSS_LOG_CENTER_DIAGNOSTICS:-true}" \
METADATA_V2_CROSS_LOG_WASSERSTEIN_DIAGNOSTICS="${METADATA_V2_CROSS_LOG_WASSERSTEIN_DIAGNOSTICS:-true}" \
METADATA_V2_CROSS_CENTER_DIAGNOSTICS_INTERVAL="${METADATA_V2_CROSS_CENTER_DIAGNOSTICS_INTERVAL:-10}" \
bash scripts/gpu/run_metadata_v2_cross_reconst_swd.sh

echo "[$(date '+%F %T')] [DONE] filter_rms cross-site reconst frac1.00 launcher finished" | tee -a "${WATCH_LOG}"
