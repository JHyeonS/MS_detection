#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLBACKEND=Agg

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

DATASET="visualbest_filter_rms_fs1000_rms0p15_lp50"
RUN_ROOT_BASE="runs/visualbest_filter_rms_fs1000_rms0p15_lp50_site_main_pre50_v1"
RUN_ROOT="${RUN_ROOT_BASE}/pohang_contrast"
LOG_ROOT="${FILTER_RMS_LOG_ROOT:-logs/filter_rms_remaining_gpu12_v3}"
SCHEDULER_LOG="${LOG_ROOT}/scheduler.log"
LOG_FILE="${LOG_ROOT}/${DATASET}__pohang_contrast_after_remaining_gpu12.stdout.log"
WAIT_LOG="${LOG_ROOT}/pohang_contrast_after_remaining_gpu12.wait.log"
FRACTIONS="0.05,0.10,0.25,0.50,1.00"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"

is_complete() {
  local frac tag
  for frac in 0p05 0p1 0p25 0p5 1; do
    if [[ ! -f "${RUN_ROOT}/contrast/test/pohang__frac${frac}/test_metrics_fixed_threshold.json" ]]; then
      return 1
    fi
  done
  return 0
}

pick_gpu() {
  python - <<'PY'
import subprocess

preferred = {"1", "2"}
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
best = None
for line in out.splitlines():
    if not line.strip():
        continue
    idx, mem = [x.strip() for x in line.split(",")[:2]]
    if idx not in preferred:
        continue
    mem_i = int(mem)
    if best is None or mem_i < best[1]:
        best = (idx, mem_i)
if best is not None and best[1] < 1000:
    print(best[0])
PY
}

if is_complete; then
  echo "[$(date '+%F %T')] [SKIP_COMPLETE] pohang contrast already complete" | tee -a "${WAIT_LOG}"
  exit 0
fi

echo "[$(date '+%F %T')] [WAIT] waiting for main filter_rms scheduler to finish" | tee -a "${WAIT_LOG}"
while ! grep -q "\[COMPLETE\] filter_rms remaining tasks finished" "${SCHEDULER_LOG}" 2>/dev/null; do
  sleep 120
done

if is_complete; then
  echo "[$(date '+%F %T')] [SKIP_COMPLETE] pohang contrast completed while waiting" | tee -a "${WAIT_LOG}"
  exit 0
fi

echo "[$(date '+%F %T')] [WAIT] scheduler complete; waiting for a free GPU among 1,2" | tee -a "${WAIT_LOG}"
GPU=""
while [[ -z "${GPU}" ]]; do
  GPU="$(pick_gpu)"
  if [[ -z "${GPU}" ]]; then
    sleep 120
  fi
done

echo "[$(date '+%F %T')] [LAUNCH] gpu=${GPU} pohang contrast ${FRACTIONS}" | tee -a "${WAIT_LOG}"
export SITE_STUDY_SPLIT_DIR="data/${DATASET}/metadata/experiments/stage1_pohang_only"
export SITE_STUDY_PREPROCESS="load_only"
export SITE_STUDY_NORMALIZE="none"
export SITE_STUDY_METHODS="contrast"
export SITE_STUDY_FRACTIONS="${FRACTIONS}"
export SITE_STUDY_RUN_ANALYZE="true"
export SITE_STUDY_RUN_TSNE="false"
export SITE_STUDY_RUN_PRETRAIN="true"
export SITE_STUDY_LOG_CENTER_DIAGNOSTICS="false"
export SITE_STUDY_LOG_WASSERSTEIN_DIAGNOSTICS="false"
export SITE_STUDY_NUM_WORKERS="1"
export SITE_STUDY_CACHE_MODE="${SITE_STUDY_CACHE_MODE:-ram}"
export SITE_STUDY_PRETRAIN_EPOCHS="50"
export SITE_STUDY_PREFETCH_FACTOR="2"
export SITE_STUDY_PERSISTENT_WORKERS="true"
export SITE_STUDY_LOG_SLUG="${DATASET}_pohang_contrast_after_remaining_gpu12"

bash scripts/gpu/site_main_study.sh "pohang" "${GPU}" "${LOG_ROOT}" "${RUN_ROOT}" > "${LOG_FILE}" 2>&1
