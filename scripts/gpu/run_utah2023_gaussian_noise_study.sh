#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LOG_ROOT="${2:-logs/utah_2023_gaussian_noise_study}"
RUN_ROOT="${3:-runs/utah_2023_gaussian_noise_study}"
SPLIT_ROOT="${4:-runs/utah_2023_gaussian_noise_split/splits}"
NOISE_ROOT="${5:-runs/utah_2023_gaussian_noise_split/gaussian_noise}"

export PYTHONPATH=.
export MPLBACKEND=Agg

PYTHON_BIN="${PYTHON_BIN:-python}"

GAUSSIAN_MEAN="${UTAH2023_GAUSSIAN_MEAN:-0.0}"
GAUSSIAN_STD="${UTAH2023_GAUSSIAN_STD:-1.0}"
GAUSSIAN_SEED="${UTAH2023_GAUSSIAN_SEED:-42}"
METHODS="${SITE_STUDY_METHODS:-scratch,reconst}"
FRACTIONS="${SITE_STUDY_FRACTIONS:-0.25,1.00}"
RUN_ANALYZE="${SITE_STUDY_RUN_ANALYZE:-true}"
RUN_TSNE="${SITE_STUDY_RUN_TSNE:-false}"
RUN_PRETRAIN="${SITE_STUDY_RUN_PRETRAIN:-true}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"

echo "[RUN] Prepare Utah 2023 Gaussian-noise split"
echo "split_root=${SPLIT_ROOT}"
echo "noise_root=${NOISE_ROOT}"
echo "gaussian_mean=${GAUSSIAN_MEAN}"
echo "gaussian_std=${GAUSSIAN_STD}"

"${PYTHON_BIN}" scripts/gpu/prepare_utah2023_gaussian_noise_split.py \
  --out-split-dir "${SPLIT_ROOT}" \
  --noise-root "${NOISE_ROOT}" \
  --mean "${GAUSSIAN_MEAN}" \
  --std "${GAUSSIAN_STD}" \
  --seed "${GAUSSIAN_SEED}"

echo "[RUN] Utah 2023 Gaussian-noise training"
echo "gpu=${GPU}"
echo "run_root=${RUN_ROOT}"
echo "methods=${METHODS}"
echo "fractions=${FRACTIONS}"

SITE_STUDY_SPLIT_DIR="${SPLIT_ROOT}" \
SITE_STUDY_METHODS="${METHODS}" \
SITE_STUDY_FRACTIONS="${FRACTIONS}" \
SITE_STUDY_RUN_PRETRAIN="${RUN_PRETRAIN}" \
SITE_STUDY_RUN_ANALYZE="${RUN_ANALYZE}" \
SITE_STUDY_RUN_TSNE="${RUN_TSNE}" \
bash scripts/gpu/site_main_study.sh utah_2023 "${GPU}" "${LOG_ROOT}" "${RUN_ROOT}"

RECONST_DIR="${RUN_ROOT}/reconst/pretrain/base_utah_2023"
RECONST_CKPT="${RECONST_DIR}/best.pt"
BASE_CFG="${RECONST_DIR}/config_snapshot/base_config.yaml"
STAGE_CFG="${RECONST_DIR}/config_snapshot/stage_config.yaml"

if [[ -f "${RECONST_CKPT}" && -f "${BASE_CFG}" && -f "${STAGE_CFG}" ]]; then
  echo "[RUN] Reconstruction patch diagnostics"
  "${PYTHON_BIN}" scripts/gpu/plot_reconstruction_patches.py \
    --base-cfg "${BASE_CFG}" \
    --stage-cfg "${STAGE_CFG}" \
    --checkpoint "${RECONST_CKPT}" \
    --csv "${SPLIT_ROOT}/test.csv" \
    --out-dir "${RUN_ROOT}/reconst/reconstruction_diagnostics" \
    --num-samples "${RECONST_DIAG_NUM_SAMPLES:-6}" \
    --seed "${GAUSSIAN_SEED}" \
    --device "cuda" \
    --split-name "test"
else
  echo "[WARN] Reconstruction checkpoint/config not found; skip patch diagnostics"
  echo "[WARN] expected checkpoint: ${RECONST_CKPT}"
fi

echo "[DONE] Utah 2023 Gaussian-noise study"
echo "[INFO] run_root=${RUN_ROOT}"
