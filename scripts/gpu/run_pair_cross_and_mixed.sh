#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
GPU_LIST="${2:-2}"
LOG_ROOT="${3:-logs}"
RUN_ROOT_BASE="${4:-runs/pair_pohang_utah2019_study}"

export PYTHONPATH=.
export LD_LIBRARY_PATH=/home/anaconda3/lib:${LD_LIBRARY_PATH:-}

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

PAIR_METHODS="${PAIR_METHODS:-scratch,reconst}"
PAIR_RUN_ANALYZE="${PAIR_RUN_ANALYZE:-true}"
PAIR_RUN_TSNE="${PAIR_RUN_TSNE:-false}"
PAIR_RUN_PRETRAIN="${PAIR_RUN_PRETRAIN:-true}"

case "${MODE}" in
  full)
    PAIR_RUN_MIXED=true
    PAIR_RUN_CROSS=true
    PAIR_RUN_CROSS_P2U=true
    PAIR_RUN_CROSS_U2P=true
    ;;
  mixed)
    PAIR_RUN_MIXED=true
    PAIR_RUN_CROSS=false
    PAIR_RUN_CROSS_P2U=false
    PAIR_RUN_CROSS_U2P=false
    ;;
  cross)
    PAIR_RUN_MIXED=false
    PAIR_RUN_CROSS=true
    PAIR_RUN_CROSS_P2U=true
    PAIR_RUN_CROSS_U2P=true
    ;;
  cross_p2u)
    PAIR_RUN_MIXED=false
    PAIR_RUN_CROSS=true
    PAIR_RUN_CROSS_P2U=true
    PAIR_RUN_CROSS_U2P=false
    ;;
  cross_u2p)
    PAIR_RUN_MIXED=false
    PAIR_RUN_CROSS=true
    PAIR_RUN_CROSS_P2U=false
    PAIR_RUN_CROSS_U2P=true
    ;;
  *)
    echo "[ERROR] unsupported MODE: ${MODE}"
    echo "usage: bash scripts/gpu/run_pair_cross_and_mixed.sh [full|mixed|cross|cross_p2u|cross_u2p] [GPU] [LOG_ROOT] [RUN_ROOT_BASE]"
    exit 1
    ;;
esac

export PYTHON_BIN
export PAIR_METHODS
export PAIR_RUN_ANALYZE
export PAIR_RUN_TSNE
export PAIR_RUN_PRETRAIN
export PAIR_RUN_MIXED
export PAIR_RUN_CROSS
export PAIR_RUN_CROSS_P2U
export PAIR_RUN_CROSS_U2P

echo "[INFO] MODE=${MODE}"
echo "[INFO] GPU=${GPU_LIST}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] RUN_ROOT_BASE=${RUN_ROOT_BASE}"
echo "[INFO] PAIR_METHODS=${PAIR_METHODS}"
echo "[INFO] PAIR_RUN_ANALYZE=${PAIR_RUN_ANALYZE}"
echo "[INFO] PAIR_RUN_TSNE=${PAIR_RUN_TSNE}"
echo "[INFO] PAIR_RUN_PRETRAIN=${PAIR_RUN_PRETRAIN}"
echo "[INFO] PAIR_RUN_CROSS_P2U=${PAIR_RUN_CROSS_P2U}"
echo "[INFO] PAIR_RUN_CROSS_U2P=${PAIR_RUN_CROSS_U2P}"

bash scripts/gpu/pair_cross_and_mixed_study.sh "${GPU_LIST}" "${LOG_ROOT}" "${RUN_ROOT_BASE}"
