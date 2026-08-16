#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLBACKEND=Agg

PYTHON_BIN="${PYTHON_BIN:-python3}"
BASE_CFG="${1:-configs/train/base_pohang.yaml}"
STAGE_CFG="${2:-configs/train/analyze.yaml}"

"${PYTHON_BIN}" -m src.detection.analysis.analyze \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg "${STAGE_CFG}"
