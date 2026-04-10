#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLBACKEND=Agg

BASE_CFG="${1:-configs/train/base_reconst.yaml}"
STAGE_CFG="${2:-configs/train/test.yaml}"

python -m src.detection.training.trainer_test \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg "${STAGE_CFG}"