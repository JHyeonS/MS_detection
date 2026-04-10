#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLBACKEND=Agg

BASE_CFG="${1:-config/base_reconst.yaml}"
STAGE_CFG="${2:-config/test.yaml}"

python src/training/trainer_test.py \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg "${STAGE_CFG}"