#!/usr/bin/env bash
set -euo pipefail

BASE_CFG="${1:-configs/train/base.yaml}"
STAGE_CFG="${2:-configs/train/analyze.yaml}"

python src/analysis/analyze.py \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg "${STAGE_CFG}"