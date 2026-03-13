#!/usr/bin/env bash
set -euo pipefail

BASE_CFG="${1:-config/base.yaml}"
STAGE_CFG="${2:-config/analyze.yaml}"

python src/analysis/analyze.py \
  --base_cfg "${BASE_CFG}" \
  --stage_cfg "${STAGE_CFG}"