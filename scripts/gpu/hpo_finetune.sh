#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.
export MPLBACKEND=Agg

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/home/anaconda3/bin/python3.9" ]]; then
    PYTHON_BIN="/home/anaconda3/bin/python3.9"
  else
    PYTHON_BIN="python3"
  fi
fi

SPEC_PATH="${1:-configs/experiments/hpo/finetune_stage1_pohang.yaml}"

"${PYTHON_BIN}" scripts/gpu/hpo_finetune.py --spec "${SPEC_PATH}"
