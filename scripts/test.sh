#!/usr/bin/env bash
set -e

export PYTHONPATH=.

BASE_CONFIG=config/base.yaml
CONFIG=config/test.yaml

python src/training/trainer_test.py \
  --base_cfg $BASE_CONFIG \
  --stage_cfg $CONFIG