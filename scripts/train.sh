#!/usr/bin/env bash
set -e

export PYTHONPATH=.

BASE_CONFIG=config/base.yaml
CONFIG=config/train.yaml

python src/training/trainer_msd.py \
  --base_config $BASE_CONFIG \
  --config $CONFIG