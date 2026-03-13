#!/usr/bin/env bash
set -e

export PYTHONPATH=.

BASE_CONFIG=config/base.yaml
CONFIG=config/pretrain_contrast.yaml

python src/training/trainer_contrast.py \
  --base_config $BASE_CONFIG \
  --config $CONFIG