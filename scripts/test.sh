#!/usr/bin/env bash
set -e

export PYTHONPATH=.

BASE_CONFIG=config/base.yaml
CONFIG=config/pretrain.yaml

python src/training/trainer_cae.py \
  --base_config $BASE_CONFIG \
  --config $CONFIG