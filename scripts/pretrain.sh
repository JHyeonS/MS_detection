#!/usr/bin/env bash
set -e

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=1

export MPLBACKEND=Agg

BASE_CONFIG=config/base.yaml
CONFIG=config/pretrain.yaml

python src/training/trainer_pretrain.py \
  --base_cfg $BASE_CONFIG \
  --stage_cfg $CONFIG