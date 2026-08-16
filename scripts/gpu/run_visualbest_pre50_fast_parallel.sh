#!/usr/bin/env bash
set -euo pipefail

# Faster visualbest protocol:
# - keep fewer concurrent jobs to reduce dataloader/file-open contention
# - increase pretrain batch size to reduce epoch iterations
# - keep pretrain workers alive across epochs

export VISUALBEST_GPUS="${VISUALBEST_GPUS:-0,1,2,3,4,5}"
export VISUALBEST_PRETRAIN_EPOCHS="${VISUALBEST_PRETRAIN_EPOCHS:-50}"
export VISUALBEST_PRETRAIN_BATCH_SIZE="${VISUALBEST_PRETRAIN_BATCH_SIZE:-32}"
export VISUALBEST_NUM_WORKERS="${VISUALBEST_NUM_WORKERS:-1}"
export VISUALBEST_PREFETCH_FACTOR="${VISUALBEST_PREFETCH_FACTOR:-4}"
export VISUALBEST_PERSISTENT_WORKERS="${VISUALBEST_PERSISTENT_WORKERS:-true}"
export VISUALBEST_RUN_SUFFIX="${VISUALBEST_RUN_SUFFIX:-pre50_fast_v1}"
export VISUALBEST_PARALLEL_LOG_ROOT="${VISUALBEST_PARALLEL_LOG_ROOT:-logs/visualbest_pre50_fast_parallel_v1}"

bash scripts/gpu/run_visualbest_pre50_parallel.sh
