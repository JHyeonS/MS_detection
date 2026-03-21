#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${1:-logs_stage2_analyze}"

mkdir -p "${LOG_ROOT}"
mkdir -p .tmp_stage2_analyze_cfg

export PYTHONPATH=.

BASE_CONFIGS=(
  "config/base_contrast_top1.yaml"
  "config/base_reconst_top1.yaml"
  "config/base_reconst_top2.yaml"
  "config/base_reconst_top3.yaml"
)

TRAIN_CONFIGS=(
  "config/train_stage2_freeze_lr1e-04_aw0p01.yaml"
  "config/train_stage2_freeze_lr3e-04_aw0p01.yaml"
  "config/train_stage2_unfreeze_lr1e-04_aw0p01.yaml"
  "config/train_stage2_unfreeze_lr3e-04_aw0p01.yaml"
  "config/train_stage2_freeze_lr1e-04_aw0p10.yaml"
  "config/train_stage2_unfreeze_lr1e-04_aw0p10.yaml"
  "config/train_stage2_freeze_lr3e-04_aw0p10.yaml"
  "config/train_stage2_unfreeze_lr3e-04_aw0p10.yaml"
)

make_analyze_base_cfg() {
  local src_base_cfg="$1"
  local exp_suffix="$2"
  local out_cfg="$3"

  python - "${src_base_cfg}" "${exp_suffix}" "${out_cfg}" <<'PY'
import sys, yaml
src, suffix, outp = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
base_exp = cfg["data"]["experiment"]
cfg["data"]["experiment"] = f"{base_exp}__{suffix}"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(cfg["data"]["experiment"])
PY
}

for base_cfg in "${BASE_CONFIGS[@]}"; do
  for train_cfg in "${TRAIN_CONFIGS[@]}"; do
    base_name="$(basename "${base_cfg}" .yaml)"
    train_name="$(basename "${train_cfg}" .yaml)"
    exp_suffix="${train_name}"
    analyze_base_cfg=".tmp_stage2_analyze_cfg/${base_name}__${train_name}.yaml"
    log_file="${LOG_ROOT}/${base_name}__${train_name}.log"

    {
      echo "============================================================"
      echo "[START] $(date '+%F %T')"
      echo "base_cfg=${base_cfg}"
      echo "train_cfg=${train_cfg}"
      echo "exp_suffix=${exp_suffix}"
      echo "analyze_base_cfg=${analyze_base_cfg}"
      echo "============================================================"
    } | tee -a "${log_file}"

    make_analyze_base_cfg "${base_cfg}" "${exp_suffix}" "${analyze_base_cfg}" >> "${log_file}" 2>&1

    python src/analysis/analyze.py \
      --base_cfg "${analyze_base_cfg}" \
      --stage_cfg "config/analyze.yaml" >> "${log_file}" 2>&1

    status=$?
    echo "[DONE] ${base_name} + ${train_name} status=${status}" | tee -a "${log_file}"

    if [[ "${status}" -ne 0 ]]; then
      echo "[ERROR] analyze failed for ${base_name} + ${train_name}" | tee -a "${log_file}"
      exit "${status}"
    fi
  done
done

echo "[DONE] all selected stage2 analyze jobs finished successfully."
