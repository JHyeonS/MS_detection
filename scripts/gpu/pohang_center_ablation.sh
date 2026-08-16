#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LOG_ROOT="${2:-logs}"
RUN_ROOT="${3:-runs/pohang_center_ablation/reconst_dynamic}"

export PYTHONPATH=.
export MPLBACKEND=Agg
export LD_LIBRARY_PATH=/home/anaconda3/lib:${LD_LIBRARY_PATH:-}
export MS_JOB_OWNER="${MS_JOB_OWNER:-${USER:-unknown}}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [[ -x "/home/ted1204/.conda/envs/ms_detection/bin/python" ]]; then
    PYTHON_BIN="/home/ted1204/.conda/envs/ms_detection/bin/python"
  elif [[ -x "/home/anaconda3/bin/python3.9" ]]; then
    PYTHON_BIN="/home/anaconda3/bin/python3.9"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "[ERROR] could not resolve PYTHON_BIN"
    exit 1
  fi
fi

BASE_TEMPLATE="configs/train/base_pohang_arch_best.yaml"
FINETUNE_TEMPLATE="configs/train/final_pohang_best.yaml"
TEST_TEMPLATE="configs/train/test.yaml"
ANALYZE_TEMPLATE="configs/train/analyze.yaml"
PRETRAIN_ENCODER="${POHANG_RECONST_ENCODER:-runs/pohang_main_study/reconst/pretrain/pohang/best_encoder.pt}"

FRACTIONS="${POHANG_CENTER_FRACTIONS:-0.05,0.10,0.50,1.00}"
RUN_TSNE="${POHANG_CENTER_RUN_TSNE:-true}"
RUN_ANALYZE="${POHANG_CENTER_RUN_ANALYZE:-true}"

TMP_DIR=".tmp_pohang_center_ablation_cfg"
mkdir -p "${LOG_ROOT}" "${TMP_DIR}" "${RUN_ROOT}"

LOG_FILE="${LOG_ROOT}/pohang_center_ablation.log"

if [[ ! -f "${PRETRAIN_ENCODER}" ]]; then
  echo "[ERROR] reconstruction pretrain encoder not found: ${PRETRAIN_ENCODER}" | tee -a "${LOG_FILE}"
  echo "[ERROR] run pohang_main_study reconst first, or set POHANG_RECONST_ENCODER=/path/to/best_encoder.pt" | tee -a "${LOG_FILE}"
  exit 1
fi

frac_tag() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
x = float(sys.argv[1])
print(("{:.6f}".format(x)).rstrip("0").rstrip(".").replace(".", "p"))
PY
}

make_base_cfg() {
  local out_cfg="$1"
  "${PYTHON_BIN}" - "${BASE_TEMPLATE}" "${RUN_ROOT}" "${out_cfg}" <<'PY'
import sys
import yaml
src, run_root, outp = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
cfg.setdefault("data", {})["experiment"] = "pohang"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_eval_base_cfg() {
  local base_cfg="$1"
  local suffix="$2"
  local out_cfg="$3"
  "${PYTHON_BIN}" - "${base_cfg}" "${suffix}" "${out_cfg}" <<'PY'
import sys
import yaml
src, suffix, outp = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg["data"]["experiment"] = f"{cfg['data']['experiment']}__{suffix}"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_dynamic_stage_cfg() {
  local fraction="$1"
  local out_cfg="$2"
  "${PYTHON_BIN}" - "${FINETUNE_TEMPLATE}" "${RUN_ROOT}" "${fraction}" "${PRETRAIN_ENCODER}" "${out_cfg}" <<'PY'
import sys
import yaml
src, run_root, fraction, pretrained_encoder, outp = sys.argv[1:6]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
train = cfg.setdefault("train", {})
cfg.setdefault("paths", {})["run_root"] = run_root
fraction = float(fraction)
train["use_labeled_fraction"] = fraction < 1.0
train["labeled_fraction"] = fraction
train["fraction_seed"] = 42
train["balance_fraction_by_class"] = True
train["min_samples_per_class"] = 1
train["drop_last"] = False
train["use_pretrained_encoder"] = True
train["pretrained_encoder_path"] = pretrained_encoder
train["center_mode"] = "target_noise"
train["center_update"] = "every_epoch"
train["log_center_diagnostics"] = True
train["center_ablation"] = "dynamic_target_noise_every_epoch"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

run_eval_steps() {
  local eval_base_cfg="$1"
  local finetune_dir="$2"
  local log_file="$3"

  echo "[POHANG CENTER] test" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
    --base_cfg "${eval_base_cfg}" \
    --stage_cfg "${TEST_TEMPLATE}" >> "${log_file}" 2>&1

  if [[ "${RUN_ANALYZE}" == "true" ]]; then
    echo "[POHANG CENTER] analyze" | tee -a "${log_file}"
    "${PYTHON_BIN}" -m src.detection.analysis.analyze \
      --base_cfg "${eval_base_cfg}" \
      --stage_cfg "${ANALYZE_TEMPLATE}" >> "${log_file}" 2>&1
  fi

  if [[ "${RUN_TSNE}" == "true" ]]; then
    echo "[POHANG CENTER] tsne" | tee -a "${log_file}"
    PYTHON_BIN="${PYTHON_BIN}" bash scripts/gpu/run_tsne.sh "${GPU}" "${finetune_dir}" >> "${log_file}" 2>&1
  fi
}

{
  echo "============================================================"
  echo "[POHANG CENTER ABLATION START] $(date '+%F %T')"
  echo "python_bin=${PYTHON_BIN}"
  echo "gpu=${GPU}"
  echo "run_root=${RUN_ROOT}"
  echo "pretrain_encoder=${PRETRAIN_ENCODER}"
  echo "fractions=${FRACTIONS}"
  echo "center_mode=target_noise"
  echo "center_update=every_epoch"
  echo "run_analyze=${RUN_ANALYZE}"
  echo "run_tsne=${RUN_TSNE}"
  echo "============================================================"
} | tee -a "${LOG_FILE}"

base_cfg="${TMP_DIR}/base_reconst_dynamic.yaml"
make_base_cfg "${base_cfg}"

IFS=',' read -r -a FRACTION_ARRAY <<< "${FRACTIONS}"
for fraction in "${FRACTION_ARRAY[@]}"; do
  fraction="$(echo "${fraction}" | xargs)"
  tag="frac$(frac_tag "${fraction}")"
  suffix="${tag}"
  stage_cfg="${TMP_DIR}/finetune_reconst_dynamic_${tag}.yaml"
  eval_base_cfg="${TMP_DIR}/eval_base_reconst_dynamic_${tag}.yaml"
  run_log="${LOG_ROOT}/pohang_center_ablation__reconst_dynamic__${tag}.log"
  finetune_dir="${RUN_ROOT}/finetune/pohang__${suffix}"

  make_dynamic_stage_cfg "${fraction}" "${stage_cfg}"
  make_eval_base_cfg "${base_cfg}" "${suffix}" "${eval_base_cfg}"

  {
    echo "============================================================"
    echo "[POHANG CENTER RUN] $(date '+%F %T') fraction=${fraction} gpu=${GPU}"
    echo "base_cfg=${base_cfg}"
    echo "stage_cfg=${stage_cfg}"
    echo "eval_base_cfg=${eval_base_cfg}"
    echo "finetune_dir=${finetune_dir}"
    echo "============================================================"
  } | tee -a "${LOG_FILE}" | tee -a "${run_log}"

  echo "[POHANG CENTER] finetune dynamic center" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
    --base_cfg "${base_cfg}" \
    --stage_cfg "${stage_cfg}" \
    --exp_suffix "${suffix}" >> "${run_log}" 2>&1

  run_eval_steps "${eval_base_cfg}" "${finetune_dir}" "${run_log}"
  echo "[DONE] reconst_dynamic fraction=${fraction}" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
done

"${PYTHON_BIN}" scripts/gpu/summarize_pohang_main_study.py \
  --root "$(dirname "${RUN_ROOT}")" \
  --out "$(dirname "${RUN_ROOT}")/summary.csv" >> "${LOG_FILE}" 2>&1

echo "[DONE] Pohang center ablation completed" | tee -a "${LOG_FILE}"
echo "[INFO] summary: $(dirname "${RUN_ROOT}")/summary.csv" | tee -a "${LOG_FILE}"
