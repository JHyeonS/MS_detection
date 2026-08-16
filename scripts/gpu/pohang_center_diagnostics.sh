#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LOG_ROOT="${2:-logs}"
RUN_ROOT_BASE="${3:-runs/pohang_center_diagnostics}"

export PYTHONPATH=.
export MPLBACKEND=Agg
export LD_LIBRARY_PATH=/home/anaconda3/lib:${LD_LIBRARY_PATH:-}

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
PRETRAIN_ENCODER="${POHANG_RECONST_ENCODER:-runs/pohang_main_study/reconst/pretrain/pohang/best_encoder.pt}"
FRACTIONS="${POHANG_CENTER_FRACTIONS:-0.05,0.10,0.50,1.00}"
METHODS="${POHANG_CENTER_METHODS:-fixed_center,dynamic_center}"

TMP_DIR=".tmp_pohang_center_diagnostics_cfg"
mkdir -p "${LOG_ROOT}" "${TMP_DIR}" "${RUN_ROOT_BASE}"
LOG_FILE="${LOG_ROOT}/pohang_center_diagnostics.log"

if [[ ! -f "${PRETRAIN_ENCODER}" ]]; then
  echo "[ERROR] reconstruction pretrain encoder not found: ${PRETRAIN_ENCODER}" | tee -a "${LOG_FILE}"
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
  local run_root="$1"
  local out_cfg="$2"
  "${PYTHON_BIN}" - "${BASE_TEMPLATE}" "${run_root}" "${out_cfg}" <<'PY'
import sys, yaml
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
import sys, yaml
src, suffix, outp = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg["data"]["experiment"] = f"{cfg['data']['experiment']}__{suffix}"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_stage_cfg() {
  local run_root="$1"
  local fraction="$2"
  local center_update="$3"
  local out_cfg="$4"
  "${PYTHON_BIN}" - "${FINETUNE_TEMPLATE}" "${run_root}" "${fraction}" "${PRETRAIN_ENCODER}" "${center_update}" "${out_cfg}" <<'PY'
import sys, yaml
src, run_root, fraction, encoder, center_update, outp = sys.argv[1:7]
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
train["pretrained_encoder_path"] = encoder
train["center_mode"] = "target_noise"
train["center_update"] = center_update
train["log_center_diagnostics"] = True
train["center_ablation"] = f"target_noise_{center_update}"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

{
  echo "============================================================"
  echo "[POHANG CENTER DIAGNOSTICS START] $(date '+%F %T')"
  echo "python_bin=${PYTHON_BIN}"
  echo "gpu=${GPU}"
  echo "run_root_base=${RUN_ROOT_BASE}"
  echo "pretrain_encoder=${PRETRAIN_ENCODER}"
  echo "methods=${METHODS}"
  echo "fractions=${FRACTIONS}"
  echo "============================================================"
} | tee -a "${LOG_FILE}"

IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"
IFS=',' read -r -a FRACTION_ARRAY <<< "${FRACTIONS}"

for method in "${METHOD_ARRAY[@]}"; do
  method="$(echo "${method}" | xargs)"
  case "${method}" in
    fixed_center) center_update="once" ;;
    dynamic_center) center_update="every_epoch" ;;
    *)
      echo "[ERROR] unsupported method: ${method}" | tee -a "${LOG_FILE}"
      exit 1
      ;;
  esac

  method_root="${RUN_ROOT_BASE}/${method}"
  base_cfg="${TMP_DIR}/base_${method}.yaml"
  make_base_cfg "${method_root}" "${base_cfg}"

  for fraction in "${FRACTION_ARRAY[@]}"; do
    fraction="$(echo "${fraction}" | xargs)"
    tag="frac$(frac_tag "${fraction}")"
    stage_cfg="${TMP_DIR}/finetune_${method}_${tag}.yaml"
    eval_base_cfg="${TMP_DIR}/eval_base_${method}_${tag}.yaml"
    run_log="${LOG_ROOT}/pohang_center_diagnostics__${method}__${tag}.log"
    finetune_dir="${method_root}/finetune/pohang__${tag}"

    make_stage_cfg "${method_root}" "${fraction}" "${center_update}" "${stage_cfg}"
    make_eval_base_cfg "${base_cfg}" "${tag}" "${eval_base_cfg}"

    {
      echo "============================================================"
      echo "[POHANG CENTER DIAGNOSTICS RUN] $(date '+%F %T') method=${method} fraction=${fraction} gpu=${GPU}"
      echo "center_update=${center_update}"
      echo "stage_cfg=${stage_cfg}"
      echo "finetune_dir=${finetune_dir}"
      echo "============================================================"
    } | tee -a "${LOG_FILE}" | tee -a "${run_log}"

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
      --base_cfg "${base_cfg}" \
      --stage_cfg "${stage_cfg}" \
      --exp_suffix "${tag}" >> "${run_log}" 2>&1

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
      --base_cfg "${eval_base_cfg}" \
      --stage_cfg "${TEST_TEMPLATE}" >> "${run_log}" 2>&1

    echo "[DONE] method=${method} fraction=${fraction}" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
  done
done

"${PYTHON_BIN}" scripts/gpu/plot_center_diagnostics.py \
  --fixed-root "${RUN_ROOT_BASE}/fixed_center" \
  --dynamic-root "${RUN_ROOT_BASE}/dynamic_center" \
  --out-dir "${RUN_ROOT_BASE}/figures" \
  --site-title "Pohang" >> "${LOG_FILE}" 2>&1

echo "[DONE] Pohang center diagnostics completed" | tee -a "${LOG_FILE}"
echo "[INFO] figures: ${RUN_ROOT_BASE}/figures" | tee -a "${LOG_FILE}"
