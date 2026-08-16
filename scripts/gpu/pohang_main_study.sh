#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LOG_ROOT="${2:-logs}"
RUN_ROOT_BASE="${3:-runs/pohang_main_study}"

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
PRETRAIN_RECONST_TEMPLATE="configs/train/pretrain_reconst.yaml"
PRETRAIN_CONTRAST_TEMPLATE="configs/train/pretrain_contrast.yaml"
TEST_TEMPLATE="configs/train/test.yaml"
ANALYZE_TEMPLATE="configs/train/analyze.yaml"

FRACTIONS="${POHANG_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
METHODS="${POHANG_METHODS:-scratch,reconst,contrast,reconst_noanom}"
RUN_TSNE="${POHANG_RUN_TSNE:-true}"
RUN_ANALYZE="${POHANG_RUN_ANALYZE:-true}"
RUN_PRETRAIN="${POHANG_RUN_PRETRAIN:-true}"

TMP_DIR=".tmp_pohang_main_study_cfg"
mkdir -p "${LOG_ROOT}" "${TMP_DIR}" "${RUN_ROOT_BASE}"

LOG_FILE="${LOG_ROOT}/pohang_main_study.log"

frac_tag() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
x = float(sys.argv[1])
print(("{:.6f}".format(x)).rstrip("0").rstrip(".").replace(".", "p"))
PY
}

make_base_cfg() {
  local method="$1"
  local run_root="$2"
  local out_cfg="$3"
  "${PYTHON_BIN}" - "${BASE_TEMPLATE}" "${run_root}" "${out_cfg}" <<'PY'
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

make_pretrain_stage_cfg() {
  local template="$1"
  local run_root="$2"
  local out_cfg="$3"
  "${PYTHON_BIN}" - "${template}" "${run_root}" "${out_cfg}" <<'PY'
import sys
import yaml

src, run_root, outp = sys.argv[1:4]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
cfg.setdefault("pretrain", {})["epochs"] = int(cfg["pretrain"].get("epochs", 150))
cfg["pretrain"]["use_amp"] = True
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_finetune_stage_cfg() {
  local method="$1"
  local run_root="$2"
  local fraction="$3"
  local pretrained_encoder="$4"
  local anomaly_loss_weight="$5"
  local out_cfg="$6"
  "${PYTHON_BIN}" - "${FINETUNE_TEMPLATE}" "${run_root}" "${fraction}" "${pretrained_encoder}" "${anomaly_loss_weight}" "${out_cfg}" <<'PY'
import sys
import yaml

src, run_root, fraction, pretrained_encoder, anomaly_loss_weight, outp = sys.argv[1:7]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

train = cfg.setdefault("train", {})
paths = cfg.setdefault("paths", {})
paths["run_root"] = run_root

fraction = float(fraction)
train["use_labeled_fraction"] = fraction < 1.0
train["labeled_fraction"] = fraction
train["fraction_seed"] = 42
train["balance_fraction_by_class"] = True
train["min_samples_per_class"] = 1
train["drop_last"] = False
train["anomaly_loss_weight"] = float(anomaly_loss_weight)

if pretrained_encoder == "none":
    train["use_pretrained_encoder"] = False
    train["pretrained_encoder_path"] = None
else:
    train["use_pretrained_encoder"] = True
    train["pretrained_encoder_path"] = pretrained_encoder

with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

run_pretrain_if_needed() {
  local method="$1"
  local run_root="$2"
  local base_cfg="$3"
  local log_file="$4"

  if [[ "${RUN_PRETRAIN}" != "true" ]]; then
    echo "[SKIP] pretrain disabled by POHANG_RUN_PRETRAIN=${RUN_PRETRAIN}" | tee -a "${log_file}"
    return
  fi

  case "${method}" in
    reconst)
      local stage_cfg="${TMP_DIR}/pretrain_${method}.yaml"
      make_pretrain_stage_cfg "${PRETRAIN_RECONST_TEMPLATE}" "${run_root}" "${stage_cfg}"
      ;;
    contrast)
      local stage_cfg="${TMP_DIR}/pretrain_${method}.yaml"
      make_pretrain_stage_cfg "${PRETRAIN_CONTRAST_TEMPLATE}" "${run_root}" "${stage_cfg}"
      ;;
    *)
      return
      ;;
  esac

  local ckpt="${run_root}/pretrain/pohang/best_encoder.pt"
  if [[ -f "${ckpt}" ]]; then
    echo "[SKIP] existing pretrain checkpoint: ${ckpt}" | tee -a "${log_file}"
    return
  fi

  echo "[POHANG STUDY] pretrain method=${method}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_pretrain \
    --base_cfg "${base_cfg}" \
    --stage_cfg "${stage_cfg}" >> "${log_file}" 2>&1
}

run_eval_steps() {
  local gpu="$1"
  local eval_base_cfg="$2"
  local finetune_dir="$3"
  local log_file="$4"

  echo "[POHANG STUDY] test" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
    --base_cfg "${eval_base_cfg}" \
    --stage_cfg "${TEST_TEMPLATE}" >> "${log_file}" 2>&1

  if [[ "${RUN_ANALYZE}" == "true" ]]; then
    echo "[POHANG STUDY] analyze" | tee -a "${log_file}"
    "${PYTHON_BIN}" -m src.detection.analysis.analyze \
      --base_cfg "${eval_base_cfg}" \
      --stage_cfg "${ANALYZE_TEMPLATE}" >> "${log_file}" 2>&1
  fi

  if [[ "${RUN_TSNE}" == "true" ]]; then
    echo "[POHANG STUDY] tsne" | tee -a "${log_file}"
    PYTHON_BIN="${PYTHON_BIN}" bash scripts/gpu/run_tsne.sh "${gpu}" "${finetune_dir}" >> "${log_file}" 2>&1
  fi
}

{
  echo "============================================================"
  echo "[POHANG MAIN STUDY START] $(date '+%F %T')"
  echo "python_bin=${PYTHON_BIN}"
  echo "gpu=${GPU}"
  echo "run_root_base=${RUN_ROOT_BASE}"
  echo "methods=${METHODS}"
  echo "fractions=${FRACTIONS}"
  echo "run_pretrain=${RUN_PRETRAIN}"
  echo "run_analyze=${RUN_ANALYZE}"
  echo "run_tsne=${RUN_TSNE}"
  echo "============================================================"
} | tee -a "${LOG_FILE}"

IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"
IFS=',' read -r -a FRACTION_ARRAY <<< "${FRACTIONS}"

for method in "${METHOD_ARRAY[@]}"; do
  method="$(echo "${method}" | xargs)"
  method_run_root="${RUN_ROOT_BASE}/${method}"
  base_cfg="${TMP_DIR}/base_${method}.yaml"
  make_base_cfg "${method}" "${method_run_root}" "${base_cfg}"

  case "${method}" in
    scratch)
      pretrained_encoder="none"
      anomaly_loss_weight="0.05"
      ;;
    reconst)
      run_pretrain_if_needed "reconst" "${method_run_root}" "${base_cfg}" "${LOG_FILE}"
      pretrained_encoder="${method_run_root}/pretrain/pohang/best_encoder.pt"
      anomaly_loss_weight="0.05"
      ;;
    contrast)
      run_pretrain_if_needed "contrast" "${method_run_root}" "${base_cfg}" "${LOG_FILE}"
      pretrained_encoder="${method_run_root}/pretrain/pohang/best_encoder.pt"
      anomaly_loss_weight="0.05"
      ;;
    reconst_noanom)
      pretrained_encoder="${RUN_ROOT_BASE}/reconst/pretrain/pohang/best_encoder.pt"
      if [[ ! -f "${pretrained_encoder}" ]]; then
        reconst_base_cfg="${TMP_DIR}/base_reconst.yaml"
        make_base_cfg "reconst" "${RUN_ROOT_BASE}/reconst" "${reconst_base_cfg}"
        run_pretrain_if_needed "reconst" "${RUN_ROOT_BASE}/reconst" "${reconst_base_cfg}" "${LOG_FILE}"
      fi
      anomaly_loss_weight="0.0"
      ;;
    *)
      echo "[ERROR] unsupported method: ${method}" | tee -a "${LOG_FILE}"
      exit 1
      ;;
  esac

  for fraction in "${FRACTION_ARRAY[@]}"; do
    fraction="$(echo "${fraction}" | xargs)"
    tag="frac$(frac_tag "${fraction}")"
    suffix="${tag}"
    stage_cfg="${TMP_DIR}/finetune_${method}_${tag}.yaml"
    eval_base_cfg="${TMP_DIR}/eval_base_${method}_${tag}.yaml"
    run_log="${LOG_ROOT}/pohang_main_study__${method}__${tag}.log"

    make_finetune_stage_cfg \
      "${method}" \
      "${method_run_root}" \
      "${fraction}" \
      "${pretrained_encoder}" \
      "${anomaly_loss_weight}" \
      "${stage_cfg}"

    make_eval_base_cfg "${base_cfg}" "${suffix}" "${eval_base_cfg}"
    finetune_dir="${method_run_root}/finetune/pohang__${suffix}"

    {
      echo "============================================================"
      echo "[POHANG STUDY RUN] $(date '+%F %T') method=${method} fraction=${fraction} gpu=${GPU}"
      echo "base_cfg=${base_cfg}"
      echo "stage_cfg=${stage_cfg}"
      echo "eval_base_cfg=${eval_base_cfg}"
      echo "pretrained_encoder=${pretrained_encoder}"
      echo "anomaly_loss_weight=${anomaly_loss_weight}"
      echo "finetune_dir=${finetune_dir}"
      echo "============================================================"
    } | tee -a "${LOG_FILE}" | tee -a "${run_log}"

    echo "[POHANG STUDY] finetune" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
      --base_cfg "${base_cfg}" \
      --stage_cfg "${stage_cfg}" \
      --exp_suffix "${suffix}" >> "${run_log}" 2>&1

    run_eval_steps "${GPU}" "${eval_base_cfg}" "${finetune_dir}" "${run_log}"
    echo "[DONE] method=${method} fraction=${fraction}" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
  done
done

"${PYTHON_BIN}" scripts/gpu/summarize_pohang_main_study.py \
  --root "${RUN_ROOT_BASE}" \
  --out "${RUN_ROOT_BASE}/summary.csv" >> "${LOG_FILE}" 2>&1

echo "[DONE] Pohang main study completed" | tee -a "${LOG_FILE}"
echo "[INFO] summary: ${RUN_ROOT_BASE}/summary.csv" | tee -a "${LOG_FILE}"
