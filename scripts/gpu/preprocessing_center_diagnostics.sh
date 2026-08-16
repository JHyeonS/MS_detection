#!/usr/bin/env bash
set -euo pipefail

SITE="${1:-pohang}"
VARIANT="${2:-baseline}"
GPU="${3:-0}"
LOG_ROOT="${4:-logs}"
RUN_ROOT_BASE="${5:-runs/preprocessing_center_diagnostics}"

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

CENTER_DIAG_FRACTIONS="${CENTER_DIAG_FRACTIONS:-0.10,0.50,1.00}"
CENTER_DIAG_UPDATE="${CENTER_DIAG_UPDATE:-every_epoch}"
CENTER_DIAG_RUN_TEST="${CENTER_DIAG_RUN_TEST:-true}"
CENTER_DIAG_LOG_WASSERSTEIN="${CENTER_DIAG_LOG_WASSERSTEIN:-true}"
CENTER_DIAG_WASSERSTEIN_PROJECTIONS="${CENTER_DIAG_WASSERSTEIN_PROJECTIONS:-32}"
CENTER_DIAG_WASSERSTEIN_QUANTILES="${CENTER_DIAG_WASSERSTEIN_QUANTILES:-128}"

TMP_DIR=".tmp_preprocessing_center_diag_cfg"
mkdir -p "${LOG_ROOT}" "${TMP_DIR}" "${RUN_ROOT_BASE}"
LOG_FILE="${LOG_ROOT}/preprocessing_center_diagnostics__${SITE}__${VARIANT}.log"

resolve_site_templates() {
  case "${SITE}" in
    pohang)
      BASE_TEMPLATE="configs/train/base_pohang_arch_best.yaml"
      FINETUNE_TEMPLATE="configs/train/final_pohang_best.yaml"
      SITE_EXPERIMENT="pohang"
      ;;
    utah_2019)
      BASE_TEMPLATE="configs/train/base_utah_2019_arch_best.yaml"
      FINETUNE_TEMPLATE="configs/train/final_utah_2019_best.yaml"
      SITE_EXPERIMENT="base_utah_2019"
      ;;
    utah_2023)
      BASE_TEMPLATE="configs/train/base_utah_2023_arch_best.yaml"
      FINETUNE_TEMPLATE="configs/train/final_utah_2023_best.yaml"
      SITE_EXPERIMENT="base_utah_2023"
      ;;
    *)
      echo "[ERROR] unsupported site: ${SITE}" | tee -a "${LOG_FILE}"
      exit 1
      ;;
  esac
}

resolve_variant_settings() {
  case "${SITE}:${VARIANT}" in
    pohang:baseline)
      PRETRAIN_ENCODER="runs/pohang_main_study/reconst/pretrain/pohang/best_encoder.pt"
      DATA_NORMALIZE="robust"
      PREP_BANDPASS="true"
      PREP_AGC="false"
      ;;
    pohang:bandpass_agc_none)
      PRETRAIN_ENCODER="runs/pohang_normalization_ablation_v2/bandpass_agc_none/reconst/pretrain/pohang/best_encoder.pt"
      DATA_NORMALIZE="none"
      PREP_BANDPASS="true"
      PREP_AGC="true"
      ;;
    pohang:bandpass_agc_robust)
      PRETRAIN_ENCODER="runs/pohang_normalization_ablation_v2/bandpass_agc_robust/reconst/pretrain/pohang/best_encoder.pt"
      DATA_NORMALIZE="robust"
      PREP_BANDPASS="true"
      PREP_AGC="true"
      ;;
    utah_2019:baseline)
      PRETRAIN_ENCODER="runs/utah_2019_main_study/reconst/pretrain/base_utah_2019/best_encoder.pt"
      DATA_NORMALIZE="robust"
      PREP_BANDPASS="true"
      PREP_AGC="false"
      ;;
    utah_2019:bandpass_agc_none)
      PRETRAIN_ENCODER="runs/utah_2019_normalization_ablation_v2/bandpass_agc_none/reconst/pretrain/base_utah_2019/best_encoder.pt"
      DATA_NORMALIZE="none"
      PREP_BANDPASS="true"
      PREP_AGC="true"
      ;;
    utah_2019:bandpass_agc_robust)
      PRETRAIN_ENCODER="runs/utah_2019_normalization_ablation_v2/bandpass_agc_robust/reconst/pretrain/base_utah_2019/best_encoder.pt"
      DATA_NORMALIZE="robust"
      PREP_BANDPASS="true"
      PREP_AGC="true"
      ;;
    utah_2023:baseline)
      PRETRAIN_ENCODER="runs/utah_2023_main_study/reconst/pretrain/base_utah_2023/best_encoder.pt"
      DATA_NORMALIZE="robust"
      PREP_BANDPASS="true"
      PREP_AGC="false"
      ;;
    utah_2023:bandpass_agc_none)
      PRETRAIN_ENCODER="runs/utah_2023_normalization_ablation_v2/bandpass_agc_none/reconst/pretrain/base_utah_2023/best_encoder.pt"
      DATA_NORMALIZE="none"
      PREP_BANDPASS="true"
      PREP_AGC="true"
      ;;
    utah_2023:bandpass_agc_robust)
      PRETRAIN_ENCODER="runs/utah_2023_normalization_ablation_v2/bandpass_agc_robust/reconst/pretrain/base_utah_2023/best_encoder.pt"
      DATA_NORMALIZE="robust"
      PREP_BANDPASS="true"
      PREP_AGC="true"
      ;;
    *)
      echo "[ERROR] unsupported site/variant: ${SITE}/${VARIANT}" | tee -a "${LOG_FILE}"
      exit 1
      ;;
  esac
}

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
  "${PYTHON_BIN}" - "${BASE_TEMPLATE}" "${run_root}" "${SITE_EXPERIMENT}" "${DATA_NORMALIZE}" "${PREP_BANDPASS}" "${PREP_AGC}" "${out_cfg}" <<'PY'
import sys, yaml

src, run_root, experiment, normalize, bandpass, agc, outp = sys.argv[1:8]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
data = cfg.setdefault("data", {})
data["experiment"] = experiment
data["normalize"] = normalize
prep = data.setdefault("preprocess", {})
prep["detrend"] = True
prep["bandpass"] = bandpass.lower() == "true"
prep["bandpass_low"] = float(prep.get("bandpass_low", 3))
prep["bandpass_high"] = float(prep.get("bandpass_high", 50))
prep["bandpass_order"] = int(prep.get("bandpass_order", 4))
prep["agc"] = agc.lower() == "true"
prep["agc_window_sec"] = float(prep.get("agc_window_sec", 0.2))
prep["agc_target_rms"] = float(prep.get("agc_target_rms", 1.0))
prep["agc_clip"] = float(prep.get("agc_clip", 10.0))
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
  local out_cfg="$3"
  "${PYTHON_BIN}" - "${FINETUNE_TEMPLATE}" "${run_root}" "${fraction}" "${PRETRAIN_ENCODER}" "${CENTER_DIAG_UPDATE}" "${DATA_NORMALIZE}" "${PREP_BANDPASS}" "${PREP_AGC}" "${CENTER_DIAG_LOG_WASSERSTEIN}" "${CENTER_DIAG_WASSERSTEIN_PROJECTIONS}" "${CENTER_DIAG_WASSERSTEIN_QUANTILES}" "${out_cfg}" <<'PY'
import sys, yaml

src, run_root, fraction, encoder, center_update, normalize, bandpass, agc, log_wasserstein, wasserstein_projections, wasserstein_quantiles, outp = sys.argv[1:13]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
data = cfg.setdefault("data", {})
data["normalize"] = normalize
prep = data.setdefault("preprocess", {})
prep["detrend"] = True
prep["bandpass"] = bandpass.lower() == "true"
prep["bandpass_low"] = float(prep.get("bandpass_low", 3))
prep["bandpass_high"] = float(prep.get("bandpass_high", 50))
prep["bandpass_order"] = int(prep.get("bandpass_order", 4))
prep["agc"] = agc.lower() == "true"
prep["agc_window_sec"] = float(prep.get("agc_window_sec", 0.2))
prep["agc_target_rms"] = float(prep.get("agc_target_rms", 1.0))
prep["agc_clip"] = float(prep.get("agc_clip", 10.0))

train = cfg.setdefault("train", {})
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
train["log_wasserstein_diagnostics"] = log_wasserstein.lower() == "true"
train["wasserstein_num_projections"] = int(wasserstein_projections)
train["wasserstein_num_quantiles"] = int(wasserstein_quantiles)
train["center_ablation"] = f"preproc_{normalize}_{'agc' if agc.lower() == 'true' else 'noagc'}_{center_update}"
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

resolve_site_templates
resolve_variant_settings

if [[ ! -f "${PRETRAIN_ENCODER}" ]]; then
  echo "[ERROR] pretrained encoder not found: ${PRETRAIN_ENCODER}" | tee -a "${LOG_FILE}"
  exit 1
fi

RUN_ROOT="${RUN_ROOT_BASE}/${SITE}/${VARIANT}"
mkdir -p "${RUN_ROOT}"
BASE_CFG="${TMP_DIR}/${SITE}_${VARIANT}_base.yaml"
make_base_cfg "${RUN_ROOT}" "${BASE_CFG}"

{
  echo "============================================================"
  echo "[PREPROCESS CENTER START] $(date '+%F %T')"
  echo "site=${SITE}"
  echo "variant=${VARIANT}"
  echo "gpu=${GPU}"
  echo "run_root=${RUN_ROOT}"
  echo "pretrain_encoder=${PRETRAIN_ENCODER}"
  echo "fractions=${CENTER_DIAG_FRACTIONS}"
  echo "center_update=${CENTER_DIAG_UPDATE}"
  echo "log_wasserstein=${CENTER_DIAG_LOG_WASSERSTEIN}"
  echo "wasserstein_num_projections=${CENTER_DIAG_WASSERSTEIN_PROJECTIONS}"
  echo "wasserstein_num_quantiles=${CENTER_DIAG_WASSERSTEIN_QUANTILES}"
  echo "============================================================"
} | tee -a "${LOG_FILE}"

IFS=',' read -r -a FRACTION_ARRAY <<< "${CENTER_DIAG_FRACTIONS}"
for fraction in "${FRACTION_ARRAY[@]}"; do
  fraction="$(echo "${fraction}" | xargs)"
  tag="frac$(frac_tag "${fraction}")"
  STAGE_CFG="${TMP_DIR}/${SITE}_${VARIANT}_finetune_${tag}.yaml"
  EVAL_BASE_CFG="${TMP_DIR}/${SITE}_${VARIANT}_eval_base_${tag}.yaml"
  RUN_LOG="${LOG_ROOT}/preprocessing_center__${SITE}__${VARIANT}__${tag}.log"

  make_stage_cfg "${RUN_ROOT}" "${fraction}" "${STAGE_CFG}"
  make_eval_base_cfg "${BASE_CFG}" "${tag}" "${EVAL_BASE_CFG}"

  {
    echo "============================================================"
    echo "[PREPROCESS CENTER RUN] $(date '+%F %T') site=${SITE} variant=${VARIANT} fraction=${fraction} gpu=${GPU}"
    echo "stage_cfg=${STAGE_CFG}"
    echo "============================================================"
  } | tee -a "${LOG_FILE}" | tee -a "${RUN_LOG}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
    --base_cfg "${BASE_CFG}" \
    --stage_cfg "${STAGE_CFG}" \
    --exp_suffix "${tag}" >> "${RUN_LOG}" 2>&1

  if [[ "${CENTER_DIAG_RUN_TEST}" == "true" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
      --base_cfg "${EVAL_BASE_CFG}" \
      --stage_cfg "configs/train/test.yaml" >> "${RUN_LOG}" 2>&1
  fi

  echo "[DONE] site=${SITE} variant=${VARIANT} fraction=${fraction}" | tee -a "${LOG_FILE}" | tee -a "${RUN_LOG}"
done

echo "[DONE] preprocessing center diagnostics completed for ${SITE}/${VARIANT}" | tee -a "${LOG_FILE}"
