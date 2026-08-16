#!/usr/bin/env bash
set -euo pipefail

SITE="${1:-utah_2023}"
GPU="${2:-0}"
LOG_ROOT="${3:-logs}"
RUN_ROOT_BASE="${4:-runs/${SITE}_main_study}"
ACT_VARIANT="${SITE_STUDY_ACT:-relu}"

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

case "${SITE}:${ACT_VARIANT}" in
  pohang:relu)
    BASE_TEMPLATE="configs/train/base_pohang_arch_best.yaml"
    FINETUNE_TEMPLATE="configs/train/final_pohang_best.yaml"
    EXPERIMENT="pohang"
    DEFAULT_ANOMALY="0.05"
    ;;
  pohang:silu)
    BASE_TEMPLATE="configs/train/base_pohang_arch_best_silu.yaml"
    FINETUNE_TEMPLATE="configs/train/final_pohang_best_silu.yaml"
    EXPERIMENT="pohang"
    DEFAULT_ANOMALY="0.05"
    ;;
  utah_2019:relu)
    BASE_TEMPLATE="configs/train/base_utah_2019_arch_best.yaml"
    FINETUNE_TEMPLATE="configs/train/final_utah_2019_best.yaml"
    EXPERIMENT="base_utah_2019"
    DEFAULT_ANOMALY="0.01"
    ;;
  utah_2019:silu)
    BASE_TEMPLATE="configs/train/base_utah_2019_arch_best_silu.yaml"
    FINETUNE_TEMPLATE="configs/train/final_utah_2019_best_silu.yaml"
    EXPERIMENT="base_utah_2019"
    DEFAULT_ANOMALY="0.01"
    ;;
  utah_2023:relu)
    BASE_TEMPLATE="configs/train/base_utah_2023_arch_best.yaml"
    FINETUNE_TEMPLATE="configs/train/final_utah_2023_best.yaml"
    EXPERIMENT="base_utah_2023"
    DEFAULT_ANOMALY="0.3"
    ;;
  utah_2023:silu)
    BASE_TEMPLATE="configs/train/base_utah_2023_arch_best_silu.yaml"
    FINETUNE_TEMPLATE="configs/train/final_utah_2023_best_silu.yaml"
    EXPERIMENT="base_utah_2023"
    DEFAULT_ANOMALY="0.3"
    ;;
  *)
    echo "[ERROR] unsupported SITE/ACT_VARIANT: ${SITE}/${ACT_VARIANT}"
    echo "[ERROR] SITE: pohang, utah_2019, utah_2023"
    echo "[ERROR] SITE_STUDY_ACT: relu, silu"
    exit 1
    ;;
esac

PRETRAIN_RECONST_TEMPLATE="configs/train/pretrain_reconst.yaml"
PRETRAIN_CONTRAST_TEMPLATE="configs/train/pretrain_contrast.yaml"
TEST_TEMPLATE="configs/train/test.yaml"
ANALYZE_TEMPLATE="configs/train/analyze.yaml"

FRACTIONS="${SITE_STUDY_FRACTIONS:-0.05,0.10,0.25,0.50,1.00}"
METHODS="${SITE_STUDY_METHODS:-scratch,reconst,contrast,reconst_noanom}"
RUN_TSNE="${SITE_STUDY_RUN_TSNE:-true}"
RUN_ANALYZE="${SITE_STUDY_RUN_ANALYZE:-true}"
RUN_PRETRAIN="${SITE_STUDY_RUN_PRETRAIN:-true}"
PREPROCESS_VARIANT="${SITE_STUDY_PREPROCESS:-base}"
NORMALIZE_OVERRIDE="${SITE_STUDY_NORMALIZE:-}"
SPLIT_DIR_OVERRIDE="${SITE_STUDY_SPLIT_DIR:-}"
AGC_WINDOW_SEC="${SITE_STUDY_AGC_WINDOW_SEC:-0.2}"
AGC_CLIP="${SITE_STUDY_AGC_CLIP:-10.0}"
TRAIN_SEED="${SITE_STUDY_SEED:-42}"
FRACTION_SEED="${SITE_STUDY_FRACTION_SEED:-${TRAIN_SEED}}"
ENCODER_POOLING="${SITE_STUDY_POOLING:-avg}"
ENCODER_POOLING_P="${SITE_STUDY_POOLING_P:-3.0}"
ENCODER_POOLING_CHANNELWISE="${SITE_STUDY_POOLING_CHANNELWISE:-true}"
LOG_CENTER_DIAGNOSTICS="${SITE_STUDY_LOG_CENTER_DIAGNOSTICS:-false}"
LOG_WASSERSTEIN_DIAGNOSTICS="${SITE_STUDY_LOG_WASSERSTEIN_DIAGNOSTICS:-false}"
WASSERSTEIN_NUM_PROJECTIONS="${SITE_STUDY_WASSERSTEIN_NUM_PROJECTIONS:-32}"
WASSERSTEIN_NUM_QUANTILES="${SITE_STUDY_WASSERSTEIN_NUM_QUANTILES:-128}"
NUM_WORKERS_OVERRIDE="${SITE_STUDY_NUM_WORKERS:-}"
PRETRAIN_EPOCHS_OVERRIDE="${SITE_STUDY_PRETRAIN_EPOCHS:-}"
PRETRAIN_BATCH_SIZE_OVERRIDE="${SITE_STUDY_PRETRAIN_BATCH_SIZE:-}"
DATA_PREFETCH_FACTOR_OVERRIDE="${SITE_STUDY_PREFETCH_FACTOR:-}"
DATA_PERSISTENT_WORKERS_OVERRIDE="${SITE_STUDY_PERSISTENT_WORKERS:-}"
DATA_CACHE_MODE_OVERRIDE="${SITE_STUDY_CACHE_MODE:-}"

apply_stage_overrides_py='
import sys
import yaml

def apply_overrides(cfg, preprocess_variant, normalize_override, agc_window_sec, agc_clip):
    data = cfg.setdefault("data", {})
    preprocess = data.setdefault("preprocess", {})

    variant = str(preprocess_variant or "base").strip().lower()
    if normalize_override:
        data["normalize"] = str(normalize_override).strip().lower()

    if variant not in {"base", "bandpass", "agc", "bandpass_agc", "raw", "load_only"}:
        raise ValueError(
            f"Unsupported SITE_STUDY_PREPROCESS={preprocess_variant!r}. "
            "Use base, bandpass, agc, bandpass_agc, raw, or load_only."
        )

    if variant != "base":
        preprocess["load_only"] = variant == "load_only"
        preprocess["detrend"] = variant not in {"raw", "load_only"}
        preprocess["bandpass"] = variant in {"bandpass", "bandpass_agc"}
        preprocess["agc"] = variant in {"agc", "bandpass_agc"}
        preprocess["agc_window_sec"] = float(agc_window_sec)
        preprocess["agc_clip"] = float(agc_clip) if str(agc_clip).strip().lower() not in {"", "none", "null"} else None
        if variant in {"raw", "load_only"} and not normalize_override:
            data["normalize"] = "none"

    data["preprocess_variant"] = variant
    return cfg
'

RUN_ROOT_SLUG="$(printf '%s' "${RUN_ROOT_BASE}" | sed 's#[/ ]#_#g' | sed 's#[^A-Za-z0-9._-]#_#g')"
LOG_SLUG_RAW="${SITE_STUDY_LOG_SLUG:-${RUN_ROOT_SLUG}}"
LOG_SLUG="$(printf '%s' "${LOG_SLUG_RAW}" | sed 's#[/ ]#_#g' | sed 's#[^A-Za-z0-9._-]#_#g')"
TMP_DIR=".tmp_${SITE}_main_study_cfg_${LOG_SLUG}"
mkdir -p "${LOG_ROOT}" "${TMP_DIR}" "${RUN_ROOT_BASE}"

LOG_FILE="${LOG_ROOT}/${SITE}_main_study__${LOG_SLUG}.log"

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
  "${PYTHON_BIN}" - "${BASE_TEMPLATE}" "${run_root}" "${out_cfg}" "${PREPROCESS_VARIANT}" "${NORMALIZE_OVERRIDE}" "${AGC_WINDOW_SEC}" "${AGC_CLIP}" "${TRAIN_SEED}" "${ENCODER_POOLING}" "${ENCODER_POOLING_P}" "${ENCODER_POOLING_CHANNELWISE}" "${SPLIT_DIR_OVERRIDE}" "${NUM_WORKERS_OVERRIDE}" "${DATA_CACHE_MODE_OVERRIDE}" <<'PY'
import sys
import yaml
def apply_overrides(cfg, preprocess_variant, normalize_override, agc_window_sec, agc_clip):
    data = cfg.setdefault("data", {})
    preprocess = data.setdefault("preprocess", {})

    variant = str(preprocess_variant or "base").strip().lower()
    if normalize_override:
        data["normalize"] = str(normalize_override).strip().lower()

    if variant not in {"base", "bandpass", "agc", "bandpass_agc", "raw", "load_only"}:
        raise ValueError(
            f"Unsupported SITE_STUDY_PREPROCESS={preprocess_variant!r}. "
            "Use base, bandpass, agc, bandpass_agc, raw, or load_only."
        )

    if variant != "base":
        preprocess["load_only"] = variant == "load_only"
        preprocess["detrend"] = variant not in {"raw", "load_only"}
        preprocess["bandpass"] = variant in {"bandpass", "bandpass_agc"}
        preprocess["agc"] = variant in {"agc", "bandpass_agc"}
        preprocess["agc_window_sec"] = float(agc_window_sec)
        preprocess["agc_clip"] = float(agc_clip) if str(agc_clip).strip().lower() not in {"", "none", "null"} else None
        if variant in {"raw", "load_only"} and not normalize_override:
            data["normalize"] = "none"

    data["preprocess_variant"] = variant
    return cfg

src, run_root, outp, preprocess_variant, normalize_override, agc_window_sec, agc_clip, train_seed, pooling, pooling_p, pooling_channelwise, split_dir_override, num_workers_override, cache_mode_override = sys.argv[1:15]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
cfg["seed"] = int(train_seed)
if split_dir_override:
    cfg.setdefault("data", {})["split_dir"] = split_dir_override
if str(num_workers_override).strip() != "":
    cfg.setdefault("data", {})["num_workers"] = int(num_workers_override)
if str(cache_mode_override).strip() != "":
    cfg.setdefault("data", {})["cache_mode"] = str(cache_mode_override).strip().lower()
cfg = apply_overrides(cfg, preprocess_variant, normalize_override, agc_window_sec, agc_clip)
enc = cfg.setdefault("model", {}).setdefault("encoder", {})
pooling = str(pooling or "avg").strip().lower()
if pooling not in {"avg", "average", "gap", "adaptive_avg", "gem", "signed_gem", "generalized_mean"}:
    raise ValueError(f"Unsupported SITE_STUDY_POOLING={pooling!r}")
enc["pooling"] = pooling
enc["pooling_p"] = float(pooling_p)
enc["pooling_channelwise"] = str(pooling_channelwise).strip().lower() in {"1", "true", "yes", "y"}
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
  "${PYTHON_BIN}" - "${template}" "${run_root}" "${out_cfg}" "${PREPROCESS_VARIANT}" "${NORMALIZE_OVERRIDE}" "${AGC_WINDOW_SEC}" "${AGC_CLIP}" "${TRAIN_SEED}" "${NUM_WORKERS_OVERRIDE}" "${PRETRAIN_EPOCHS_OVERRIDE}" "${PRETRAIN_BATCH_SIZE_OVERRIDE}" "${DATA_PREFETCH_FACTOR_OVERRIDE}" "${DATA_PERSISTENT_WORKERS_OVERRIDE}" "${DATA_CACHE_MODE_OVERRIDE}" <<'PY'
import sys
import yaml
def apply_overrides(cfg, preprocess_variant, normalize_override, agc_window_sec, agc_clip):
    data = cfg.setdefault("data", {})
    preprocess = data.setdefault("preprocess", {})

    variant = str(preprocess_variant or "base").strip().lower()
    if normalize_override:
        data["normalize"] = str(normalize_override).strip().lower()

    if variant not in {"base", "bandpass", "agc", "bandpass_agc", "raw", "load_only"}:
        raise ValueError(
            f"Unsupported SITE_STUDY_PREPROCESS={preprocess_variant!r}. "
            "Use base, bandpass, agc, bandpass_agc, raw, or load_only."
        )

    if variant != "base":
        preprocess["load_only"] = variant == "load_only"
        preprocess["detrend"] = variant not in {"raw", "load_only"}
        preprocess["bandpass"] = variant in {"bandpass", "bandpass_agc"}
        preprocess["agc"] = variant in {"agc", "bandpass_agc"}
        preprocess["agc_window_sec"] = float(agc_window_sec)
        preprocess["agc_clip"] = float(agc_clip) if str(agc_clip).strip().lower() not in {"", "none", "null"} else None
        if variant in {"raw", "load_only"} and not normalize_override:
            data["normalize"] = "none"

    data["preprocess_variant"] = variant
    return cfg

src, run_root, outp, preprocess_variant, normalize_override, agc_window_sec, agc_clip, train_seed, num_workers_override, pretrain_epochs_override, pretrain_batch_size_override, prefetch_factor_override, persistent_workers_override, cache_mode_override = sys.argv[1:15]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
cfg["seed"] = int(train_seed)
cfg.setdefault("pretrain", {})["epochs"] = int(cfg["pretrain"].get("epochs", 150))
if str(pretrain_epochs_override).strip() != "":
    cfg["pretrain"]["epochs"] = int(pretrain_epochs_override)
if str(pretrain_batch_size_override).strip() != "":
    cfg["pretrain"]["batch_size"] = int(pretrain_batch_size_override)
cfg["pretrain"]["use_amp"] = True
if str(num_workers_override).strip() != "":
    cfg.setdefault("data", {})["num_workers"] = int(num_workers_override)
if str(cache_mode_override).strip() != "":
    cfg.setdefault("data", {})["cache_mode"] = str(cache_mode_override).strip().lower()
if str(prefetch_factor_override).strip() != "":
    cfg.setdefault("data", {})["prefetch_factor"] = int(prefetch_factor_override)
if str(persistent_workers_override).strip() != "":
    value = str(persistent_workers_override).strip().lower()
    cfg.setdefault("data", {})["persistent_workers"] = value in {"1", "true", "yes", "y", "on"}
cfg = apply_overrides(cfg, preprocess_variant, normalize_override, agc_window_sec, agc_clip)
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_finetune_stage_cfg() {
  local run_root="$1"
  local fraction="$2"
  local pretrained_encoder="$3"
  local anomaly_loss_weight="$4"
  local out_cfg="$5"
  "${PYTHON_BIN}" - "${FINETUNE_TEMPLATE}" "${run_root}" "${fraction}" "${pretrained_encoder}" "${anomaly_loss_weight}" "${out_cfg}" "${PREPROCESS_VARIANT}" "${NORMALIZE_OVERRIDE}" "${AGC_WINDOW_SEC}" "${AGC_CLIP}" "${TRAIN_SEED}" "${FRACTION_SEED}" "${LOG_CENTER_DIAGNOSTICS}" "${LOG_WASSERSTEIN_DIAGNOSTICS}" "${WASSERSTEIN_NUM_PROJECTIONS}" "${WASSERSTEIN_NUM_QUANTILES}" "${NUM_WORKERS_OVERRIDE}" "${DATA_CACHE_MODE_OVERRIDE}" <<'PY'
import sys
import yaml
def apply_overrides(cfg, preprocess_variant, normalize_override, agc_window_sec, agc_clip):
    data = cfg.setdefault("data", {})
    preprocess = data.setdefault("preprocess", {})

    variant = str(preprocess_variant or "base").strip().lower()
    if normalize_override:
        data["normalize"] = str(normalize_override).strip().lower()

    if variant not in {"base", "bandpass", "agc", "bandpass_agc", "raw", "load_only"}:
        raise ValueError(
            f"Unsupported SITE_STUDY_PREPROCESS={preprocess_variant!r}. "
            "Use base, bandpass, agc, bandpass_agc, raw, or load_only."
        )

    if variant != "base":
        preprocess["load_only"] = variant == "load_only"
        preprocess["detrend"] = variant not in {"raw", "load_only"}
        preprocess["bandpass"] = variant in {"bandpass", "bandpass_agc"}
        preprocess["agc"] = variant in {"agc", "bandpass_agc"}
        preprocess["agc_window_sec"] = float(agc_window_sec)
        preprocess["agc_clip"] = float(agc_clip) if str(agc_clip).strip().lower() not in {"", "none", "null"} else None
        if variant in {"raw", "load_only"} and not normalize_override:
            data["normalize"] = "none"

    data["preprocess_variant"] = variant
    return cfg

def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

src, run_root, fraction, pretrained_encoder, anomaly_loss_weight, outp, preprocess_variant, normalize_override, agc_window_sec, agc_clip, train_seed, fraction_seed, log_center, log_wasserstein, wasserstein_num_projections, wasserstein_num_quantiles, num_workers_override, cache_mode_override = sys.argv[1:19]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
train = cfg.setdefault("train", {})
cfg.setdefault("paths", {})["run_root"] = run_root
cfg["seed"] = int(train_seed)
if str(num_workers_override).strip() != "":
    cfg.setdefault("data", {})["num_workers"] = int(num_workers_override)
if str(cache_mode_override).strip() != "":
    cfg.setdefault("data", {})["cache_mode"] = str(cache_mode_override).strip().lower()
fraction = float(fraction)
train["use_labeled_fraction"] = fraction < 1.0
train["labeled_fraction"] = fraction
train["seed"] = int(train_seed)
train["fraction_seed"] = int(fraction_seed)
train["balance_fraction_by_class"] = True
train["min_samples_per_class"] = 1
train["drop_last"] = False
train["anomaly_loss_weight"] = float(anomaly_loss_weight)
train["log_center_diagnostics"] = as_bool(log_center) or as_bool(log_wasserstein)
train["log_wasserstein_diagnostics"] = as_bool(log_wasserstein)
train["wasserstein_num_projections"] = int(wasserstein_num_projections)
train["wasserstein_num_quantiles"] = int(wasserstein_num_quantiles)
if pretrained_encoder == "none":
    train["use_pretrained_encoder"] = False
    train["pretrained_encoder_path"] = None
else:
    train["use_pretrained_encoder"] = True
    train["pretrained_encoder_path"] = pretrained_encoder
cfg = apply_overrides(cfg, preprocess_variant, normalize_override, agc_window_sec, agc_clip)
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_test_stage_cfg() {
  local out_cfg="$1"
  "${PYTHON_BIN}" - "${TEST_TEMPLATE}" "${out_cfg}" "${PREPROCESS_VARIANT}" "${NORMALIZE_OVERRIDE}" "${AGC_WINDOW_SEC}" "${AGC_CLIP}" "${NUM_WORKERS_OVERRIDE}" "${DATA_CACHE_MODE_OVERRIDE}" <<'PY'
import sys
import yaml

def apply_overrides(cfg, preprocess_variant, normalize_override, agc_window_sec, agc_clip):
    data = cfg.setdefault("data", {})
    preprocess = data.setdefault("preprocess", {})

    variant = str(preprocess_variant or "base").strip().lower()
    if normalize_override:
        data["normalize"] = str(normalize_override).strip().lower()

    if variant not in {"base", "bandpass", "agc", "bandpass_agc", "raw", "load_only"}:
        raise ValueError(
            f"Unsupported SITE_STUDY_PREPROCESS={preprocess_variant!r}. "
            "Use base, bandpass, agc, bandpass_agc, raw, or load_only."
        )

    if variant != "base":
        preprocess["load_only"] = variant == "load_only"
        preprocess["detrend"] = variant not in {"raw", "load_only"}
        preprocess["bandpass"] = variant in {"bandpass", "bandpass_agc"}
        preprocess["agc"] = variant in {"agc", "bandpass_agc"}
        preprocess["agc_window_sec"] = float(agc_window_sec)
        preprocess["agc_clip"] = float(agc_clip) if str(agc_clip).strip().lower() not in {"", "none", "null"} else None
        if variant in {"raw", "load_only"} and not normalize_override:
            data["normalize"] = "none"

    data["preprocess_variant"] = variant
    return cfg

src, outp, preprocess_variant, normalize_override, agc_window_sec, agc_clip, num_workers_override, cache_mode_override = sys.argv[1:9]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
if str(num_workers_override).strip() != "":
    cfg.setdefault("data", {})["num_workers"] = int(num_workers_override)
    cfg.setdefault("test", {})["num_workers"] = int(num_workers_override)
if str(cache_mode_override).strip() != "":
    cfg.setdefault("data", {})["cache_mode"] = str(cache_mode_override).strip().lower()
cfg = apply_overrides(cfg, preprocess_variant, normalize_override, agc_window_sec, agc_clip)
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

run_pretrain_if_needed() {
  local method="$1"
  local run_root="$2"
  local base_cfg="$3"
  local log_file="$4"
  local stage_cfg

  if [[ "${RUN_PRETRAIN}" != "true" ]]; then
    echo "[SKIP] pretrain disabled by SITE_STUDY_RUN_PRETRAIN=${RUN_PRETRAIN}" | tee -a "${log_file}"
    return
  fi

  case "${method}" in
    reconst)
      stage_cfg="${TMP_DIR}/pretrain_${method}.yaml"
      make_pretrain_stage_cfg "${PRETRAIN_RECONST_TEMPLATE}" "${run_root}" "${stage_cfg}"
      ;;
    contrast)
      stage_cfg="${TMP_DIR}/pretrain_${method}.yaml"
      make_pretrain_stage_cfg "${PRETRAIN_CONTRAST_TEMPLATE}" "${run_root}" "${stage_cfg}"
      ;;
    *)
      return
      ;;
  esac

  local ckpt="${run_root}/pretrain/${EXPERIMENT}/best_encoder.pt"
  if [[ -f "${ckpt}" ]]; then
    echo "[SKIP] existing pretrain checkpoint: ${ckpt}" | tee -a "${log_file}"
    return
  fi

  echo "[${SITE} STUDY] pretrain method=${method}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_pretrain \
    --base_cfg "${base_cfg}" \
    --stage_cfg "${stage_cfg}" >> "${log_file}" 2>&1
}

run_eval_steps() {
  local eval_base_cfg="$1"
  local finetune_dir="$2"
  local log_file="$3"
  local test_stage_cfg="$4"

  echo "[${SITE} STUDY] test" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
    --base_cfg "${eval_base_cfg}" \
    --stage_cfg "${test_stage_cfg}" >> "${log_file}" 2>&1

  if [[ "${RUN_ANALYZE}" == "true" ]]; then
    echo "[${SITE} STUDY] analyze" | tee -a "${log_file}"
    "${PYTHON_BIN}" -m src.detection.analysis.analyze \
      --base_cfg "${eval_base_cfg}" \
      --stage_cfg "${ANALYZE_TEMPLATE}" >> "${log_file}" 2>&1
  fi

  if [[ "${RUN_TSNE}" == "true" ]]; then
    echo "[${SITE} STUDY] tsne" | tee -a "${log_file}"
    PYTHON_BIN="${PYTHON_BIN}" bash scripts/gpu/run_tsne.sh "${GPU}" "${finetune_dir}" >> "${log_file}" 2>&1
  fi
}

{
  echo "============================================================"
  echo "[${SITE} MAIN STUDY START] $(date '+%F %T')"
  echo "python_bin=${PYTHON_BIN}"
  echo "gpu=${GPU}"
  echo "site=${SITE}"
  echo "act_variant=${ACT_VARIANT}"
  echo "experiment=${EXPERIMENT}"
  echo "base_template=${BASE_TEMPLATE}"
  echo "finetune_template=${FINETUNE_TEMPLATE}"
  echo "run_root_base=${RUN_ROOT_BASE}"
  echo "methods=${METHODS}"
  echo "fractions=${FRACTIONS}"
  echo "run_pretrain=${RUN_PRETRAIN}"
  echo "run_analyze=${RUN_ANALYZE}"
  echo "run_tsne=${RUN_TSNE}"
  echo "preprocess_variant=${PREPROCESS_VARIANT}"
  echo "normalize_override=${NORMALIZE_OVERRIDE:-<base>}"
  echo "split_dir_override=${SPLIT_DIR_OVERRIDE:-<base>}"
  echo "agc_window_sec=${AGC_WINDOW_SEC}"
  echo "agc_clip=${AGC_CLIP}"
  echo "train_seed=${TRAIN_SEED}"
  echo "fraction_seed=${FRACTION_SEED}"
  echo "log_center_diagnostics=${LOG_CENTER_DIAGNOSTICS}"
  echo "log_wasserstein_diagnostics=${LOG_WASSERSTEIN_DIAGNOSTICS}"
  echo "wasserstein_num_projections=${WASSERSTEIN_NUM_PROJECTIONS}"
  echo "wasserstein_num_quantiles=${WASSERSTEIN_NUM_QUANTILES}"
  echo "num_workers_override=${NUM_WORKERS_OVERRIDE:-<base>}"
  echo "pretrain_epochs_override=${PRETRAIN_EPOCHS_OVERRIDE:-<base>}"
  echo "pretrain_batch_size_override=${PRETRAIN_BATCH_SIZE_OVERRIDE:-<base>}"
  echo "prefetch_factor_override=${DATA_PREFETCH_FACTOR_OVERRIDE:-<base>}"
  echo "persistent_workers_override=${DATA_PERSISTENT_WORKERS_OVERRIDE:-<base>}"
  echo "cache_mode_override=${DATA_CACHE_MODE_OVERRIDE:-<base>}"
  echo "============================================================"
} | tee -a "${LOG_FILE}"

IFS=',' read -r -a METHOD_ARRAY <<< "${METHODS}"
IFS=',' read -r -a FRACTION_ARRAY <<< "${FRACTIONS}"

for method in "${METHOD_ARRAY[@]}"; do
  method="$(echo "${method}" | xargs)"
  method_run_root="${RUN_ROOT_BASE}/${method}"
  base_cfg="${TMP_DIR}/base_${method}.yaml"
  make_base_cfg "${method_run_root}" "${base_cfg}"

  case "${method}" in
    scratch)
      pretrained_encoder="none"
      anomaly_loss_weight="${DEFAULT_ANOMALY}"
      ;;
    reconst)
      run_pretrain_if_needed "reconst" "${method_run_root}" "${base_cfg}" "${LOG_FILE}"
      pretrained_encoder="${method_run_root}/pretrain/${EXPERIMENT}/best_encoder.pt"
      anomaly_loss_weight="${DEFAULT_ANOMALY}"
      ;;
    contrast)
      run_pretrain_if_needed "contrast" "${method_run_root}" "${base_cfg}" "${LOG_FILE}"
      pretrained_encoder="${method_run_root}/pretrain/${EXPERIMENT}/best_encoder.pt"
      anomaly_loss_weight="${DEFAULT_ANOMALY}"
      ;;
    reconst_noanom)
      pretrained_encoder="${RUN_ROOT_BASE}/reconst/pretrain/${EXPERIMENT}/best_encoder.pt"
      if [[ ! -f "${pretrained_encoder}" ]]; then
        reconst_base_cfg="${TMP_DIR}/base_reconst.yaml"
        make_base_cfg "${RUN_ROOT_BASE}/reconst" "${reconst_base_cfg}"
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
    test_stage_cfg="${TMP_DIR}/test_${method}_${tag}.yaml"
    run_log="${LOG_ROOT}/${SITE}_main_study__${LOG_SLUG}__${method}__${tag}.log"

    make_finetune_stage_cfg \
      "${method_run_root}" \
      "${fraction}" \
      "${pretrained_encoder}" \
      "${anomaly_loss_weight}" \
      "${stage_cfg}"

    make_eval_base_cfg "${base_cfg}" "${suffix}" "${eval_base_cfg}"
    make_test_stage_cfg "${test_stage_cfg}"
    finetune_dir="${method_run_root}/finetune/${EXPERIMENT}__${suffix}"

    {
      echo "============================================================"
      echo "[${SITE} STUDY RUN] $(date '+%F %T') method=${method} fraction=${fraction} gpu=${GPU}"
      echo "base_cfg=${base_cfg}"
	      echo "stage_cfg=${stage_cfg}"
	      echo "eval_base_cfg=${eval_base_cfg}"
	      echo "pooling=${ENCODER_POOLING}"
	      echo "pretrained_encoder=${pretrained_encoder}"
      echo "anomaly_loss_weight=${anomaly_loss_weight}"
      echo "finetune_dir=${finetune_dir}"
      echo "============================================================"
    } | tee -a "${LOG_FILE}" | tee -a "${run_log}"

    echo "[${SITE} STUDY] finetune" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
      --base_cfg "${base_cfg}" \
      --stage_cfg "${stage_cfg}" \
      --exp_suffix "${suffix}" >> "${run_log}" 2>&1

    run_eval_steps "${eval_base_cfg}" "${finetune_dir}" "${run_log}" "${test_stage_cfg}"
    echo "[DONE] site=${SITE} method=${method} fraction=${fraction}" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
  done
done

"${PYTHON_BIN}" scripts/gpu/summarize_pohang_main_study.py \
  --root "${RUN_ROOT_BASE}" \
  --out "${RUN_ROOT_BASE}/summary.csv" >> "${LOG_FILE}" 2>&1

echo "[DONE] ${SITE} main study completed" | tee -a "${LOG_FILE}"
echo "[INFO] summary: ${RUN_ROOT_BASE}/summary.csv" | tee -a "${LOG_FILE}"
