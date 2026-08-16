#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LOG_ROOT="${2:-logs}"
RUN_ROOT_BASE="${3:-runs/pair_pohang_utah_2019_study}"

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

PAIR_METHODS="${PAIR_METHODS:-scratch,reconst}"
PAIR_RUN_ANALYZE="${PAIR_RUN_ANALYZE:-true}"
PAIR_RUN_TSNE="${PAIR_RUN_TSNE:-true}"
PAIR_RUN_PRETRAIN="${PAIR_RUN_PRETRAIN:-true}"
PAIR_RUN_CROSS="${PAIR_RUN_CROSS:-true}"
PAIR_RUN_MIXED="${PAIR_RUN_MIXED:-true}"
PAIR_RUN_CROSS_P2U="${PAIR_RUN_CROSS_P2U:-true}"
PAIR_RUN_CROSS_U2P="${PAIR_RUN_CROSS_U2P:-true}"
PAIR_STUDY_SEED="${PAIR_STUDY_SEED:-42}"
PAIR_STUDY_FRACTION_SEED="${PAIR_STUDY_FRACTION_SEED:-${PAIR_STUDY_SEED}}"
PAIR_STUDY_POOLING="${PAIR_STUDY_POOLING:-avg}"
PAIR_STUDY_POOLING_P="${PAIR_STUDY_POOLING_P:-3.0}"
PAIR_STUDY_POOLING_CHANNELWISE="${PAIR_STUDY_POOLING_CHANNELWISE:-true}"

TMP_DIR=".tmp_pair_cross_mixed_cfg"
mkdir -p "${LOG_ROOT}" "${TMP_DIR}" "${RUN_ROOT_BASE}"
LOG_FILE="${LOG_ROOT}/pair_cross_and_mixed_study.log"

PRETRAIN_RECONST_TEMPLATE="configs/train/pretrain_reconst.yaml"
TEST_TEMPLATE="configs/train/test.yaml"
ANALYZE_TEMPLATE="configs/train/analyze.yaml"

MIXED_SPLIT_DIR="data/0406/metadata/experiments/stage2_joint_pohang_utah_2019"

resolve_site_templates() {
  local site="$1"
  case "${site}" in
    pohang)
      SITE_BASE_TEMPLATE="configs/train/base_pohang_arch_best.yaml"
      SITE_FINETUNE_TEMPLATE="configs/train/final_pohang_best.yaml"
      SITE_EXPERIMENT="pohang"
      ;;
    utah_2019)
      SITE_BASE_TEMPLATE="configs/train/base_utah_2019_arch_best.yaml"
      SITE_FINETUNE_TEMPLATE="configs/train/final_utah_2019_best.yaml"
      SITE_EXPERIMENT="base_utah_2019"
      ;;
    *)
      echo "[ERROR] unsupported site template: ${site}"
      exit 1
      ;;
  esac
}

make_pair_mixed_split() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  "${PYTHON_BIN}" - "${out_dir}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

out_dir = Path(sys.argv[1])
src_dir = Path("data/0406/metadata/experiments/stage2_joint_all")
sites = {"pohang", "utah_2019"}

summary = {
    "experiment": "stage2_joint_pohang_utah_2019",
    "derived_from": str(src_dir),
    "sites": sorted(sites),
}

for split in ["pretrain", "train", "val", "test"]:
    df = pd.read_csv(src_dir / f"{split}.csv")
    df = df[df["site"].isin(sites)].reset_index(drop=True)
    df.to_csv(out_dir / f"{split}.csv", index=False)

    summary[split] = {
        "n_rows": int(len(df)),
        "n_groups": int(df["group_id"].nunique()) if "group_id" in df.columns else None,
        "site_counts": {str(k): int(v) for k, v in df["site"].value_counts().to_dict().items()},
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()} if "label" in df.columns else {},
        "label_name_counts": {str(k): int(v) for k, v in df["label_name"].value_counts().to_dict().items()} if "label_name" in df.columns else {},
    }

with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
PY
}

make_base_cfg() {
  local src_template="$1"
  local run_root="$2"
  local split_dir="$3"
  local experiment_name="$4"
  local out_cfg="$5"
  "${PYTHON_BIN}" - "${src_template}" "${run_root}" "${split_dir}" "${experiment_name}" "${out_cfg}" "${PAIR_STUDY_SEED}" "${PAIR_STUDY_POOLING}" "${PAIR_STUDY_POOLING_P}" "${PAIR_STUDY_POOLING_CHANNELWISE}" <<'PY'
import sys
import yaml

src, run_root, split_dir, experiment_name, outp, train_seed, pooling, pooling_p, pooling_channelwise = sys.argv[1:10]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
cfg.setdefault("data", {})["split_dir"] = split_dir
cfg["data"]["experiment"] = experiment_name
cfg["seed"] = int(train_seed)
enc = cfg.setdefault("model", {}).setdefault("encoder", {})
pooling = str(pooling or "avg").strip().lower()
if pooling not in {"avg", "average", "gap", "adaptive_avg", "gem", "signed_gem", "generalized_mean"}:
    raise ValueError(f"Unsupported PAIR_STUDY_POOLING={pooling!r}")
enc["pooling"] = pooling
enc["pooling_p"] = float(pooling_p)
enc["pooling_channelwise"] = str(pooling_channelwise).strip().lower() in {"1", "true", "yes", "y"}
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_pretrain_stage_cfg() {
  local run_root="$1"
  local out_cfg="$2"
  "${PYTHON_BIN}" - "${PRETRAIN_RECONST_TEMPLATE}" "${run_root}" "${out_cfg}" "${PAIR_STUDY_SEED}" <<'PY'
import sys
import yaml

src, run_root, outp, train_seed = sys.argv[1:5]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
cfg["seed"] = int(train_seed)
cfg.setdefault("pretrain", {})["epochs"] = int(cfg["pretrain"].get("epochs", 150))
cfg["pretrain"]["use_amp"] = True
with open(outp, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

make_finetune_stage_cfg() {
  local template="$1"
  local run_root="$2"
  local pretrained_encoder="$3"
  local out_cfg="$4"
  "${PYTHON_BIN}" - "${template}" "${run_root}" "${pretrained_encoder}" "${out_cfg}" "${PAIR_STUDY_SEED}" "${PAIR_STUDY_FRACTION_SEED}" <<'PY'
import sys
import yaml

src, run_root, pretrained_encoder, outp, train_seed, fraction_seed = sys.argv[1:7]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
cfg.setdefault("paths", {})["run_root"] = run_root
train = cfg.setdefault("train", {})
cfg["seed"] = int(train_seed)
train["seed"] = int(train_seed)
train["fraction_seed"] = int(fraction_seed)
train["use_labeled_fraction"] = False
train["labeled_fraction"] = 1.0
train["balance_fraction_by_class"] = True
train["min_samples_per_class"] = 1
train["drop_last"] = False
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

make_eval_base_cfg() {
  local src_template="$1"
  local run_root="$2"
  local split_dir="$3"
  local experiment_name="$4"
  local out_cfg="$5"
  make_base_cfg "${src_template}" "${run_root}" "${split_dir}" "${experiment_name}" "${out_cfg}"
}

make_test_stage_cfg() {
  local out_cfg="$1"
  cp "${TEST_TEMPLATE}" "${out_cfg}"
}

run_eval_steps() {
  local eval_base_cfg="$1"
  local finetune_dir="$2"
  local test_stage_cfg="$3"
  local run_log="$4"

  echo "[PAIR STUDY] test" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_test \
    --base_cfg "${eval_base_cfg}" \
    --stage_cfg "${test_stage_cfg}" >> "${run_log}" 2>&1

  if [[ "${PAIR_RUN_ANALYZE}" == "true" ]]; then
    echo "[PAIR STUDY] analyze" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
    "${PYTHON_BIN}" -m src.detection.analysis.analyze \
      --base_cfg "${eval_base_cfg}" \
      --stage_cfg "${ANALYZE_TEMPLATE}" >> "${run_log}" 2>&1
  fi

  if [[ "${PAIR_RUN_TSNE}" == "true" ]]; then
    echo "[PAIR STUDY] tsne" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
    PYTHON_BIN="${PYTHON_BIN}" bash scripts/gpu/run_tsne.sh "${GPU}" "${finetune_dir}" >> "${run_log}" 2>&1
  fi
}

run_experiment_bundle() {
  local label="$1"
  local profile_site="$2"
  local split_dir="$3"
  local experiment_name="$4"
  local run_root="$5"

  resolve_site_templates "${profile_site}"
  local base_cfg="${TMP_DIR}/${label}_base.yaml"
  local pretrain_stage_cfg="${TMP_DIR}/${label}_pretrain.yaml"
  local test_stage_cfg="${TMP_DIR}/${label}_test.yaml"

  make_base_cfg "${SITE_BASE_TEMPLATE}" "${run_root}" "${split_dir}" "${experiment_name}" "${base_cfg}"
  make_pretrain_stage_cfg "${run_root}" "${pretrain_stage_cfg}"
  make_test_stage_cfg "${test_stage_cfg}"

  {
    echo "============================================================"
    echo "[PAIR STUDY START] $(date '+%F %T') label=${label}"
    echo "profile_site=${profile_site}"
    echo "split_dir=${split_dir}"
    echo "experiment_name=${experiment_name}"
    echo "run_root=${run_root}"
    echo "methods=${PAIR_METHODS}"
    echo "train_seed=${PAIR_STUDY_SEED}"
    echo "fraction_seed=${PAIR_STUDY_FRACTION_SEED}"
    echo "pooling=${PAIR_STUDY_POOLING}"
    echo "============================================================"
  } | tee -a "${LOG_FILE}"

  IFS=',' read -r -a METHOD_ARRAY <<< "${PAIR_METHODS}"
  for method in "${METHOD_ARRAY[@]}"; do
    method="$(echo "${method}" | xargs)"
    local method_root="${run_root}/${method}"
    local run_log="${LOG_ROOT}/${label}__${method}.log"
    local ft_stage_cfg="${TMP_DIR}/${label}_finetune_${method}.yaml"
    local eval_base_cfg="${TMP_DIR}/${label}_eval_base_${method}.yaml"
    local finetune_dir="${method_root}/finetune/${experiment_name}"

    case "${method}" in
      scratch)
        pretrained_encoder="none"
        ;;
      reconst)
        local ckpt="${run_root}/pretrain/${experiment_name}/best_encoder.pt"
        if [[ "${PAIR_RUN_PRETRAIN}" == "true" ]]; then
          if [[ ! -f "${ckpt}" ]]; then
            echo "[PAIR STUDY] pretrain method=${method} label=${label}" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
            CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_pretrain \
              --base_cfg "${base_cfg}" \
              --stage_cfg "${pretrain_stage_cfg}" >> "${run_log}" 2>&1
          else
            echo "[SKIP] existing pretrain checkpoint: ${ckpt}" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
          fi
          pretrained_encoder="${ckpt}"
        else
          if [[ -f "${ckpt}" ]]; then
            pretrained_encoder="${ckpt}"
          else
            echo "[ERROR] missing pretrained encoder checkpoint: ${ckpt}" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
            exit 1
          fi
        fi
        ;;
      *)
        echo "[ERROR] unsupported method: ${method}" | tee -a "${LOG_FILE}"
        exit 1
        ;;
    esac

    make_finetune_stage_cfg "${SITE_FINETUNE_TEMPLATE}" "${method_root}" "${pretrained_encoder}" "${ft_stage_cfg}"
    make_eval_base_cfg "${SITE_BASE_TEMPLATE}" "${method_root}" "${split_dir}" "${experiment_name}" "${eval_base_cfg}"

    {
      echo "============================================================"
      echo "[PAIR STUDY RUN] $(date '+%F %T') label=${label} method=${method}"
      echo "base_cfg=${base_cfg}"
      echo "finetune_stage_cfg=${ft_stage_cfg}"
      echo "test_stage_cfg=${test_stage_cfg}"
      echo "pretrained_encoder=${pretrained_encoder}"
      echo "finetune_dir=${finetune_dir}"
      echo "============================================================"
    } | tee -a "${LOG_FILE}" | tee -a "${run_log}"

    echo "[PAIR STUDY] finetune" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m src.detection.training.trainer_finetune \
      --base_cfg "${base_cfg}" \
      --stage_cfg "${ft_stage_cfg}" >> "${run_log}" 2>&1

    run_eval_steps "${eval_base_cfg}" "${finetune_dir}" "${test_stage_cfg}" "${run_log}"
    echo "[DONE] label=${label} method=${method}" | tee -a "${LOG_FILE}" | tee -a "${run_log}"
  done

  "${PYTHON_BIN}" scripts/gpu/summarize_pohang_main_study.py \
    --root "${run_root}" \
    --out "${run_root}/summary.csv" >> "${LOG_FILE}" 2>&1
  echo "[DONE] summary: ${run_root}/summary.csv" | tee -a "${LOG_FILE}"
}

if [[ "${PAIR_RUN_MIXED}" == "true" ]]; then
  make_pair_mixed_split "${MIXED_SPLIT_DIR}"
  run_experiment_bundle \
    "mixed_pohang_utah_2019" \
    "pohang" \
    "${MIXED_SPLIT_DIR}" \
    "stage2_joint_pohang_utah_2019" \
    "${RUN_ROOT_BASE}/mixed_pohang_utah_2019"
fi

if [[ "${PAIR_RUN_CROSS}" == "true" ]]; then
  if [[ "${PAIR_RUN_CROSS_P2U}" == "true" ]]; then
    run_experiment_bundle \
      "cross_pohang_to_utah_2019" \
      "pohang" \
      "data/0406/metadata/experiments/stage3_pohang_to_utah_2019" \
      "stage3_pohang_to_utah_2019" \
      "${RUN_ROOT_BASE}/cross_pohang_to_utah_2019"
  fi

  if [[ "${PAIR_RUN_CROSS_U2P}" == "true" ]]; then
    run_experiment_bundle \
      "cross_utah_2019_to_pohang" \
      "utah_2019" \
      "data/0406/metadata/experiments/stage3_utah_2019_to_pohang" \
      "stage3_utah_2019_to_pohang" \
      "${RUN_ROOT_BASE}/cross_utah_2019_to_pohang"
  fi
fi

echo "[DONE] pair cross and mixed study completed"
echo "[INFO] run root: ${RUN_ROOT_BASE}"
