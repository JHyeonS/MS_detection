#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def flatten_space(tree: dict[str, Any], prefix: str = ""):
    items = []
    for key, value in tree.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(flatten_space(value, prefix=full_key))
        else:
            items.append((full_key, list(value)))
    return items


def flatten_values(tree: dict[str, Any], prefix: str = ""):
    items = {}
    for key, value in tree.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_values(value, prefix=full_key))
        else:
            items[full_key] = value
    return items


def nested_from_dotkey(dotkey: str, value: Any) -> dict[str, Any]:
    keys = dotkey.split(".")
    out = value
    for key in reversed(keys):
        out = {key: out}
    return out


def sample_trials(search_space: dict[str, Any], max_trials: int, seed: int):
    flat = flatten_space(search_space)
    keys = [k for k, _ in flat]
    values = [v for _, v in flat]
    all_combinations = list(itertools.product(*values))
    rng = random.Random(seed)
    rng.shuffle(all_combinations)
    selected = all_combinations[: min(max_trials, len(all_combinations))]

    trials = []
    for combo in selected:
        merged = {}
        for key, value in zip(keys, combo):
            merged = deep_update(merged, nested_from_dotkey(key, value))
        trials.append(merged)
    return trials


def trial_suffix(idx: int, overrides: dict[str, Any]) -> str:
    train_cfg = overrides.get("train", {})
    model_cfg = overrides.get("model", {}).get("encoder", {})
    lr = train_cfg.get("lr", "na")
    wd = train_cfg.get("weight_decay", "na")
    bs = train_cfg.get("batch_size", "na")
    anom = train_cfg.get("anomaly_loss_weight", "na")
    freeze = train_cfg.get("freeze_encoder", "na")
    dropout = model_cfg.get("dropout", "na")
    return (
        f"hpo_t{idx:03d}"
        f"__lr{str(lr).replace('.', 'p')}"
        f"__wd{str(wd).replace('.', 'p')}"
        f"__bs{bs}"
        f"__drop{str(dropout).replace('.', 'p')}"
        f"__anom{str(anom).replace('.', 'p')}"
        f"__frz{int(bool(freeze))}"
    )


def metric_from_summary(summary_path: Path, metric: str):
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if metric == "best_metric":
        return summary.get("best_metric")

    last_val = summary.get("last_val_metrics") or {}
    if metric.startswith("val_"):
        return last_val.get(metric[len("val_"):])
    return summary.get(metric)


def run_trial(
    python_bin: str,
    base_cfg: str,
    stage_cfg: Path,
    exp_suffix: str,
    workdir: Path,
    env: dict[str, str],
    log_path: Path,
):
    cmd = [
        python_bin,
        "-m",
        "src.detection.training.trainer_finetune",
        "--base_cfg",
        base_cfg,
        "--stage_cfg",
        str(stage_cfg),
        "--exp_suffix",
        exp_suffix,
    ]
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write("CMD: " + " ".join(cmd) + "\n")
        log_f.flush()
        return subprocess.run(
            cmd,
            cwd=str(workdir),
            env=env,
            check=False,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )


def write_leaderboard(rows, out_path: Path):
    ensure_dir(out_path.parent)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    columns = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(col, "")) for col in columns) + "\n")


def resolve_gpu_workers() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return ["0"]
    return [gpu.strip() for gpu in visible.split(",") if gpu.strip()]


def persist_results(results: list[dict[str, Any]], mode: str, output_root: Path):
    if mode == "max":
        results.sort(key=lambda x: (x["metric_value"] is not None, x["metric_value"]), reverse=True)
    else:
        results.sort(key=lambda x: (x["metric_value"] is None, x["metric_value"] if x["metric_value"] is not None else float("inf")))

    write_leaderboard(results, output_root / "leaderboard.csv")
    with open(output_root / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def write_yaml(path: Path, data: dict[str, Any]):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_pretrain(
    python_bin: str,
    base_cfg: str,
    stage_cfg: Path,
    workdir: Path,
    env: dict[str, str],
    log_path: Path,
):
    cmd = [
        python_bin,
        "-m",
        "src.detection.training.trainer_pretrain",
        "--base_cfg",
        base_cfg,
        "--stage_cfg",
        str(stage_cfg),
    ]
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write("CMD: " + " ".join(cmd) + "\n")
        log_f.flush()
        return subprocess.run(
            cmd,
            cwd=str(workdir),
            env=env,
            check=False,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )


def main():
    parser = argparse.ArgumentParser(description="Random-search HPO runner for finetune stage")
    parser.add_argument("--spec", type=str, required=True)
    args = parser.parse_args()

    spec = load_yaml(args.spec).get("hpo", {})
    if not spec:
        raise ValueError("Spec file must contain top-level 'hpo' block.")

    workdir = Path.cwd()
    base_cfg = str(spec["base_cfg"])
    stage_cfg = load_yaml(spec["stage_cfg"])
    output_root = ensure_dir(spec["output_root"])
    generated_dir = ensure_dir(output_root / "generated_configs")
    logs_dir = ensure_dir(output_root / "logs")

    metric = str(spec.get("metric", "val_f1"))
    mode = str(spec.get("mode", "max")).lower()
    seed = int(spec.get("seed", 42))
    max_trials = int(spec.get("max_trials", 20))
    fixed_overrides = spec.get("fixed_overrides", {})
    search_space = spec.get("search_space", {})
    pretrain_spec = spec.get("pretrain", {})

    trials = sample_trials(search_space, max_trials=max_trials, seed=seed)
    python_bin = os.environ.get("PYTHON_BIN", sys.executable)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("MPLBACKEND", "Agg")

    base_cfg_yaml = load_yaml(base_cfg)
    base_experiment = str(base_cfg_yaml["data"]["experiment"])
    effective_run_root = Path(
        fixed_overrides.get("paths", {}).get(
            "run_root",
            base_cfg_yaml.get("paths", {}).get("run_root", "./runs"),
        )
    )

    pretrain_enabled = bool(pretrain_spec.get("enabled", False))
    if pretrain_enabled:
        pretrain_stage_cfg_path = str(pretrain_spec.get("stage_cfg", "configs/train/pretrain_reconst.yaml"))
        pretrain_overrides = pretrain_spec.get("overrides", {})
        pretrain_log_path = logs_dir / "pretrain_for_hpo.log"
        generated_pretrain_cfg_path = generated_dir / "pretrain_for_hpo.yaml"
        pretrain_stage_cfg = load_yaml(pretrain_stage_cfg_path)
        pretrain_stage_cfg = deep_update(pretrain_stage_cfg, {"paths": {"run_root": str(effective_run_root)}})
        pretrain_stage_cfg = deep_update(pretrain_stage_cfg, pretrain_overrides)
        write_yaml(generated_pretrain_cfg_path, pretrain_stage_cfg)

        pretrain_proc = run_pretrain(
            python_bin=python_bin,
            base_cfg=base_cfg,
            stage_cfg=generated_pretrain_cfg_path,
            workdir=workdir,
            env=env,
            log_path=pretrain_log_path,
        )
        if pretrain_proc.returncode != 0:
            raise RuntimeError(
                "Pretrain step failed before HPO. "
                f"See log: {pretrain_log_path}"
            )

    gpu_workers = resolve_gpu_workers()
    trial_buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {gpu: [] for gpu in gpu_workers}
    for idx, sampled in enumerate(trials, start=1):
        gpu = gpu_workers[(idx - 1) % len(gpu_workers)]
        trial_buckets[gpu].append((idx, sampled))

    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()

    def run_bucket(gpu: str, bucket: list[tuple[int, dict[str, Any]]]):
        for idx, sampled in bucket:
            merged_stage = deep_update(stage_cfg, fixed_overrides)
            merged_stage = deep_update(merged_stage, sampled)
            suffix = trial_suffix(idx, sampled)
            trial_cfg_path = generated_dir / f"{suffix}.yaml"
            log_path = logs_dir / f"{suffix}.log"
            trial_env = env.copy()
            trial_env["CUDA_VISIBLE_DEVICES"] = gpu

            with open(trial_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(merged_stage, f, sort_keys=False, allow_unicode=True)

            with open(log_path, "w", encoding="utf-8") as log_f:
                log_f.write(f"spec={args.spec}\n")
                log_f.write(f"trial_index={idx}\n")
                log_f.write(f"trial_suffix={suffix}\n")
                log_f.write(f"assigned_gpu={gpu}\n")
                log_f.write(json.dumps(sampled, indent=2, ensure_ascii=False) + "\n")

            proc = run_trial(
                python_bin=python_bin,
                base_cfg=base_cfg,
                stage_cfg=trial_cfg_path,
                exp_suffix=suffix,
                workdir=workdir,
                env=trial_env,
                log_path=log_path,
            )

            save_dir = effective_run_root / "finetune"
            summary_path = save_dir / f"{base_experiment}__{suffix}" / "finetune_summary.json"
            metric_value = metric_from_summary(summary_path, metric)

            row = {
                "trial_index": idx,
                "trial_suffix": suffix,
                "assigned_gpu": gpu,
                "returncode": proc.returncode,
                "metric": metric,
                "metric_value": metric_value,
                "summary_path": str(summary_path),
                "config_path": str(trial_cfg_path),
                **{f"sampled_{k}": v for k, v in flatten_values(sampled).items()},
            }
            with results_lock:
                results.append(row)
                persist_results(results, mode, output_root)

    with ThreadPoolExecutor(max_workers=len(gpu_workers)) as executor:
        futures = [executor.submit(run_bucket, gpu, bucket) for gpu, bucket in trial_buckets.items() if bucket]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
