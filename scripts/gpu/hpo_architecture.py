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
    encoder_cfg = overrides.get("model", {}).get("encoder", {})
    layers = encoder_cfg.get("num_layers", "na")
    channels = encoder_cfg.get("base_channels", "na")
    latent = encoder_cfg.get("latent_dim", "na")
    dropout = encoder_cfg.get("dropout", "na")
    return (
        f"arch_t{idx:03d}"
        f"__ly{layers}"
        f"__ch{channels}"
        f"__ld{latent}"
        f"__drop{str(dropout).replace('.', 'p')}"
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


def write_yaml(path: Path, data: dict[str, Any]):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def run_stage(
    *,
    python_bin: str,
    module: str,
    base_cfg: Path,
    stage_cfg: Path,
    workdir: Path,
    env: dict[str, str],
    log_path: Path,
    exp_suffix: str | None = None,
):
    cmd = [
        python_bin,
        "-m",
        module,
        "--base_cfg",
        str(base_cfg),
        "--stage_cfg",
        str(stage_cfg),
    ]
    if exp_suffix:
        cmd.extend(["--exp_suffix", exp_suffix])

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
        results.sort(
            key=lambda x: (
                x["metric_value"] is None,
                x["metric_value"] if x["metric_value"] is not None else float("inf"),
            )
        )

    write_leaderboard(results, output_root / "leaderboard.csv")
    with open(output_root / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Architecture HPO runner with per-trial pretrain + finetune")
    parser.add_argument("--spec", type=str, required=True)
    args = parser.parse_args()

    spec = load_yaml(args.spec).get("hpo_arch", {})
    if not spec:
        raise ValueError("Spec file must contain top-level 'hpo_arch' block.")

    workdir = Path.cwd()
    base_cfg_path = Path(spec["base_cfg"])
    base_cfg = load_yaml(base_cfg_path)
    pretrain_stage_cfg = load_yaml(spec["pretrain_stage_cfg"])
    finetune_stage_cfg = load_yaml(spec["finetune_stage_cfg"])
    output_root = ensure_dir(spec["output_root"])
    generated_dir = ensure_dir(output_root / "generated_configs")
    logs_dir = ensure_dir(output_root / "logs")
    trials_root = ensure_dir(output_root / "trial_runs")

    metric = str(spec.get("metric", "val_f1"))
    mode = str(spec.get("mode", "max")).lower()
    seed = int(spec.get("seed", 42))
    max_trials = int(spec.get("max_trials", 12))
    pretrain_overrides = spec.get("pretrain_overrides", {})
    finetune_overrides = spec.get("finetune_overrides", {})
    search_space = spec.get("search_space", {})

    trials = sample_trials(search_space, max_trials=max_trials, seed=seed)
    python_bin = os.environ.get("PYTHON_BIN", sys.executable)
    base_experiment = str(base_cfg["data"]["experiment"])

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("MPLBACKEND", "Agg")

    gpu_workers = resolve_gpu_workers()
    trial_buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {gpu: [] for gpu in gpu_workers}
    for idx, sampled in enumerate(trials, start=1):
        gpu = gpu_workers[(idx - 1) % len(gpu_workers)]
        trial_buckets[gpu].append((idx, sampled))

    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()

    def run_bucket(gpu: str, bucket: list[tuple[int, dict[str, Any]]]):
        for idx, sampled in bucket:
            trial_env = env.copy()
            trial_env["CUDA_VISIBLE_DEVICES"] = gpu

            suffix = trial_suffix(idx, sampled)
            trial_run_root = trials_root / suffix
            ensure_dir(trial_run_root)

            sampled_base_cfg = deep_update(base_cfg, sampled)
            sampled_base_cfg = deep_update(sampled_base_cfg, {"paths": {"run_root": str(trial_run_root)}})

            sampled_pretrain_cfg = deep_update(pretrain_stage_cfg, pretrain_overrides)
            sampled_finetune_cfg = deep_update(finetune_stage_cfg, finetune_overrides)

            base_cfg_out = generated_dir / f"{suffix}__base.yaml"
            pretrain_cfg_out = generated_dir / f"{suffix}__pretrain.yaml"
            finetune_cfg_out = generated_dir / f"{suffix}__finetune.yaml"
            log_path = logs_dir / f"{suffix}.log"

            write_yaml(base_cfg_out, sampled_base_cfg)
            write_yaml(pretrain_cfg_out, sampled_pretrain_cfg)
            write_yaml(finetune_cfg_out, sampled_finetune_cfg)

            with open(log_path, "w", encoding="utf-8") as log_f:
                log_f.write(f"spec={args.spec}\n")
                log_f.write(f"trial_index={idx}\n")
                log_f.write(f"trial_suffix={suffix}\n")
                log_f.write(f"assigned_gpu={gpu}\n")
                log_f.write(json.dumps(sampled, indent=2, ensure_ascii=False) + "\n")

            pretrain_proc = run_stage(
                python_bin=python_bin,
                module="src.detection.training.trainer_pretrain",
                base_cfg=base_cfg_out,
                stage_cfg=pretrain_cfg_out,
                workdir=workdir,
                env=trial_env,
                log_path=log_path,
            )

            finetune_returncode = None
            if pretrain_proc.returncode == 0:
                finetune_proc = run_stage(
                    python_bin=python_bin,
                    module="src.detection.training.trainer_finetune",
                    base_cfg=base_cfg_out,
                    stage_cfg=finetune_cfg_out,
                    workdir=workdir,
                    env=trial_env,
                    log_path=log_path,
                    exp_suffix=suffix,
                )
                finetune_returncode = finetune_proc.returncode
            else:
                with open(log_path, "a", encoding="utf-8") as log_f:
                    log_f.write(f"[ERROR] pretrain failed for {suffix}; skipping finetune\n")

            summary_path = trial_run_root / "finetune" / f"{base_experiment}__{suffix}" / "finetune_summary.json"
            metric_value = metric_from_summary(summary_path, metric)
            row = {
                "trial_index": idx,
                "trial_suffix": suffix,
                "assigned_gpu": gpu,
                "pretrain_returncode": pretrain_proc.returncode,
                "finetune_returncode": finetune_returncode,
                "metric": metric,
                "metric_value": metric_value,
                "summary_path": str(summary_path),
                "base_cfg_path": str(base_cfg_out),
                "pretrain_cfg_path": str(pretrain_cfg_out),
                "finetune_cfg_path": str(finetune_cfg_out),
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
