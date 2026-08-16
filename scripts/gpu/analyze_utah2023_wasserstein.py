#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from src.detection.training.trainer_finetune import (
    compute_center_diagnostics,
    compute_wasserstein_diagnostics,
)
from src.detection.training.trainer_test import (
    TestMSDNet,
    load_finetuned_model_and_center,
    resolve_val_test_dataloaders,
)
from src.detection.utils.config_io import ensure_dir, load_config


def _load_existing_summary_metrics(finetune_dir: Path) -> Dict[str, float]:
    summary_path = finetune_dir.parent.parent / "summary.csv"
    if not summary_path.exists():
        return {}
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return {}

    run_name = finetune_dir.name
    row = df[df["run_name"] == run_name]
    if row.empty:
        return {}
    row = row.iloc[0]
    out = {}
    for col in [
        "best_val_metric",
        "best_val_epoch",
        "test_f1",
        "test_bal_acc",
        "test_specificity",
        "test_precision",
        "test_recall",
        "test_best_branch",
    ]:
        if col in row.index:
            out[col] = row[col]
    return out


def analyze_run(site: str, variant: str, finetune_dir: Path, checkpoint_path: Path) -> Dict[str, object]:
    base_cfg = finetune_dir / "config_snapshot" / "base_config.yaml"
    stage_cfg = finetune_dir / "config_snapshot" / "stage_config.yaml"
    if not base_cfg.exists() or not stage_cfg.exists():
        raise FileNotFoundError(f"missing config snapshots under {finetune_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    cfg = load_config(str(base_cfg), str(stage_cfg))
    cfg.setdefault("train", {})
    cfg["train"]["log_wasserstein_diagnostics"] = True
    cfg["train"]["wasserstein_num_projections"] = int(cfg["train"].get("wasserstein_num_projections", 32))
    cfg["train"]["wasserstein_num_quantiles"] = int(cfg["train"].get("wasserstein_num_quantiles", 128))

    val_loader, test_loader = resolve_val_test_dataloaders(cfg)
    device = torch.device("cpu")

    model = TestMSDNet(cfg).to(device)
    center_c, _ = load_finetuned_model_and_center(model, checkpoint_path, device)

    row: Dict[str, object] = {
        "site": site,
        "variant": variant,
        "run_name": finetune_dir.name,
        "checkpoint": str(checkpoint_path),
        "wasserstein_num_projections": int(cfg["train"]["wasserstein_num_projections"]),
        "wasserstein_num_quantiles": int(cfg["train"]["wasserstein_num_quantiles"]),
    }

    if val_loader is not None:
        row.update(compute_center_diagnostics(model, val_loader, device, center_c, prefix="val"))
        row.update(compute_wasserstein_diagnostics(model, val_loader, device, prefix="val", cfg=cfg))
    if test_loader is not None:
        row.update(compute_center_diagnostics(model, test_loader, device, center_c, prefix="test"))
        row.update(compute_wasserstein_diagnostics(model, test_loader, device, prefix="test", cfg=cfg))

    row.update(_load_existing_summary_metrics(finetune_dir))
    return row


def build_specs() -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    main_root = Path("runs/utah_2023_main_study/reconst/finetune")
    for frac in ["0p05", "0p1", "0p25", "0p5", "1"]:
        specs.append(
            {
                "site": "utah_2023",
                "variant": "baseline",
                "finetune_dir": str(main_root / f"base_utah_2023__frac{frac}"),
            }
        )

    for variant in ["bandpass_agc_none", "bandpass_agc_robust"]:
        var_root = Path(f"runs/utah_2023_normalization_ablation_v2/{variant}/reconst/finetune")
        for frac in ["0p25", "0p5", "1"]:
            specs.append(
                {
                    "site": "utah_2023",
                    "variant": variant,
                    "finetune_dir": str(var_root / f"base_utah_2023__frac{frac}"),
                }
            )
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=str,
        default="runs/utah_2023_wasserstein_offline_v1",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    ensure_dir(output_root)

    rows: List[Dict[str, object]] = []
    for spec in build_specs():
        finetune_dir = Path(spec["finetune_dir"])
        checkpoint_path = finetune_dir / "best.pt"
        row = analyze_run(
            site=spec["site"],
            variant=spec["variant"],
            finetune_dir=finetune_dir,
            checkpoint_path=checkpoint_path,
        )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["variant", "run_name"])
    csv_path = output_root / "summary.csv"
    df.to_csv(csv_path, index=False)

    quick_cols = [
        "variant",
        "run_name",
        "val_event_noise_swd",
        "val_dist_gap_event_minus_noise",
        "test_event_noise_swd",
        "test_dist_gap_event_minus_noise",
        "test_f1",
        "test_bal_acc",
        "test_specificity",
        "test_best_branch",
    ]
    present_cols = [c for c in quick_cols if c in df.columns]
    print(df[present_cols].to_string(index=False))

    stats_path = output_root / "summary.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"num_runs": len(df), "csv": str(csv_path)}, f, indent=2)


if __name__ == "__main__":
    main()
