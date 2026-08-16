#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "temp" / "current_results_summary" / "figures_metadata_v2" / "site_preprocessing_examples"

DATASETS = {
    "logenv": ROOT / "data" / "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5",
    "filter_rms": ROOT / "data" / "visualbest_filter_rms_fs1000_rms0p15_lp50",
}
PREPROCESS_LABEL = {
    "logenv": "Log-envelope",
    "filter_rms": "Filter + RMS",
}
SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABEL = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
CHANNEL_SPACING_M = {
    "pohang": 2.0,
    "utah_2019": 3.35,
    "utah_2023": 1.0249,
}
TARGET_QUANTILES = [0.70, 0.90]


def robust_score(path: Path) -> float:
    arr = np.load(path)
    return float(np.nanpercentile(np.abs(arr), 99.5))


def load_event_manifest(dataset_root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(dataset_root / "metadata" / "all_samples.csv")
    manifest = manifest[(manifest["label_name"] == "event") | (manifest["label"] == 1)].copy()
    manifest["sample_id"] = manifest["npy_path"].map(lambda p: Path(str(p)).stem)
    return manifest


def select_samples() -> pd.DataFrame:
    log_manifest = load_event_manifest(DATASETS["logenv"])
    selections = []
    for site in SITES:
        site_df = log_manifest[log_manifest["site"] == site].copy()
        scored = []
        for _, row in site_df.iterrows():
            path = ROOT / row["npy_path"]
            if not path.exists():
                continue
            scored.append(
                {
                    "site": site,
                    "sample_id": row["sample_id"],
                    "group_id": row.get("group_id", ""),
                    "start_sec": row.get("start_sec", np.nan),
                    "end_sec": row.get("end_sec", np.nan),
                    "ch_start": row.get("ch_start", np.nan),
                    "ch_end": row.get("ch_end", np.nan),
                    "score": robust_score(path),
                    "logenv_path": str(path.relative_to(ROOT)),
                }
            )
        scored_df = pd.DataFrame(scored).sort_values("score").reset_index(drop=True)
        used_groups: set[str] = set()
        for sample_no, quantile in enumerate(TARGET_QUANTILES, start=1):
            target = scored_df["score"].quantile(quantile)
            candidates = scored_df.copy()
            if used_groups and candidates["group_id"].nunique() > len(used_groups):
                candidates = candidates[~candidates["group_id"].isin(used_groups)]
            idx = (candidates["score"] - target).abs().idxmin()
            selected = scored_df.loc[idx].to_dict()
            selected["sample_no"] = sample_no
            selected["selection_quantile"] = quantile
            used_groups.add(str(selected["group_id"]))
            selections.append(selected)
    return pd.DataFrame(selections)


def image_limits(arr: np.ndarray) -> tuple[float, float]:
    limit = float(np.nanpercentile(np.abs(arr), 99.0))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def plot_one(path: Path, site: str, out_base: Path) -> None:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}: {path}")
    n_channels, n_samples = arr.shape
    duration_sec = n_samples / 1000.0
    aperture_m = (n_channels - 1) * CHANNEL_SPACING_M[site]
    vmin, vmax = image_limits(arr)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(
        arr,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=[0.0, duration_sec, 0.0, aperture_m],
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative fiber distance (m)")
    ax.tick_params(labelsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label("Amplitude (a.u.)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_samples()
    output_rows = []
    for _, row in selected.iterrows():
        site = row["site"]
        sample_id = row["sample_id"]
        sample_no = int(row["sample_no"])
        for preprocess, dataset_root in DATASETS.items():
            path = dataset_root / site / "1_event" / f"{sample_id}.npy"
            if not path.exists():
                continue
            out_base = OUT_DIR / f"{site}_{preprocess}_event_example_{sample_no}"
            plot_one(path, site, out_base)
            output_rows.append(
                {
                    "site": site,
                    "site_label": SITE_LABEL[site],
                    "preprocessing": preprocess,
                    "preprocessing_label": PREPROCESS_LABEL[preprocess],
                    "sample_no": sample_no,
                    "sample_id": sample_id,
                    "source_npy": str(path.relative_to(ROOT)),
                    "output_pdf": str(out_base.with_suffix(".pdf").relative_to(ROOT)),
                    "output_png": str(out_base.with_suffix(".png").relative_to(ROOT)),
                    "selection_score_logenv_p99p5_abs": row["score"],
                    "selection_quantile": row["selection_quantile"],
                    "group_id": row["group_id"],
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "ch_start": row["ch_start"],
                    "ch_end": row["ch_end"],
                    "channel_spacing_m": CHANNEL_SPACING_M[site],
                }
            )
    pd.DataFrame(output_rows).to_csv(OUT_DIR / "site_preprocessing_example_selection.csv", index=False)
    print(f"[DONE] wrote {len(output_rows)} examples to {OUT_DIR}")


if __name__ == "__main__":
    main()
