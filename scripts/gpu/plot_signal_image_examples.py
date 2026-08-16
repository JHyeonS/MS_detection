#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "figures" / "signal_image"

DATASETS = {
    "raw": ROOT / "data" / "visualbest_raw_rms_fs1000_rms0p15_nofilter",
    "lowpass": ROOT / "data" / "visualbest_filter_rms_fs1000_rms0p15_lp50",
    "log_envelope": ROOT / "data" / "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5",
}
PANEL_LABELS = {
    "raw": "Raw",
    "lowpass": "Low-pass + RMS",
    "log_envelope": "Log-envelope",
}
SITES = ["pohang", "utah_2019", "utah_2023"]
SITE_LABELS = {
    "pohang": "Pohang",
    "utah_2019": "Utah 2019",
    "utah_2023": "Utah 2023",
}
CHANNEL_SPACING_M = {
    "pohang": 2.0,
    "utah_2019": 3.35,
    "utah_2023": 1.0249,
}
TARGET_QUANTILES = [0.35, 0.60, 0.85]


def sample_id_from_row(row: pd.Series) -> str:
    return Path(str(row["npy_path"])).stem


def npy_path(dataset_key: str, site: str, sample_id: str) -> Path:
    return DATASETS[dataset_key] / site / "1_event" / f"{sample_id}.npy"


def load_event_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(DATASETS["log_envelope"] / "metadata" / "all_samples.csv")
    manifest = manifest[(manifest["site"].isin(SITES)) & (manifest["label"].eq(1))].copy()
    manifest["sample_id"] = manifest.apply(sample_id_from_row, axis=1)
    return manifest


def robust_energy(path: Path) -> float:
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr[0]
    return float(np.nanpercentile(np.abs(arr), 99.5))


def select_examples() -> pd.DataFrame:
    manifest = load_event_manifest()
    rows = []
    for site in SITES:
        site_df = manifest[manifest["site"].eq(site)].copy()
        scored = []
        for _, row in site_df.iterrows():
            sample_id = str(row["sample_id"])
            paths = {key: npy_path(key, site, sample_id) for key in DATASETS}
            if not all(path.exists() for path in paths.values()):
                continue
            scored.append(
                {
                    "site": site,
                    "sample_id": sample_id,
                    "group_id": row.get("group_id", ""),
                    "start_sec": row.get("start_sec", np.nan),
                    "end_sec": row.get("end_sec", np.nan),
                    "ch_start": row.get("ch_start", np.nan),
                    "ch_end": row.get("ch_end", np.nan),
                    "score": robust_energy(paths["log_envelope"]),
                }
            )
        scored_df = pd.DataFrame(scored).sort_values("score").reset_index(drop=True)
        if scored_df.empty:
            raise RuntimeError(f"No matched event samples found for {site}")

        used_groups: set[str] = set()
        for example_no, quantile in enumerate(TARGET_QUANTILES, start=1):
            target = scored_df["score"].quantile(quantile)
            candidates = scored_df.copy()
            if used_groups and candidates["group_id"].nunique() > len(used_groups):
                candidates = candidates[~candidates["group_id"].astype(str).isin(used_groups)]
            idx = (candidates["score"] - target).abs().idxmin()
            selected = scored_df.loc[idx].to_dict()
            selected["example_no"] = example_no
            selected["selection_quantile"] = quantile
            used_groups.add(str(selected["group_id"]))
            rows.append(selected)
    return pd.DataFrame(rows)


def read_array(path: Path) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D DAS window, got {arr.shape}: {path}")
    return arr


def robust_limits(arr: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    lim = float(np.nanpercentile(np.abs(arr), percentile))
    if not np.isfinite(lim) or lim <= 0:
        lim = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0
    return -lim, lim


def plot_array(
    arr: np.ndarray,
    site: str,
    out_base: Path,
    title: str | None = None,
    show_colorbar: bool = True,
) -> None:
    n_channels, n_samples = arr.shape
    duration_sec = n_samples / 1000.0
    aperture_m = (n_channels - 1) * CHANNEL_SPACING_M[site]
    vmin, vmax = robust_limits(arr)

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    im = ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[0.0, duration_sec, 0.0, aperture_m],
    )
    if title:
        ax.set_title(title, fontsize=11, fontweight="normal", pad=7)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Relative fiber distance (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label("Amplitude (a.u.)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_multipanel(arrays: dict[str, np.ndarray], site: str, out_base: Path) -> None:
    n_channels, n_samples = next(iter(arrays.values())).shape
    duration_sec = n_samples / 1000.0
    aperture_m = (n_channels - 1) * CHANNEL_SPACING_M[site]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), sharex=True, sharey=True)
    for ax, key in zip(axes, ["raw", "lowpass", "log_envelope"]):
        arr = arrays[key]
        vmin, vmax = robust_limits(arr)
        im = ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[0.0, duration_sec, 0.0, aperture_m],
        )
        ax.set_title(PANEL_LABELS[key], fontsize=10.5, fontweight="normal", pad=6)
        ax.set_xlabel("Time (s)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.018)
        cbar.ax.tick_params(labelsize=6.5)
    axes[0].set_ylabel("Relative fiber distance (m)", fontsize=8.5)
    fig.tight_layout(w_pad=0.85)
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_array_no_colorbar(arr: np.ndarray, site: str, out_base: Path, title: str | None = None) -> None:
    n_channels, n_samples = arr.shape
    duration_sec = n_samples / 1000.0
    aperture_m = (n_channels - 1) * CHANNEL_SPACING_M[site]
    vmin, vmax = robust_limits(arr)

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[0.0, duration_sec, 0.0, aperture_m],
    )
    if title:
        ax.set_title(title, fontsize=11, fontweight="normal", pad=7)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Relative fiber distance (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_array_image_only(arr: np.ndarray, site: str, out_base: Path) -> None:
    n_channels, n_samples = arr.shape
    duration_sec = n_samples / 1000.0
    aperture_m = (n_channels - 1) * CHANNEL_SPACING_M[site]
    vmin, vmax = robust_limits(arr)

    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    ax.imshow(
        arr,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[0.0, duration_sec, 0.0, aperture_m],
    )
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def plot_multipanel_no_colorbar(arrays: dict[str, np.ndarray], site: str, out_base: Path) -> None:
    n_channels, n_samples = next(iter(arrays.values())).shape
    duration_sec = n_samples / 1000.0
    aperture_m = (n_channels - 1) * CHANNEL_SPACING_M[site]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45), sharex=True, sharey=True)
    for ax, key in zip(axes, ["raw", "lowpass", "log_envelope"]):
        arr = arrays[key]
        vmin, vmax = robust_limits(arr)
        ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[0.0, duration_sec, 0.0, aperture_m],
        )
        ax.set_title(PANEL_LABELS[key], fontsize=10.5, fontweight="normal", pad=6)
        ax.set_xlabel("Time (s)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    axes[0].set_ylabel("Relative fiber distance (m)", fontsize=8.5)
    fig.tight_layout(w_pad=0.55)
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_multipanel_image_only(arrays: dict[str, np.ndarray], site: str, out_base: Path) -> None:
    n_channels, n_samples = next(iter(arrays.values())).shape
    duration_sec = n_samples / 1000.0
    aperture_m = (n_channels - 1) * CHANNEL_SPACING_M[site]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45), sharex=True, sharey=True)
    for ax, key in zip(axes, ["raw", "lowpass", "log_envelope"]):
        arr = arrays[key]
        vmin, vmax = robust_limits(arr)
        ax.imshow(
            arr,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extent=[0.0, duration_sec, 0.0, aperture_m],
        )
        ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.035)
    for ext in ("pdf", "png"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Pretendard", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    selected = select_examples()
    summary_rows = []
    for _, row in selected.iterrows():
        site = str(row["site"])
        sample_id = str(row["sample_id"])
        example_no = int(row["example_no"])
        example_dir = OUT_ROOT / site / f"example_{example_no}"
        example_dir.mkdir(parents=True, exist_ok=True)

        arrays = {}
        for key in ["raw", "lowpass", "log_envelope"]:
            path = npy_path(key, site, sample_id)
            arrays[key] = read_array(path)
            out_base = example_dir / key
            plot_array(
                arrays[key],
                site=site,
                out_base=out_base,
                title=f"{SITE_LABELS[site]} example {example_no}: {PANEL_LABELS[key]}",
            )
            summary_rows.append(
                {
                    "site": site,
                    "site_label": SITE_LABELS[site],
                    "example_no": example_no,
                    "preprocessing": key,
                    "preprocessing_label": PANEL_LABELS[key],
                    "sample_id": sample_id,
                    "source_npy": str(path.relative_to(ROOT)),
                    "output_png": str(out_base.with_suffix(".png").relative_to(ROOT)),
                    "output_pdf": str(out_base.with_suffix(".pdf").relative_to(ROOT)),
                    "group_id": row.get("group_id", ""),
                    "start_sec": row.get("start_sec", np.nan),
                    "end_sec": row.get("end_sec", np.nan),
                    "ch_start": row.get("ch_start", np.nan),
                    "ch_end": row.get("ch_end", np.nan),
                    "selection_score_logenv_p99p5_abs": row["score"],
                    "selection_quantile": row["selection_quantile"],
                }
            )
            if site == "pohang":
                no_cbar_base = example_dir / f"{key}_no_colorbar"
                plot_array_no_colorbar(
                    arrays[key],
                    site=site,
                    out_base=no_cbar_base,
                    title=f"{SITE_LABELS[site]} example {example_no}: {PANEL_LABELS[key]}",
                )
                summary_rows.append(
                    {
                        "site": site,
                        "site_label": SITE_LABELS[site],
                        "example_no": example_no,
                        "preprocessing": f"{key}_no_colorbar",
                        "preprocessing_label": f"{PANEL_LABELS[key]} no colorbar",
                        "sample_id": sample_id,
                        "source_npy": str(path.relative_to(ROOT)),
                        "output_png": str(no_cbar_base.with_suffix(".png").relative_to(ROOT)),
                        "output_pdf": str(no_cbar_base.with_suffix(".pdf").relative_to(ROOT)),
                        "group_id": row.get("group_id", ""),
                        "start_sec": row.get("start_sec", np.nan),
                        "end_sec": row.get("end_sec", np.nan),
                        "ch_start": row.get("ch_start", np.nan),
                        "ch_end": row.get("ch_end", np.nan),
                        "selection_score_logenv_p99p5_abs": row["score"],
                        "selection_quantile": row["selection_quantile"],
                    }
                )
                image_only_base = example_dir / f"{key}_image_only"
                plot_array_image_only(arrays[key], site=site, out_base=image_only_base)
                summary_rows.append(
                    {
                        "site": site,
                        "site_label": SITE_LABELS[site],
                        "example_no": example_no,
                        "preprocessing": f"{key}_image_only",
                        "preprocessing_label": f"{PANEL_LABELS[key]} image only",
                        "sample_id": sample_id,
                        "source_npy": str(path.relative_to(ROOT)),
                        "output_png": str(image_only_base.with_suffix(".png").relative_to(ROOT)),
                        "output_pdf": str(image_only_base.with_suffix(".pdf").relative_to(ROOT)),
                        "group_id": row.get("group_id", ""),
                        "start_sec": row.get("start_sec", np.nan),
                        "end_sec": row.get("end_sec", np.nan),
                        "ch_start": row.get("ch_start", np.nan),
                        "ch_end": row.get("ch_end", np.nan),
                        "selection_score_logenv_p99p5_abs": row["score"],
                        "selection_quantile": row["selection_quantile"],
                    }
                )

        multi_base = example_dir / "multipanel_raw_lowpass_log_envelope"
        plot_multipanel(arrays, site=site, out_base=multi_base)
        if site == "pohang":
            multi_no_cbar_base = example_dir / "multipanel_raw_lowpass_log_envelope_no_colorbar"
            plot_multipanel_no_colorbar(arrays, site=site, out_base=multi_no_cbar_base)
            summary_rows.append(
                {
                    "site": site,
                    "site_label": SITE_LABELS[site],
                    "example_no": example_no,
                    "preprocessing": "multipanel_no_colorbar",
                    "preprocessing_label": "Raw / Low-pass + RMS / Log-envelope no colorbar",
                    "sample_id": sample_id,
                    "source_npy": "",
                    "output_png": str(multi_no_cbar_base.with_suffix(".png").relative_to(ROOT)),
                    "output_pdf": str(multi_no_cbar_base.with_suffix(".pdf").relative_to(ROOT)),
                    "group_id": row.get("group_id", ""),
                    "start_sec": row.get("start_sec", np.nan),
                    "end_sec": row.get("end_sec", np.nan),
                    "ch_start": row.get("ch_start", np.nan),
                    "ch_end": row.get("ch_end", np.nan),
                    "selection_score_logenv_p99p5_abs": row["score"],
                    "selection_quantile": row["selection_quantile"],
                }
            )
            multi_image_only_base = example_dir / "multipanel_raw_lowpass_log_envelope_image_only"
            plot_multipanel_image_only(arrays, site=site, out_base=multi_image_only_base)
            summary_rows.append(
                {
                    "site": site,
                    "site_label": SITE_LABELS[site],
                    "example_no": example_no,
                    "preprocessing": "multipanel_image_only",
                    "preprocessing_label": "Raw / Low-pass + RMS / Log-envelope image only",
                    "sample_id": sample_id,
                    "source_npy": "",
                    "output_png": str(multi_image_only_base.with_suffix(".png").relative_to(ROOT)),
                    "output_pdf": str(multi_image_only_base.with_suffix(".pdf").relative_to(ROOT)),
                    "group_id": row.get("group_id", ""),
                    "start_sec": row.get("start_sec", np.nan),
                    "end_sec": row.get("end_sec", np.nan),
                    "ch_start": row.get("ch_start", np.nan),
                    "ch_end": row.get("ch_end", np.nan),
                    "selection_score_logenv_p99p5_abs": row["score"],
                    "selection_quantile": row["selection_quantile"],
                }
            )
        summary_rows.append(
            {
                "site": site,
                "site_label": SITE_LABELS[site],
                "example_no": example_no,
                "preprocessing": "multipanel",
                "preprocessing_label": "Raw / Low-pass + RMS / Log-envelope",
                "sample_id": sample_id,
                "source_npy": "",
                "output_png": str(multi_base.with_suffix(".png").relative_to(ROOT)),
                "output_pdf": str(multi_base.with_suffix(".pdf").relative_to(ROOT)),
                "group_id": row.get("group_id", ""),
                "start_sec": row.get("start_sec", np.nan),
                "end_sec": row.get("end_sec", np.nan),
                "ch_start": row.get("ch_start", np.nan),
                "ch_end": row.get("ch_end", np.nan),
                "selection_score_logenv_p99p5_abs": row["score"],
                "selection_quantile": row["selection_quantile"],
            }
        )

    pd.DataFrame(summary_rows).to_csv(OUT_ROOT / "signal_image_selection_summary.csv", index=False)
    print(f"[DONE] wrote {len(summary_rows)} figure records under {OUT_ROOT}")


if __name__ == "__main__":
    main()
