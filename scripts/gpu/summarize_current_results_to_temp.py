#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "temp" / "current_results_summary"
FRACTIONS = ["0p05", "0p1", "0p25", "0p5", "1"]
FRACTION_LABEL = {
    "0p05": "0.05",
    "0p1": "0.10",
    "0p25": "0.25",
    "0p5": "0.50",
    "1": "1.00",
}
SITE_EXPERIMENT = {
    "pohang": "pohang",
    "utah_2019": "base_utah_2019",
    "utah_2023": "base_utah_2023",
}
BRANCHES = [
    ("anomaly", "anomaly_metrics_fixed_threshold"),
    ("fc", "fc_metrics_fixed_threshold"),
    ("or", "or_metrics_fixed_threshold"),
    ("and", "and_metrics_fixed_threshold"),
]

EXPERIMENTS = {
    ("logenv", "in_domain"): (
        "experiment_1_log_env",
        "Experiment 1: log env",
    ),
    ("filter_rms", "in_domain"): (
        "experiment_2_no_log_env",
        "Experiment 2: no log env",
    ),
    ("logenv", "cross_site_reconst"): (
        "experiment_3_log_env_cross_site_transfer",
        "Experiment 3: log env cross-site transfer",
    ),
    ("filter_rms", "cross_site_reconst"): (
        "experiment_4_filter_rms_cross_site_transfer",
        "Experiment 4: filter rms cross-site transfer",
    ),
}

EXPERIMENT_ORDER = [value[0] for value in EXPERIMENTS.values()]


def experiment_info(dataset_name: str, study: str) -> tuple[str, str]:
    return EXPERIMENTS[(dataset_name, study)]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def metric_value(metrics: dict, key: str):
    value = metrics.get(key)
    if value is None:
        return ""
    return value


def add_metric_rows(rows: list[dict], base: dict, summary_path: Path) -> None:
    data = load_json(summary_path)
    if not data:
        return
    for branch, json_key in BRANCHES:
        metrics = data.get(json_key)
        if not isinstance(metrics, dict):
            continue
        row = dict(base)
        row.update(
            {
                "branch": branch,
                "f1": metric_value(metrics, "f1"),
                "balanced_acc": metric_value(metrics, "balanced_acc"),
                "specificity": metric_value(metrics, "specificity"),
                "recall": metric_value(metrics, "recall"),
                "precision": metric_value(metrics, "precision"),
                "accuracy": metric_value(metrics, "acc"),
                "tp": metric_value(metrics, "tp"),
                "tn": metric_value(metrics, "tn"),
                "fp": metric_value(metrics, "fp"),
                "fn": metric_value(metrics, "fn"),
                "test_summary": str(summary_path.relative_to(ROOT)),
            }
        )
        rows.append(row)


def collect_in_domain(dataset_name: str, run_root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    status: list[dict] = []
    experiment_id, experiment_name = experiment_info(dataset_name, "in_domain")
    groups = [
        ("scratch", ["scratch"]),
        ("contrast", ["contrast"]),
        ("reconst_reconst_noanom", ["reconst", "reconst_noanom"]),
    ]
    for site, experiment in SITE_EXPERIMENT.items():
        for group, methods in groups:
            for method in methods:
                method_root = run_root / f"{site}_{group}" / method
                for frac in FRACTIONS:
                    exp_name = f"{experiment}__frac{frac}"
                    finetune_dir = method_root / "finetune" / exp_name
                    test_path = method_root / "test" / exp_name / "test_metrics_fixed_threshold.json"
                    fin_done = (finetune_dir / "best.pt").exists()
                    test_done = test_path.exists()
                    base = {
                        "experiment_id": experiment_id,
                        "experiment_name": experiment_name,
                        "study": "in_domain",
                        "dataset": dataset_name,
                        "site": site,
                        "source_site": site if method != "scratch" else "",
                        "target_site": site,
                        "direction": f"{site}_in_domain",
                        "method": method,
                        "fraction": FRACTION_LABEL[frac],
                        "fraction_tag": frac,
                        "finetune_done": fin_done,
                        "test_done": test_done,
                    }
                    status.append(dict(base))
                    if test_done:
                        add_metric_rows(rows, base, test_path)
    return rows, status


def collect_cross(dataset_name: str, run_root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    status: list[dict] = []
    experiment_id, experiment_name = experiment_info(dataset_name, "cross_site_reconst")
    pairs = [
        ("pohang", "utah_2019"),
        ("utah_2019", "pohang"),
        ("pohang", "utah_2023"),
        ("utah_2023", "pohang"),
        ("utah_2019", "utah_2023"),
        ("utah_2023", "utah_2019"),
    ]
    for source, target in pairs:
        pair = f"{source}_to_{target}"
        experiment = SITE_EXPERIMENT[target]
        for frac in FRACTIONS:
            exp_name = f"{experiment}__frac{frac}"
            method_root = run_root / pair / "reconst"
            finetune_dir = method_root / "finetune" / exp_name
            test_path = method_root / "test" / exp_name / "test_metrics_fixed_threshold.json"
            fin_done = (finetune_dir / "best.pt").exists()
            test_done = test_path.exists()
            base = {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "study": "cross_site_reconst",
                "dataset": dataset_name,
                "site": target,
                "source_site": source,
                "target_site": target,
                "direction": pair,
                "method": "reconst",
                "fraction": FRACTION_LABEL[frac],
                "fraction_tag": frac,
                "finetune_done": fin_done,
                "test_done": test_done,
            }
            status.append(dict(base))
            if test_done:
                add_metric_rows(rows, base, test_path)
    return rows, status


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_branch_rows(metric_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in metric_rows:
        key = (
            row["experiment_id"],
            row["study"],
            row["dataset"],
            row["direction"],
            row["method"],
            row["fraction_tag"],
        )
        grouped.setdefault(key, []).append(row)
    best_rows = []
    for group_rows in grouped.values():
        best = max(
            group_rows,
            key=lambda r: (
                float(r["balanced_acc"]) if r["balanced_acc"] != "" else -1.0,
                float(r["f1"]) if r["f1"] != "" else -1.0,
            ),
        )
        best_rows.append(best)
    return sorted(
        best_rows,
        key=lambda r: (r["experiment_id"], r["direction"], r["method"], float(r["fraction"])),
    )


def fc_rows(metric_rows: list[dict]) -> list[dict]:
    return sorted(
        [r for r in metric_rows if r["branch"] == "fc"],
        key=lambda r: (r["experiment_id"], r["direction"], r["method"], float(r["fraction"])),
    )


def completion_table(status_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in status_rows:
        key = (
            row["experiment_id"],
            row["experiment_name"],
            row["study"],
            row["dataset"],
            row["direction"],
            row["method"],
        )
        grouped.setdefault(key, []).append(row)
    out = []
    for (experiment_id, experiment_name, study, dataset, direction, method), rows in sorted(grouped.items()):
        out.append(
            {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "study": study,
                "dataset": dataset,
                "direction": direction,
                "method": method,
                "finetune_complete": sum(bool(r["finetune_done"]) for r in rows),
                "test_complete": sum(bool(r["test_done"]) for r in rows),
                "total": len(rows),
                "missing_test_fractions": ";".join(r["fraction"] for r in rows if not r["test_done"]),
            }
        )
    return out


def safe_float(value) -> float | None:
    if value == "" or value is None:
        return None
    return float(value)


def metric_summary(rows: list[dict], dataset: str, study: str, direction: str | None = None) -> str:
    sub = [r for r in rows if r["dataset"] == dataset and r["study"] == study and r["branch"] == "fc"]
    if direction:
        sub = [r for r in sub if r["direction"] == direction]
    vals = [safe_float(r["balanced_acc"]) for r in sub]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "no completed FC metrics yet"
    return f"FC balanced accuracy range {min(vals):.3f}-{max(vals):.3f}, mean {mean(vals):.3f}"


def metric_summary_by_experiment(rows: list[dict], experiment_id: str) -> str:
    sub = [r for r in rows if r["experiment_id"] == experiment_id and r["branch"] == "fc"]
    vals = [safe_float(r["balanced_acc"]) for r in sub]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "no completed FC metrics yet"
    return f"FC balanced accuracy range {min(vals):.3f}-{max(vals):.3f}, mean {mean(vals):.3f}"


def write_markdown(path: Path, metric_rows: list[dict], status_rows: list[dict]) -> None:
    completion = completion_table(status_rows)
    fc = fc_rows(metric_rows)
    best = best_branch_rows(metric_rows)

    def line_for_completion(experiment_id: str) -> str:
        sub = [r for r in completion if r["experiment_id"] == experiment_id]
        tests = sum(int(r["test_complete"]) for r in sub)
        total = sum(int(r["total"]) for r in sub)
        fins = sum(int(r["finetune_complete"]) for r in sub)
        return f"finetune {fins}/{total}, test {tests}/{total}"

    def top_fc(experiment_id: str, n: int = 8) -> list[str]:
        sub = [r for r in fc if r["experiment_id"] == experiment_id]
        sub = sorted(
            sub,
            key=lambda r: (
                safe_float(r["balanced_acc"]) or -1,
                safe_float(r["f1"]) or -1,
            ),
            reverse=True,
        )[:n]
        return [
            f"- `{r['direction']}` `{r['method']}` frac `{r['fraction']}`: F1={float(r['f1']):.3f}, BalAcc={float(r['balanced_acc']):.3f}, Spec={float(r['specificity']):.3f}"
            for r in sub
        ]

    missing = [r for r in completion if r["missing_test_fractions"]]
    lines = [
        "# Current Results Summary",
        "",
        "Generated from completed `test_metrics_fixed_threshold.json` files. FC metrics are the primary supervised classifier readout; branch-level metrics are also saved in CSV.",
        "",
        "## Experiment Map",
        "",
        "- Experiment 1: `log env` in-domain study.",
        "- Experiment 2: `no log env` in-domain study, implemented by the filter + RMS dataset.",
        "- Experiment 3: `log env` cross-site transfer, using reconst pretraining checkpoints.",
        "- Experiment 4: `filter rms` cross-site transfer, using reconst pretraining checkpoints.",
        "",
        "## Completion",
        "",
        f"- Experiment 1: {line_for_completion('experiment_1_log_env')}",
        f"- Experiment 2: {line_for_completion('experiment_2_no_log_env')}",
        f"- Experiment 3: {line_for_completion('experiment_3_log_env_cross_site_transfer')}",
        f"- Experiment 4: {line_for_completion('experiment_4_filter_rms_cross_site_transfer')}",
        "",
        "## Experiment-Level Takeaways",
        "",
        "- Experiment 1 is the complete log-envelope in-domain baseline.",
        "- Experiment 2 is the no-log/filter-RMS in-domain comparison; check completion status before using `Pohang contrast` because it has been the main missing block.",
        "- Experiment 3 is the complete log-envelope cross-site transfer result across all six source-target directions and five label fractions.",
        "- Experiment 4 is the filter-RMS cross-site transfer result; if completion is below 30/30, treat it as interim.",
        "- For interpretation, do not rely on F1 alone. Several runs show high F1 with collapsed specificity, especially in low-label or shifted settings.",
        "",
        "## Metric Ranges",
        "",
        f"- Experiment 1: {metric_summary_by_experiment(metric_rows, 'experiment_1_log_env')}",
        f"- Experiment 2: {metric_summary_by_experiment(metric_rows, 'experiment_2_no_log_env')}",
        f"- Experiment 3: {metric_summary_by_experiment(metric_rows, 'experiment_3_log_env_cross_site_transfer')}",
        f"- Experiment 4: {metric_summary_by_experiment(metric_rows, 'experiment_4_filter_rms_cross_site_transfer')}",
        "",
        "## Best Completed FC Rows",
        "",
        "### Experiment 1: Log Env",
        *top_fc("experiment_1_log_env"),
        "",
        "### Experiment 2: No Log Env",
        *top_fc("experiment_2_no_log_env"),
        "",
        "### Experiment 3: Log Env Cross-Site Transfer",
        *top_fc("experiment_3_log_env_cross_site_transfer"),
        "",
        "### Experiment 4: Filter RMS Cross-Site Transfer",
        *top_fc("experiment_4_filter_rms_cross_site_transfer"),
        "",
        "## Missing Tests",
        "",
    ]
    if missing:
        for row in missing:
            lines.append(
                f"- {row['experiment_name']} `{row['direction']}` `{row['method']}` missing fractions: {row['missing_test_fractions']}"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `all_metrics_long.csv`: all completed metrics for anomaly/fc/or/and branches.",
            "- `fc_metrics.csv`: completed FC-only metrics, one row per completed run.",
            "- `best_branch_metrics.csv`: best branch per completed run by balanced accuracy, then F1.",
            "- `completion_status.csv`: finetune/test completion counts and missing fractions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("logenv", "in_domain", ROOT / "runs" / "visualbest_filter_logenv_rms_fs1000_rms0p15_lp50_log1_sm1x0p5_site_main_pre50_v1"),
        ("filter_rms", "in_domain", ROOT / "runs" / "visualbest_filter_rms_fs1000_rms0p15_lp50_site_main_pre50_v1"),
        ("logenv", "cross_site_reconst", ROOT / "runs" / "logenv_cross_site_reconst_pre50_v1"),
        ("filter_rms", "cross_site_reconst", ROOT / "runs" / "filter_rms_cross_site_reconst_pre50_v1"),
    ]
    metric_rows: list[dict] = []
    status_rows: list[dict] = []
    for dataset, study, run_root in specs:
        if not run_root.exists():
            continue
        if study == "in_domain":
            rows, status = collect_in_domain(dataset, run_root)
        else:
            rows, status = collect_cross(dataset, run_root)
        metric_rows.extend(rows)
        status_rows.extend(status)

    metric_fields = [
        "experiment_id",
        "experiment_name",
        "study",
        "dataset",
        "site",
        "source_site",
        "target_site",
        "direction",
        "method",
        "fraction",
        "fraction_tag",
        "finetune_done",
        "test_done",
        "branch",
        "f1",
        "balanced_acc",
        "specificity",
        "recall",
        "precision",
        "accuracy",
        "tp",
        "tn",
        "fp",
        "fn",
        "test_summary",
    ]
    status_fields = [
        "experiment_id",
        "experiment_name",
        "study",
        "dataset",
        "site",
        "source_site",
        "target_site",
        "direction",
        "method",
        "fraction",
        "fraction_tag",
        "finetune_done",
        "test_done",
    ]
    completion_fields = [
        "experiment_id",
        "experiment_name",
        "study",
        "dataset",
        "direction",
        "method",
        "finetune_complete",
        "test_complete",
        "total",
        "missing_test_fractions",
    ]

    write_csv(OUT_DIR / "all_metrics_long.csv", metric_rows, metric_fields)
    write_csv(OUT_DIR / "run_status_long.csv", status_rows, status_fields)
    write_csv(OUT_DIR / "completion_status.csv", completion_table(status_rows), completion_fields)
    write_csv(OUT_DIR / "fc_metrics.csv", fc_rows(metric_rows), metric_fields)
    write_csv(OUT_DIR / "best_branch_metrics.csv", best_branch_rows(metric_rows), metric_fields)
    write_markdown(OUT_DIR / "summary.md", metric_rows, status_rows)
    print(f"[DONE] wrote results summary to {OUT_DIR}")
    print(f"[INFO] metric rows: {len(metric_rows)}")
    print(f"[INFO] status rows: {len(status_rows)}")


if __name__ == "__main__":
    main()
