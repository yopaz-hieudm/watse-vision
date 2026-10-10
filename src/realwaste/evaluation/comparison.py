from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_EXPERIMENTS = {"E1", "E2", "E3", "E4"}
SUMMARY_FIELDS = [
    "experiment_id",
    "model",
    "augmentation",
    "best_epoch",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "checkpoint_path",
    "split_fingerprint",
]


def collect_experiment_metrics(input_dir: str | Path) -> list[dict]:
    input_dir = Path(input_dir)
    rows: dict[str, dict] = {}
    for experiment_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        test_metrics_path = experiment_dir / "evaluation" / "test" / "metrics.json"
        legacy_metrics_path = experiment_dir / "metrics.json"
        metrics_path = test_metrics_path if test_metrics_path.is_file() else legacy_metrics_path
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        # Legacy root artifacts may omit the split, so preserve that fallback. Any artifact
        # that explicitly identifies itself as Validation must never enter the comparison.
        split = metrics.get("split")
        if split is not None and str(split).lower() != "test":
            raise ValueError(f"{metrics_path} is not Test metrics")
        experiment_id = metrics.get("experiment_id")
        if experiment_id in REQUIRED_EXPERIMENTS:
            if experiment_id in rows:
                raise ValueError(f"Duplicate metrics for {experiment_id}")
            missing = [field for field in SUMMARY_FIELDS if field not in metrics]
            if missing:
                raise ValueError(f"{metrics_path} is missing fields: {missing}")
            config_path = experiment_dir / "config.yaml"
            history_path = experiment_dir / "history.csv"
            if not config_path.is_file() or not history_path.is_file():
                raise ValueError(
                    f"{experiment_id} requires config.yaml and history.csv for comparison"
                )
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError(f"{config_path} must contain a mapping")
            if str(config.get("experiment_id", "")).upper() != experiment_id:
                raise ValueError(f"{config_path} has a mismatched experiment_id")
            if str(config.get("model", {}).get("name")) != str(metrics["model"]):
                raise ValueError(f"{config_path} model does not match metrics")
            configured_augmentation = bool(
                config.get("data", {}).get("augmentation", {}).get("enabled")
            )
            if configured_augmentation != bool(metrics["augmentation"]):
                raise ValueError(f"{config_path} augmentation does not match metrics")

            with history_path.open("r", encoding="utf-8", newline="") as handle:
                history_rows = list(csv.DictReader(handle))
            try:
                history_epochs = {int(row["epoch"]) for row in history_rows}
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{history_path} has invalid epoch data") from error
            if int(metrics["best_epoch"]) not in history_epochs:
                raise ValueError(f"{metrics_path} best_epoch is not present in history.csv")

            row = {field: metrics[field] for field in SUMMARY_FIELDS}
            row["_seed"] = config.get("seed")
            row["_test_csv"] = str(config.get("data", {}).get("test_csv", ""))
            row["_feedback_fingerprint"] = config.get("feedback", {}).get("fingerprint")
            rows[experiment_id] = row
    missing_experiments = REQUIRED_EXPERIMENTS - set(rows)
    if missing_experiments:
        raise ValueError(f"Missing experiments: {sorted(missing_experiments)}")
    fingerprints = {row["split_fingerprint"] for row in rows.values()}
    if len(fingerprints) != 1:
        raise ValueError("E1-E4 do not use the same split fingerprint")
    seeds = {row["_seed"] for row in rows.values()}
    if len(seeds) != 1:
        raise ValueError("E1-E4 do not use the same seed")
    test_csv_paths = {row["_test_csv"] for row in rows.values()}
    if len(test_csv_paths) != 1 or not next(iter(test_csv_paths)):
        raise ValueError("E1-E4 do not use the same Test CSV")
    if len({row["_feedback_fingerprint"] for row in rows.values()}) != 1:
        raise ValueError("E1-E4 do not use the same reviewed feedback snapshot")
    return [
        {field: row[field] for field in SUMMARY_FIELDS}
        for row in sorted(rows.values(), key=lambda row: row["experiment_id"])
    ]


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(rows: list[dict], path: Path) -> None:
    columns = [
        "experiment_id",
        "model",
        "augmentation",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_augmentation_impact(rows: list[dict], path: Path) -> None:
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[bool(row["augmentation"])] = row
    fields = [
        "model",
        "baseline_macro_f1",
        "augmented_macro_f1",
        "macro_f1_delta",
        "baseline_accuracy",
        "augmented_accuracy",
        "accuracy_delta",
    ]
    output_rows = []
    for model, variants in sorted(by_model.items()):
        if set(variants) != {False, True}:
            raise ValueError(f"Model {model} must have baseline and augmentation results")
        baseline, augmented = variants[False], variants[True]
        output_rows.append(
            {
                "model": model,
                "baseline_macro_f1": baseline["macro_f1"],
                "augmented_macro_f1": augmented["macro_f1"],
                "macro_f1_delta": augmented["macro_f1"] - baseline["macro_f1"],
                "baseline_accuracy": baseline["accuracy"],
                "augmented_accuracy": augmented["accuracy"],
                "accuracy_delta": augmented["accuracy"] - baseline["accuracy"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)


def _save_comparison_chart(rows: list[dict], path: Path) -> None:
    labels = [row["experiment_id"] for row in rows]
    positions = range(len(rows))
    width = 0.35
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        [value - width / 2 for value in positions],
        [r["accuracy"] for r in rows],
        width,
        label="Accuracy",
    )
    axis.bar(
        [value + width / 2 for value in positions],
        [r["macro_f1"] for r in rows],
        width,
        label="Macro-F1",
    )
    axis.set_xticks(list(positions), labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("RealWaste experiment comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def compare_experiments(input_dir: str | Path, output_dir: str | Path) -> list[dict]:
    rows = collect_experiment_metrics(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(rows, output_dir / "experiment_summary.csv")
    _write_summary_markdown(rows, output_dir / "experiment_summary.md")
    _write_augmentation_impact(rows, output_dir / "augmentation_impact.csv")
    _save_comparison_chart(rows, output_dir / "metrics_comparison.png")
    return sorted(rows, key=lambda row: row["macro_f1"], reverse=True)
