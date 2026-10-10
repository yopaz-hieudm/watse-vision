from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch

from realwaste.config import load_config, save_resolved_config
from realwaste.constants import CLASS_NAMES, DEFAULT_DATASET_ROOT
from realwaste.data.download import validate_dataset_structure, verify_checksum
from realwaste.data.split import compute_split_fingerprint
from realwaste.evaluation.gradcam import GradCAM, save_gradcam_figure
from realwaste.inference import Predictor
from realwaste.inference.history import PredictionHistory
from realwaste.training.progress import atomic_write_json
from realwaste.ui.experiment_registry import EXPERIMENTS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(arguments: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
    if result.returncode:
        raise RuntimeError(f"Lệnh thất bại; xem {log}: {log.read_text(encoding='utf-8')[-2000:]}")


def verify() -> Path:
    """Exercise real scripts on a small, disjoint sample of the verified UCI dataset."""
    dataset = PROJECT_ROOT / DEFAULT_DATASET_ROOT
    torch.set_num_threads(2)
    verify_checksum(
        PROJECT_ROOT / "data/downloads/realwaste.zip",
        PROJECT_ROOT / "data/checksums/realwaste.sha256",
    )
    counts = validate_dataset_structure(dataset)
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    output = PROJECT_ROOT / "outputs/verification" / run_id
    output.mkdir(parents=True)
    # Full EDA is visible in the app; the small training run stays separate from E1–E4.
    run_command(
        [
            "scripts/inspect_data.py",
            "--data-dir",
            str(dataset),
            "--output-dir",
            "outputs/eda",
        ],
        output / "eda.log",
    )

    full_splits = output / "full_splits"
    run_command(
        [
            "scripts/create_splits.py",
            "--data-dir",
            str(dataset),
            "--output-dir",
            str(full_splits),
        ],
        output / "split.log",
    )

    splits = output / "splits"
    splits.mkdir()
    for name in ("train", "val", "test"):
        original = pd.read_csv(full_splits / f"{name}.csv")
        frame = original.groupby("class_name", sort=True).head(4 if name == "train" else 2)
        if set(frame["class_name"]) != set(CLASS_NAMES):
            raise ValueError("Split kiểm chứng phải có đủ 9 lớp")
        frame.to_csv(splits / f"{name}.csv", index=False)
    fingerprint = compute_split_fingerprint(
        [splits / f"{name}.csv" for name in ("train", "val", "test")]
    )
    atomic_write_json(splits / "split_metadata.json", {"fingerprint": fingerprint})
    configs = output / "configs"
    configs.mkdir()
    checkpoints = {}
    report = {
        "status": "running",
        "scope": "real-data smoke test; not a model quality benchmark",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_images": sum(counts.values()),
        "train_samples": 36,
        "validation_samples": 18,
        "test_samples": 18,
        "epochs": 1,
        "pretrained": False,
        "image_size": 64,
        "steps": ["verify dataset checksum and class counts", "full EDA", "create disjoint splits"],
    }
    atomic_write_json(output / "report.json", report)
    for experiment_id, config_relative in EXPERIMENTS.items():
        config = load_config(PROJECT_ROOT / config_relative)
        config["device"] = "cpu"
        config["training"]["epochs"] = 1
        config["training"]["mixed_precision"] = False
        config["model"]["pretrained"] = False
        config["data"].update(
            {
                "root": str(dataset),
                "train_csv": str(splits / "train.csv"),
                "val_csv": str(splits / "val.csv"),
                "test_csv": str(splits / "test.csv"),
                "image_size": 64,
                "resize_size": 72,
                "batch_size": 18,
                "num_workers": 0,
            }
        )
        experiment_dir = output / "experiments" / config["experiment_name"]
        config["output"]["directory"] = str(experiment_dir)
        config_path = save_resolved_config(config, configs / Path(config_relative).name)
        checkpoint = experiment_dir / "best.pt"
        run_command(
            ["scripts/train.py", "--config", str(config_path)],
            output / f"{experiment_id}_train.log",
        )
        for split in ("validation", "test"):
            run_command(
                [
                    "scripts/evaluate.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    split,
                ],
                output / f"{experiment_id}_{split}.log",
            )
        checkpoints[experiment_id] = checkpoint
        report["steps"].append(f"{experiment_id}: train → validation → test")
        atomic_write_json(output / "report.json", report)
        print(f"Verified {experiment_id}", flush=True)
    run_command(
        [
            "scripts/compare_experiments.py",
            "--input-dir",
            str(output / "experiments"),
            "--output-dir",
            str(output / "comparison"),
        ],
        output / "comparison.log",
    )
    report["steps"].append("compare E1–E4")
    predictor = Predictor(checkpoints["E1"], device="cpu")
    # Pick a train image so the feedback guard can prove held-out images stay excluded.
    row = pd.read_csv(splits / "train.csv").iloc[0]
    image = dataset / row["path"]
    predictions = predictor.predict(image)
    history = PredictionHistory(PROJECT_ROOT, storage_path=output / "predictions/history.sqlite3")
    identifier = history.save(
        experiment_id="E1",
        checkpoint=checkpoints["E1"],
        filename=image.name,
        image=image.read_bytes(),
        predictions=predictions,
        elapsed_ms=0,
        retain_image=True,
    )
    history.review(identifier, row["class_name"])
    report["steps"].append("predict → persist history → review label")
    with GradCAM(predictor.model) as gradcam:
        heatmap, _predicted, _probabilities = gradcam.generate(predictor.prepare_image(image))
        save_gradcam_figure(image, heatmap, output / "gradcam.png", title="Lifecycle verification")
    report["steps"].append("Grad-CAM")
    snapshot_config = load_config(configs / Path(EXPERIMENTS["E1"]).name)
    manifest = history.create_training_snapshot(split_directory=full_splits)
    snapshot_config["output"]["directory"] = str(output / "feedback_training")
    feedback_config = save_resolved_config(snapshot_config, output / "feedback_config.yaml")
    run_command(
        [
            "scripts/train.py",
            "--config",
            str(feedback_config),
            "--feedback-manifest",
            str(manifest),
        ],
        output / "feedback_train.log",
    )
    checkpoint = torch.load(
        output / "feedback_training/best.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["config"]["feedback"]["count"] == 1
    report["steps"].append("reviewed feedback snapshot → new training")
    run_command(
        [
            "scripts/train.py",
            "--config",
            str(feedback_config),
            "--resume",
            str(output / "feedback_training/last.pt"),
        ],
        output / "feedback_resume.log",
    )
    report["steps"].append("resume using the same feedback snapshot")
    run_command(
        [
            "scripts/evaluate.py",
            "--checkpoint",
            str(output / "feedback_training/best.pt"),
            "--split",
            "test",
        ],
        output / "feedback_evaluate.log",
    )
    report["steps"].append("evaluate feedback-trained model on unchanged Test samples")
    report["status"] = "passed"
    report["finished_at"] = datetime.now(UTC).isoformat()
    report["checkpoint_sha256"] = hashlib.sha256(checkpoints["E1"].read_bytes()).hexdigest()
    atomic_write_json(output / "report.json", report)
    atomic_write_json(
        PROJECT_ROOT / "outputs/verification/latest.json",
        {
            **report,
            "report_path": str((output / "report.json").relative_to(PROJECT_ROOT)),
        },
    )
    return output / "report.json"


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Verify the lifecycle with a small sample of real data"
    ).parse_args()
    print(verify())
