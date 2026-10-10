from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from realwaste.config import load_config, save_resolved_config
from realwaste.constants import CLASS_NAMES
from realwaste.data.dataloader import build_dataloaders
from realwaste.inference.history import validate_feedback_snapshot
from realwaste.models import create_model, set_backbone_trainable
from realwaste.seed import resolve_device, seed_everything
from realwaste.training.progress import ProgressReporter
from realwaste.training.trainer import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RealWaste classifier")
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path)
    mode.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feedback-manifest", type=Path)
    parser.add_argument(
        "--progress-path",
        type=Path,
        help="Write the latest epoch progress as atomic JSON for an external UI",
    )
    return parser.parse_args()


def prepare_output_directory(
    output_dir: str | Path,
    *,
    resume: bool,
    overwrite: bool,
) -> Path:
    """Prevent accidental checkpoint/history replacement for a fresh training run."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = [output / name for name in ("config.yaml", "best.pt", "last.pt", "history.csv")]
    existing = [path for path in artifacts if path.exists()]
    if resume:
        return output
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Experiment output already contains {names}; use --resume or explicit --overwrite"
        )
    if overwrite:
        for path in artifacts:
            path.unlink(missing_ok=True)
        for directory in (output / "evaluation", output / "figures"):
            if directory.exists():
                shutil.rmtree(directory)
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.resume is not None and args.feedback_manifest is not None:
        raise ValueError("Resume phải dùng bản chụp phản hồi của phiên cũ")
    if args.resume is not None:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        if feedback := previous["config"].get("feedback"):
            config["feedback"] = validate_feedback_snapshot(feedback["manifest"], Path.cwd())
            if config["feedback"]["fingerprint"] != feedback["fingerprint"]:
                raise ValueError("Bản chụp phản hồi không khớp checkpoint")
    elif args.feedback_manifest is not None:
        config["feedback"] = validate_feedback_snapshot(args.feedback_manifest, Path.cwd())
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "auto")))
    split_metadata_path = Path(config["data"]["train_csv"]).parent / "split_metadata.json"
    if not split_metadata_path.is_file():
        raise FileNotFoundError(f"Split metadata not found: {split_metadata_path}")
    split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))

    model_config = config["model"]
    model = create_model(
        model_config["name"],
        int(model_config["num_classes"]),
        pretrained=bool(model_config.get("pretrained", True)),
    )
    mode = model_config.get("fine_tune_mode")
    if mode is None:
        mode = "last_block" if model_config.get("train_backbone", False) else "head"
    set_backbone_trainable(model, mode)

    output_dir = prepare_output_directory(
        config["output"]["directory"],
        resume=args.resume is not None,
        overwrite=args.overwrite,
    )
    save_resolved_config(config, output_dir / "config.yaml")
    loaders = build_dataloaders(config)
    progress_reporter = ProgressReporter(args.progress_path) if args.progress_path else None
    result = train_model(
        model,
        loaders,
        config,
        device,
        class_names=CLASS_NAMES,
        split_fingerprint=split_metadata["fingerprint"],
        resume_path=args.resume,
        progress_callback=progress_reporter,
    )
    print(f"Training complete on {device}: {result}")


if __name__ == "__main__":
    main()
