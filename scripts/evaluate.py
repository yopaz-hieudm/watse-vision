from __future__ import annotations

import argparse
from pathlib import Path

import torch

from realwaste.data.dataloader import build_dataloaders
from realwaste.evaluation.confusion_matrix import save_confusion_matrix
from realwaste.evaluation.curves import save_training_curves
from realwaste.evaluation.error_analysis import save_top_confusions, top_confusions
from realwaste.evaluation.runner import evaluate_model, write_evaluation_outputs
from realwaste.models import create_model
from realwaste.seed import resolve_device
from realwaste.training.trainer import create_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RealWaste checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Result directory (default: CHECKPOINT_DIR/evaluation/SPLIT)",
    )
    return parser.parse_args()


def resolve_output_dir(
    checkpoint_path: str | Path,
    split: str,
    output_dir: str | Path | None = None,
) -> Path:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    if output_dir is not None:
        return Path(output_dir)
    return Path(checkpoint_path).parent / "evaluation" / split


def resolve_history_path(checkpoint_path: str | Path) -> Path:
    """Return the training history stored alongside its checkpoint."""
    return Path(checkpoint_path).parent / "history.csv"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    class_names = checkpoint["class_names"]
    model = create_model(checkpoint["model_name"], len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loaders = build_dataloaders(config)
    result = evaluate_model(
        model,
        loaders[args.split],
        device,
        criterion=create_loss(config),
    )
    output_dir = resolve_output_dir(args.checkpoint, args.split, args.output_dir)
    augmentation = bool(config["data"]["augmentation"]["enabled"])
    metadata = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "model": checkpoint["model_name"],
        "augmentation": augmentation,
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "checkpoint_path": str(args.checkpoint),
        "split": args.split,
        "split_fingerprint": checkpoint["split_fingerprint"],
    }
    metrics = write_evaluation_outputs(
        result,
        output_dir,
        class_names,
        metadata=metadata,
    )
    figures = output_dir / "figures"
    save_confusion_matrix(
        result["y_true"], result["y_pred"], class_names, figures / "confusion_matrix.png"
    )
    save_confusion_matrix(
        result["y_true"],
        result["y_pred"],
        class_names,
        figures / "confusion_matrix_normalized.png",
        normalize=True,
    )
    save_top_confusions(
        top_confusions(result["y_true"], result["y_pred"], class_names),
        output_dir / "top_confusions.csv",
    )
    history_path = resolve_history_path(args.checkpoint)
    if history_path.is_file():
        save_training_curves(history_path, figures / "training_curves.png")
    print(metrics)


if __name__ == "__main__":
    main()
