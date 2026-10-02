from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from realwaste.evaluation.metrics import compute_classification_metrics
from realwaste.training.callbacks import EarlyStopping
from realwaste.training.checkpoint import load_checkpoint, save_checkpoint


def create_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = config["training"]["optimizer"]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    if optimizer_config["name"].lower() != "adamw":
        raise ValueError("only AdamW is supported")
    return torch.optim.AdamW(
        trainable,
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )


def create_loss(config: dict[str, Any]) -> nn.Module:
    loss_config = config["training"]["loss"]
    if loss_config["name"].lower() != "cross_entropy":
        raise ValueError("only CrossEntropyLoss is supported")
    return nn.CrossEntropyLoss(label_smoothing=float(loss_config.get("label_smoothing", 0.0)))


def create_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    scheduler_config = config["training"]["scheduler"]
    if scheduler_config["name"].lower() != "reduce_lr_on_plateau":
        raise ValueError("only ReduceLROnPlateau is supported")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config.get("mode", "max"),
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 2)),
        min_lr=float(scheduler_config.get("min_learning_rate", 1e-6)),
    )


def _run_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = False,
    scaler: Any = None,
) -> dict[str, float | int | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels, _paths in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())

    average_loss = total_loss / max(sample_count, 1)
    return compute_classification_metrics(y_true, y_pred, loss=average_loss)


def _append_history(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_model(
    model: nn.Module,
    loaders: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    *,
    class_names: list[str] | tuple[str, ...],
    split_fingerprint: str,
    start_epoch: int = 0,
    resume_path: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = create_optimizer(model, config)
    criterion = create_loss(config)
    scheduler = create_scheduler(optimizer, config)
    early_config = config["training"]["early_stopping"]
    early_stopping = EarlyStopping(
        int(early_config.get("patience", 5)),
        mode=early_config.get("mode", "max"),
        minimum_delta=float(early_config.get("minimum_delta", 0.0)),
    )
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.to(device)

    best_metric = float("-inf")
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        if checkpoint.get("split_fingerprint") != split_fingerprint:
            raise ValueError("resume checkpoint uses a different split fingerprint")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        early_stopping.best = best_metric

    epochs = int(config["training"]["epochs"])
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs):
        train_metrics = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = _run_epoch(model, loaders["validation"], criterion, device)
        monitored = float(val_metrics["macro_f1"])
        scheduler.step(monitored)
        last_epoch = epoch

        history_row = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"]}
        history_row.update({f"train_{key}": value for key, value in train_metrics.items()})
        history_row.update({f"val_{key}": value for key, value in val_metrics.items()})
        _append_history(output_dir / "history.csv", history_row)

        if monitored > best_metric:
            best_metric = monitored
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                config=config,
                class_names=class_names,
                split_fingerprint=split_fingerprint,
            )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            config=config,
            class_names=class_names,
            split_fingerprint=split_fingerprint,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "train_loss": float(train_metrics["loss"]),
                    "train_accuracy": float(train_metrics["accuracy"]),
                    "train_macro_f1": float(train_metrics["macro_f1"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_macro_f1": monitored,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "best_metric": best_metric,
                }
            )
        if early_stopping.step(monitored):
            break

    return {"best_metric": best_metric, "last_epoch": last_epoch}
