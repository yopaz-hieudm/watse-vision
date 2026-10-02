from __future__ import annotations

from pathlib import Path

import pytest
import torch

import realwaste.data.download as download_module
from realwaste.training.checkpoint import validate_resume_checkpoint
from scripts.generate_gradcam import select_gradcam_samples
from scripts.train import prepare_output_directory


def test_prepare_output_directory_requires_explicit_mode(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    output.mkdir()
    for name in ("config.yaml", "best.pt", "last.pt", "history.csv"):
        (output / name).write_text("existing", encoding="utf-8")
    (output / "evaluation").mkdir()
    (output / "evaluation" / "stale.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume or explicit --overwrite"):
        prepare_output_directory(output, resume=False, overwrite=False)

    prepare_output_directory(output, resume=True, overwrite=False)
    assert (output / "last.pt").is_file()

    prepare_output_directory(output, resume=False, overwrite=True)
    assert not any((output / name).exists() for name in ("best.pt", "last.pt", "history.csv"))
    assert not (output / "evaluation").exists()


def test_validate_resume_checkpoint_checks_identity_and_split(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    payload = {
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "epoch": 2,
        "best_metric": 0.8,
        "config": {"experiment_id": "E1"},
        "class_names": ["a", "b"],
        "model_name": "mobilenet_v3_small",
        "weights_id": "none",
        "split_fingerprint": "same-split",
    }
    torch.save(payload, checkpoint)

    loaded = validate_resume_checkpoint(
        checkpoint,
        expected_split_fingerprint="same-split",
        expected_experiment_id="E1",
        expected_model_name="mobilenet_v3_small",
    )
    assert loaded["epoch"] == 2

    with pytest.raises(ValueError, match="different split fingerprint"):
        validate_resume_checkpoint(
            checkpoint,
            expected_split_fingerprint="other-split",
            expected_experiment_id="E1",
            expected_model_name="mobilenet_v3_small",
        )

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="cannot be loaded"):
        validate_resume_checkpoint(
            corrupt,
            expected_split_fingerprint="same-split",
            expected_experiment_id="E1",
            expected_model_name="mobilenet_v3_small",
        )


def test_download_skips_archive_when_dataset_is_already_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "RealWaste"
    dataset.mkdir()
    archive = tmp_path / "missing.zip"
    monkeypatch.setattr(download_module, "validate_dataset_structure", lambda _path: {})

    def unexpected_download(*_args, **_kwargs):
        raise AssertionError("valid extracted dataset must not be downloaded again")

    monkeypatch.setattr(download_module, "download_archive", unexpected_download)
    result = download_module.download_and_extract(
        archive,
        tmp_path / "checksum.sha256",
        dataset,
    )
    assert result == dataset


def test_gradcam_sample_selection_covers_all_required_groups() -> None:
    rows = [
        {
            "path": f"correct-{index}.jpg",
            "correct": "true",
            "confidence": str(confidence),
        }
        for index, confidence in enumerate((0.99, 0.9, 0.7, 0.4, 0.2, 0.05))
    ]
    rows.extend(
        {
            "path": f"wrong-{index}.jpg",
            "correct": "false",
            "confidence": str(confidence),
        }
        for index, confidence in enumerate((0.95, 0.6, 0.1))
    )

    selected = select_gradcam_samples(rows, 6)
    groups = {group for group, _row in selected}
    assert groups == {"correct_high", "correct_low", "wrong_high"}
    by_group = {}
    for group, row in selected:
        by_group.setdefault(group, []).append(float(row["confidence"]))
    assert max(by_group["correct_high"]) == 0.99
    assert min(by_group["correct_low"]) == 0.05
    assert max(by_group["wrong_high"]) == 0.95
