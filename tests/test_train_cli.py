from pathlib import Path

import pytest
import torch

from realwaste.config import load_config, save_resolved_config
from scripts import train


@pytest.mark.parametrize("resume", [False, True])
def test_epoch_override_reaches_trainer_and_saved_config(tmp_path, monkeypatch, resume):
    source = Path(__file__).resolve().parents[1] / "configs/resnet18_aug.yaml"
    config = load_config(source)
    config["output"]["directory"] = str(tmp_path / "run")
    config["data"]["train_csv"] = str(tmp_path / "train.csv")
    (tmp_path / "split_metadata.json").write_text('{"fingerprint": "test"}', encoding="utf-8")
    config_path = save_resolved_config(config, tmp_path / "input.yaml")
    command = ["train.py", "--config", str(config_path), "--epochs", "12"]
    if resume:
        checkpoint = tmp_path / "last.pt"
        torch.save({"config": config}, checkpoint)
        command.extend(["--resume", str(checkpoint)])
    monkeypatch.setattr("sys.argv", command)
    monkeypatch.setattr(train, "create_model", lambda *_args, **_kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(train, "set_backbone_trainable", lambda *_args: None)
    monkeypatch.setattr(train, "build_dataloaders", lambda _config: {})
    received = []

    def capture_training(_model, _loaders, run_config, _device, **_kwargs):
        received.append(run_config["training"]["epochs"])
        return {}

    monkeypatch.setattr(train, "train_model", capture_training)
    train.main()
    assert received == [12]
    assert load_config(tmp_path / "run/config.yaml")["training"]["epochs"] == 12
    assert load_config(config_path)["training"]["epochs"] == 30


@pytest.mark.parametrize("value", ["0", "-1", "1.5"])
def test_invalid_epoch_override_is_rejected(monkeypatch, value):
    monkeypatch.setattr("sys.argv", ["train.py", "--config", "unused.yaml", "--epochs", value])
    with pytest.raises(SystemExit) as error:
        train.parse_args()
    assert error.value.code == 2
