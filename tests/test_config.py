from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visu_predict.config import TrainingConfig, load_config, setup_directories


def test_config_defaults_construct():
    cfg = TrainingConfig(base_output_dir="./tmp")
    assert cfg.dataset_name == "METR-LA"
    assert cfg.hidden_dim % cfg.num_heads == 0
    assert cfg.hidden_dim % 2 == 0


def test_config_adjusts_hidden_dim_for_num_heads():
    cfg = TrainingConfig(base_output_dir="./tmp", hidden_dim=100, num_heads=16)
    assert cfg.hidden_dim % cfg.num_heads == 0
    assert cfg.hidden_dim % 2 == 0


def test_config_rejects_invalid_loss():
    with pytest.raises(ValueError, match="loss_function"):
        TrainingConfig(base_output_dir="./tmp", loss_function="nope")


def test_config_rejects_invalid_decoder():
    with pytest.raises(ValueError, match="decoder_type"):
        TrainingConfig(base_output_dir="./tmp", decoder_type="rnn")


def test_load_config_from_yaml(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump({"base_output_dir": str(tmp_path), "num_epochs": 3}))
    cfg = load_config(cfg_path)
    assert cfg.num_epochs == 3
    assert cfg.base_output_dir == str(tmp_path)


def test_setup_directories_creates_paths(tmp_path: Path):
    cfg = TrainingConfig(base_output_dir=str(tmp_path))
    setup_directories(cfg, run_name="test_run")
    assert Path(cfg.output_dir).exists()
    assert Path(cfg.model_dir).exists()
    assert Path(cfg.results_dir).exists()
    assert Path(cfg.input_dir).exists()
