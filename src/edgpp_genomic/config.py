from __future__ import annotations
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(path: str | Path) -> DictConfig:
    return OmegaConf.load(str(path))


def save_config(cfg: DictConfig, path: str | Path) -> None:
    OmegaConf.save(cfg, str(path))
