"""YAML configuration loading, via Hydra/OmegaConf.

The CLI (`cli.py`) takes only a command and `--offline`; every other knob —
agent settings, maze side length, episode settings — lives in
`configs/config.yaml` (see `CONFIG_PATH`), which is also the single source of
defaults: nothing here re-declares them, so an omitted key surfaces as an
OmegaConf error at the point it's read rather than silently falling back to
something else.

Which CLI the agents drive is a Hydra config group: `defaults: - agent: claude`
composes `configs/agent/claude.yaml` (binary, model, effort) into `agent`, and
the primary config adds the agent-agnostic keys (timeout, retries) on top.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

#: Resolved relative to the current working directory, same convention as the
#: `mazes/` and `episodes/` output directories.
CONFIG_PATH = Path("configs/config.yaml")


def load_config(path: Path = CONFIG_PATH) -> DictConfig:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — see configs/ at the repo root for the expected shape")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(path.resolve().parent)):
        return compose(config_name=path.stem)
