"""What the `harness_2d` and `harness_3d` harnesses both run on.

Nothing here knows about mazes or superquadrics; a module earns its place in
this package by being needed, unchanged, by both siblings.

Layout:
  agents/       the Agent abstraction and its tmux CLI backends
  config        the Hydra/OmegaConf loader; each harness owns its own config path
  pipeline_log  one-line-per-event logging for unattended runs
"""

from .agents import (
    Agent,
    AgentError,
    ClaudeTmuxAgent,
    CodexTmuxAgent,
    ScriptedAgent,
    TmuxAgent,
)
from .config import load_config
from .pipeline_log import PipelineLogger

__all__ = [
    "Agent", "AgentError", "ScriptedAgent", "TmuxAgent", "ClaudeTmuxAgent", "CodexTmuxAgent",
    "load_config", "PipelineLogger",
]
