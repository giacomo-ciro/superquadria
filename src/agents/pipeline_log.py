"""Standard-format logging for the generation/navigation pipeline.

Every call is tagged with the stage that produced it (gen/agent/episode/nav)
and the step or stage label it belongs to — the harness runs unattended (see
`agents/tmux_agent.py`), so one legible line per event is what makes a stuck
or slow run diagnosable after the fact.
"""

from __future__ import annotations

import time
from typing import Callable


class PipelineLogger:
    """Logs `[timestamp] [site] step=N LEVEL: message` to a single sink."""

    def __init__(self, stage: str, *, sink: Callable[[str], None] = print):
        self.stage = stage
        self.sink = sink

    def _emit(self, step, level: str, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.sink(f"[{timestamp}] [{self.stage}] step={step} {level}: {message}")

    def start(self, step, message: str) -> None:
        self._emit(step, "START", message)

    def end(self, step, message: str) -> None:
        self._emit(step, "END", message)

    def info(self, step, message: str) -> None:
        self._emit(step, "INFO", message)
