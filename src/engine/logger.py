"""Unified structured logging for the 3D superquadric harness.

Every log line follows a standardized fixed-width header:
    [HH:MM:SS] [stage:substage]      message
"""

from __future__ import annotations

import time
from typing import Callable


class Logger:
    """Logs `[timestamp] [stage:substage]   message` with fixed-width column alignment."""

    STAGE_WIDTH = 21

    def __init__(self, stage: str = "engine", *, sink: Callable[[str], None] = print):
        self.stage = stage
        self.sink = sink

    def log(self, message: str, stage: str | None = None) -> None:
        target_stage = stage or self.stage
        timestamp = time.strftime("%H:%M:%S")
        tag = f"[{target_stage}]"
        self.sink(f"[{timestamp}] {tag:<{self.STAGE_WIDTH}} {message}")

    def scoped(self, stage: str) -> Logger:
        """Return a logger scoped to a specific stage/pipeline."""
        return Logger(stage=stage, sink=self.sink)

    # ------------------------------------------------ backward-compatible helpers
    def start(self, step=None, message: str = "", stage: str | None = None) -> None:
        prefix = f"{step}: " if step is not None and not str(step).isdigit() else ""
        self.log(f"{prefix}{message}" if message else str(step), stage=stage)

    def end(self, step=None, message: str = "", stage: str | None = None) -> None:
        prefix = f"{step}: " if step is not None and not str(step).isdigit() else ""
        self.log(f"{prefix}{message}" if message else str(step), stage=stage)

    def info(self, step=None, message: str = "", stage: str | None = None) -> None:
        prefix = f"{step}: " if step is not None and not str(step).isdigit() else ""
        self.log(f"{prefix}{message}" if message else str(step), stage=stage)

    def warn(self, message: str, stage: str | None = None) -> None:
        self.log(f"WARNING: {message}", stage=stage)

    def error(self, message: str, stage: str | None = None) -> None:
        self.log(f"ERROR: {message}", stage=stage)


# PipelineLogger alias for backward compatibility
PipelineLogger = Logger

# Default module-level logger instance
logger = Logger("engine")
