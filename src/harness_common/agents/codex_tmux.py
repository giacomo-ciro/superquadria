"""`CodexTmuxAgent` — the `codex` CLI driven inside a persistent tmux session."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tmux_agent import TmuxAgent

#: Codex draws one working indicator — "• Working (12s • esc to interrupt)" — for the
#: whole turn, thinking and streaming alike, so unlike claude there is no second
#: signal to fall back on. `_idle_confirmed` still requires a static pane on top of
#: this, which is what covers the redraw in which the indicator is briefly absent.
_BUSY_REGEX = re.compile(r'(?i)esc to interrupt')


@dataclass
class CodexTmuxAgent(TmuxAgent):
    """Drives the `codex` CLI interactively inside a persistent tmux session."""

    binary: str = "codex"
    model: str = "gpt-5.6-luna"
    #: `-c model_reasoning_effort`: low, medium, high, xhigh, max. None omits it.
    effort: str | None = None
    busy_regex: re.Pattern = _BUSY_REGEX

    def _spawn_command(self) -> list[str]:
        cmd = [
            self.binary,
            "--model", self.model,
            # Codex renders in the alternate screen by default, which has no
            # scrollback — anything that scrolls past the pane during a turn would be
            # unrecoverable at capture time. Inline mode keeps it in the history.
            "--no-alt-screen",
            # The agent only ever emits text, so it needs neither hooks nor write
            # access; both are dropped so a turn can never block on a dialog we are
            # not watching for, nor act on a prompt injected via generated content.
            "--disable", "hooks",
            "--sandbox", "read-only",
            "--ask-for-approval", "never",
            # Three first-run gates that would otherwise sit on the boot screen
            # waiting for a keypress nobody is there to send.
            "-c", "check_for_update_on_startup=false",
            "-c", f"projects.{self.workdir}.trust_level=trusted",
        ]
        if self.effort:
            cmd += ["-c", f"model_reasoning_effort={self.effort}"]
        return cmd
