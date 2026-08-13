"""`ClaudeTmuxAgent` — the `claude` CLI driven inside a persistent tmux session."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tmux_agent import TmuxAgent

#: Markers that mean "a turn is running". Two independent signals, because neither
#: alone is enough — and see `_idle_confirmed`, because together they still aren't:
#:
#: * The bottom status line grows an "esc to interrupt" segment during a turn — but
#:   that line is shared with transient hints, and a collapsed multi-line paste parks
#:   "paste again to expand" there for ~8s. Every prompt we send is a collapsed paste,
#:   so on any turn shorter than that hint the interrupt segment is never rendered at
#:   all and this signal alone stays silent for the whole turn.
#: * The in-transcript spinner line ("✻ Nucleating… (3s · ↓ 88 tokens)") is drawn
#:   while the model thinks, so it covers the short turns the hint hides. It is a
#:   glyph, a gerund and an ellipsis; the end-of-turn line is past tense with no
#:   ellipsis ("✻ Crunched for 1s"), so it stops matching when idle. It also stops
#:   matching once the answer starts streaming, which is why marker absence alone is
#:   never taken as "finished".
#:
#: The remaining alternatives cover other harnesses' spinners, for parity with the
#: firstmate reference regex.
_BUSY_REGEX = re.compile(
    r'(?im)esc (to )?interrupt|Working\.\.\.|Ctrl\+c:cancel|^\S +\w[\w\'’-]*…'
)


@dataclass
class ClaudeTmuxAgent(TmuxAgent):
    """Drives the `claude` CLI interactively inside a persistent tmux session."""

    binary: str = "claude"
    model: str = "claude-sonnet-5"
    #: `claude --effort`: low, medium, high, xhigh, max. None omits the flag.
    effort: str | None = None
    busy_regex: re.Pattern = _BUSY_REGEX

    def _spawn_command(self) -> list[str]:
        cmd = [self.binary, "--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        return cmd
