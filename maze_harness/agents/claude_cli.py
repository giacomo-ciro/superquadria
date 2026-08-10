"""`ClaudeSubprocessAgent` — the concrete agent of PLAN.md section 5.

Spawns the `claude` CLI in `--print` mode with a pinned JSON Schema and reads the
newline-delimited JSON stream back. Ported from the `claude.ts` reference
implementation, with the same guiding constraint: when this fails at 3am in an
unattended run, one log line has to be enough to diagnose it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .base import Agent

#: How many unparseable stdout lines to keep for diagnostics.
STDOUT_TAIL_LINES = 25
#: Truncation applied to captured text so an error stays one readable line.
_DIAG_CHARS = 600


class ClaudeCallError(RuntimeError):
    """A failed `claude` invocation, carrying everything needed to diagnose it."""

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str] | None = None,
        exit_code: int | None = None,
        stderr: str = "",
        result_event: dict | None = None,
        stdout_tail: Sequence[str] = (),
        timed_out: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.command = list(command or [])
        self.exit_code = exit_code
        self.stderr = stderr
        self.result_event = result_event
        self.stdout_tail = list(stdout_tail)
        self.timed_out = timed_out

    def __str__(self) -> str:
        # " | " delimited so a single log line is enough to diagnose an
        # unattended run (same convention as the claude.ts reference).
        parts = [f"ClaudeCallError: {self.message}"]
        if self.timed_out:
            parts.append("timedOut=true")
        if self.exit_code is not None:
            parts.append(f"exitCode={self.exit_code}")
        if self.stderr:
            parts.append(f"stderr={_clip(self.stderr)}")
        if self.result_event is not None:
            summary = {
                k: self.result_event.get(k)
                for k in ("subtype", "is_error", "api_error_status", "terminal_reason", "num_turns")
                if k in self.result_event
            }
            parts.append(f"result={json.dumps(summary, default=str)}")
        if self.stdout_tail:
            parts.append(f"stdoutTail={_clip(' / '.join(self.stdout_tail))}")
        return " | ".join(parts)


def _clip(text: str, limit: int = _DIAG_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


@dataclass
class ClaudeSubprocessAgent(Agent):
    """One-shot `claude -p` call per agent turn."""

    model: str = "claude-sonnet-5"
    binary: str = "claude"
    timeout: float = 300.0
    max_retries: int = 2
    retry_backoff: float = 3.0
    #: PLAN.md section 5: required because the harness runs unattended with no
    #: TTY to approve the StructuredOutput tool call. Safe because
    #: `--allowedTools StructuredOutput` leaves the agent no tool but "emit
    #: text" — even under prompt injection from generated maze content.
    skip_permissions: bool = True
    cwd: str | None = None
    extra_args: tuple[str, ...] = ()
    on_event: Callable[[dict], None] | None = None
    #: Populated as calls complete; surfaced in the run log.
    stats: dict = field(default_factory=lambda: {"calls": 0, "retries": 0, "cost_usd": 0.0,
                                                 "duration_s": 0.0})

    # ------------------------------------------------------------------ public

    def run(self, prompt: str, schema: dict, *, system: str | None = None) -> dict:
        last_error: ClaudeCallError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._invoke(prompt, schema, system=system)
            except ClaudeCallError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                self.stats["retries"] += 1
                time.sleep(self.retry_backoff * (attempt + 1))
        assert last_error is not None
        raise last_error

    # ----------------------------------------------------------------- private

    def _build_command(self, prompt: str, schema: dict, system: str | None) -> list[str]:
        command = [
            self.binary,
            "--model", self.model,
            "-p", prompt,
            "--output-format", "stream-json",
            "--verbose",  # required by the CLI alongside --print + stream-json
            "--json-schema", json.dumps(schema),
            "--allowedTools", "StructuredOutput",
            # No --mcp-config is passed, so this loads no MCP servers at all:
            # the agent needs none, and skipping them cuts both startup time and
            # the tool definitions carried in every prompt.
            "--strict-mcp-config",
        ]
        if self.skip_permissions:
            command.append("--dangerously-skip-permissions")
        if system:
            command += ["--append-system-prompt", system]
        command += list(self.extra_args)
        return command

    def _invoke(self, prompt: str, schema: dict, *, system: str | None) -> dict:
        command = self._build_command(prompt, schema, system)
        started = time.monotonic()

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=self.cwd,
                # Own process group so a timeout can take the whole tree down.
                start_new_session=(os.name == "posix"),
            )
        except (OSError, ValueError) as exc:
            raise ClaudeCallError(f"failed to spawn {self.binary!r}: {exc}",
                                  command=command) from exc

        result_event: dict | None = None
        stdout_tail: deque[str] = deque(maxlen=STDOUT_TAIL_LINES)
        stderr_parts: list[str] = []

        def read_stdout() -> None:
            nonlocal result_event
            assert proc.stdout is not None
            # Iterating the pipe yields complete lines and buffers partial
            # chunks across reads, which is exactly the JSONL framing we want.
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    stdout_tail.append(line[:200])  # plain-text CLI diagnostics
                    continue
                if self.on_event is not None:
                    self.on_event(event)
                if isinstance(event, dict) and event.get("type") == "result":
                    # Prefer a terminal event that actually carries the payload.
                    if result_event is None or event.get("structured_output") is not None:
                        result_event = event

        def read_stderr() -> None:
            assert proc.stderr is not None
            stderr_parts.append(proc.stderr.read())

        threads = [threading.Thread(target=read_stdout, daemon=True),
                   threading.Thread(target=read_stderr, daemon=True)]
        for thread in threads:
            thread.start()

        timed_out = False
        try:
            proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate(proc)
        for thread in threads:
            thread.join(timeout=10)

        elapsed = time.monotonic() - started
        self.stats["calls"] += 1
        self.stats["duration_s"] += elapsed
        stderr = "".join(p for p in stderr_parts if p)

        def fail(message: str) -> ClaudeCallError:
            return ClaudeCallError(
                message,
                command=command,
                exit_code=proc.returncode,
                stderr=stderr,
                result_event=result_event,
                stdout_tail=list(stdout_tail),
                timed_out=timed_out,
            )

        if timed_out:
            raise fail(f"timed out after {self.timeout:.0f}s")
        if proc.returncode != 0:
            raise fail(f"exited with code {proc.returncode}")
        if result_event is None:
            raise fail("stream ended without a terminal result event")
        if result_event.get("is_error") or result_event.get("subtype") != "success":
            raise fail(f"result reported failure (subtype={result_event.get('subtype')!r})")

        structured = result_event.get("structured_output")
        if structured is None:
            raise fail("terminal result carried no structured_output")

        self.stats["cost_usd"] += float(result_event.get("total_cost_usd") or 0.0)
        return structured

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        """SIGTERM the process group, escalating to SIGKILL if it lingers."""
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:  # pragma: no cover - the harness targets POSIX
                proc.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:  # pragma: no cover
                    proc.kill()
            except OSError:
                pass
