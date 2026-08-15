"""The Agent abstraction — interactive coding CLI driven inside persistent tmux
sessions.

Spawns and drives a coding CLI inside a dedicated tmux window, following the same
spawn/type/busy-poll shape as the `firstmate` repository's tmux backend. Each turn
tags its prompt with a nonce and reads only the pane text that follows it, so
extraction can never pick up an earlier turn's answer, and a busy pane is
interrupted before a retry resubmits, so a slow or failed turn can't leave two
turns racing in the same pane.

Subclasses (Claude, Codex) supply what differs between CLIs: the command that spawns
the interactive process (`_spawn_command`) and the pane markers that mean "a turn is
running" (`busy_regex`).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from logger import Logger

_SESSION_LOCK = threading.Lock()

#: The agent window is pinned to this size. Both CLIs hard-wrap their output to the
#: pane width with real newlines — inside JSON string values too — so a narrow pane
#: mangles every long field the model returns. Wider only reduces the damage; it does
#: not eliminate it. See `_ensure_session` for why pinning takes a `resize-window`.
SESSION_WIDTH = 200
SESSION_HEIGHT = 50

#: How long a frozen pane holding no extractable answer is given before the turn is
#: called failed. "The pane stopped changing" cannot on its own mean "the answer is
#: complete": a model that pauses mid-JSON leaves the pane static for seconds at a time,
#: and capturing there yields a truncated object that fails the whole call. So the wait
#: ends on a *parseable* answer instead — but a turn that really did finish without one
#: has to be given up on well before the call timeout, or every such retry would cost
#: minutes of waiting on a pane that will never change again.
RESPONSE_SETTLE_GRACE = 60.0

#: How long a parseable answer must sit on an unchanging pane before it is accepted.
#: Without this the wait ends the moment *an* answer parses, which is not the same as
#: the turn being over: a model that writes one complete object, pauses, then writes a
#: corrected one would have the superseded draft taken as its answer. Extraction is
#: last-wins within a capture, so all this has to buy is the confidence that nothing
#: further is coming — and `_idle_confirmed` has already required a still pane for a
#: second before we get here, so the real silence demanded is a second longer than this.
ANSWER_CONFIRM = 0.0


class AgentError(RuntimeError):
    """A failed agent invocation, whichever backend raised it."""


@dataclass
class Agent(ABC):
    """Drives an interactive CLI inside a persistent tmux session.

    Abstract: every concrete backend must say how its CLI is spawned. The tmux
    driving itself — submit, poll, extract, retry — is identical across backends
    and lives here.
    """

    session_name: str = "general-intuition"
    window_name: str = "agent"
    binary: str = ""
    model: str = ""
    timeout: float = 300.0
    max_retries: int = 2
    retry_backoff: float = 3.0
    cwd: str | None = None
    #: Reasoning effort, spelled however the concrete CLI spells it. None omits it.
    effort: str | None = None
    _is_initialized: bool = field(default=False, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False)

    #: Populated as calls complete; surfaced in the run log.
    stats: dict = field(
        default_factory=lambda: {"calls": 0, "retries": 0, "duration_s": 0.0}
    )
    log: Logger = field(default_factory=lambda: Logger("agent"))

    #: Pane markers meaning "a turn is running"; see the concrete agents.
    busy_regex: re.Pattern = re.compile(r"(?i)esc (to )?interrupt")

    @property
    def target(self) -> str:
        """This agent's window, as an exact-match tmux target.

        Both halves are `=`-prefixed: for a bare name tmux falls back to prefix and
        then fnmatch matching, so an unrelated `general-intuition-old` session or
        `player-old` window left over from something else answers to our target and
        gets driven — and killed — in place of ours.
        """
        return f"={self.session_name}:={self.window_name}"

    @property
    def workdir(self) -> str:
        return self.cwd or os.getcwd()

    @abstractmethod
    def _spawn_command(self) -> list[str]:
        """The argv that starts the interactive CLI in the agent's tmux window."""

    # ------------------------------------------------------------------ public

    def run(self, prompt: str, schema: dict, *, system: str | None = None) -> dict:
        self._cancelled = False
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            call_no = self.stats["calls"] + 1
            if attempt == 0:
                self.log.log(
                    f"call {call_no}: attempt 1/{self.max_retries + 1} "
                    f"({self.model or self.binary}, "
                    f"effort={self.effort or 'default'})"
                )
            else:
                self.log.log(
                    f"call {call_no}: attempt {attempt + 1}/{self.max_retries + 1}"
                )
            try:
                result = self._invoke(prompt, schema, system=system)
                self.log.log(f"call {call_no}: ok ({self.stats['duration_s']:.1f}s)")
                return result
            except Exception as exc:
                last_error = exc
                # A cancelled turn fails like any other; without this the retry
                # would resubmit into the pane we just interrupted.
                if self._cancelled:
                    self.log.log(f"call {call_no}: cancelled")
                    break
                if attempt == self.max_retries:
                    self.log.log(f"call {call_no}: failed, out of retries: {exc}")
                    break
                self.stats["retries"] += 1
                backoff = self.retry_backoff * (attempt + 1)
                self.log.log(
                    f"call {call_no}: failed, retrying in {backoff:.0f}s: {exc}"
                )
                # Force a fresh window for the retry instead of resubmitting into
                # this one, which may be left mid-response or otherwise confused.
                self._is_initialized = False
                time.sleep(backoff)
            except BaseException:
                # Ctrl+C leaves the pane mid-turn, still burning tokens on an
                # answer nobody will read. Interrupt it on the way out.
                self.cancel()
                raise
        assert last_error is not None
        raise last_error

    def cancel(self) -> None:
        """Interrupt the running turn and stop retrying. Called from another
        thread, so it only sends the keystroke — it does not wait for idle,
        which is the worker's job on its way out."""
        self._cancelled = True
        if self._is_initialized:
            subprocess.run(
                ["tmux", "send-keys", "-t", self.target, "Escape"], check=False
            )

    @classmethod
    def clean_session_windows(cls, session_name: str) -> None:
        """Clear all existing windows in a session without destroying the session,
        so attached clients stay connected."""
        if (
            not subprocess.run(
                ["tmux", "has-session", "-t", f"={session_name}"], capture_output=True
            ).returncode
            == 0
        ):
            return

        res = subprocess.run(
            ["tmux", "list-windows", "-t", f"={session_name}", "-F", "#{window_id}"],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            return
        old_wids = [w.strip() for w in res.stdout.strip().split("\n") if w.strip()]
        if not old_wids:
            return

        # Create a standby window first so tmux never closes the session
        standby_res = subprocess.run(
            [
                "tmux",
                "new-window",
                "-P",
                "-F",
                "#{window_id}",
                "-t",
                f"={session_name}:",
                "-n",
                "standby",
                "tail",
                "-f",
                "/dev/null",
            ],
            capture_output=True,
            text=True,
        )
        if standby_res.returncode != 0:
            return

        for wid in old_wids:
            subprocess.run(["tmux", "kill-window", "-t", wid], check=False)

    def clean(self) -> None:
        """Clear all windows in this agent's session while preserving the session."""
        self.clean_session_windows(self.session_name)

    # ----------------------------------------------------------------- private

    def _session_exists(self) -> bool:
        return (
            subprocess.run(
                ["tmux", "has-session", "-t", f"={self.session_name}"],
                capture_output=True,
            ).returncode
            == 0
        )

    def _ensure_session(self) -> None:
        """Spawn a brand-new CLI process in this agent's window.

        Any window a previous run left behind is torn down first: the CLI in it holds
        the whole previous episode as conversation history, and reusing it would carry
        that episode's map and moves into this one.
        """
        if self._is_initialized:
            return

        with _SESSION_LOCK:
            if self._is_initialized:
                return

            if not self._session_exists():
                cmd = [
                    "tmux",
                    "new-session",
                    "-d",
                    "-s",
                    self.session_name,
                    "-n",
                    self.window_name,
                    "-c",
                    self.workdir,
                    "-x",
                    str(SESSION_WIDTH),
                    "-y",
                    str(SESSION_HEIGHT),
                    "-P",
                    "-F",
                    "#{window_id}",
                ]
                cmd.extend(self._spawn_command())
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                wid = res.stdout.strip()
            else:
                # Find any existing window with the same name or a "standby" placeholder
                res = subprocess.run(
                    [
                        "tmux",
                        "list-windows",
                        "-t",
                        f"={self.session_name}",
                        "-F",
                        "#{window_id}:#{window_name}",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                to_kill = []
                for line in res.stdout.strip().split("\n"):
                    if not line:
                        continue
                    w_id, w_name = line.split(":", 1)
                    if w_name == self.window_name or w_name == "standby":
                        to_kill.append(w_id)

                # Create the new window first to ensure the session never reaches
                # 0 windows
                cmd = [
                    "tmux",
                    "new-window",
                    "-P",
                    "-F",
                    "#{window_id}",
                    "-t",
                    f"={self.session_name}:",
                    "-n",
                    self.window_name,
                    "-c",
                    self.workdir,
                ]
                cmd.extend(self._spawn_command())
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                wid = res.stdout.strip()

                # Clean up the old windows with the same name / standby
                for old_id in to_kill:
                    subprocess.run(["tmux", "kill-window", "-t", old_id], check=False)

            # Pin the window name so it doesn't change on directory switches
            subprocess.run(
                ["tmux", "set-window-option", "-t", wid, "automatic-rename", "off"],
                check=True,
            )
            subprocess.run(
                ["tmux", "set-window-option", "-t", wid, "allow-rename", "off"],
                check=True,
            )

            # Pin the window size. `new-session -x/-y` only covers a server with no
            # clients at all: the default `window-size latest` resizes the window to
            # the terminal of whichever client was used last, even one attached to an
            # unrelated session, so in practice the pane silently took the size of
            # whatever else was open (200x50 asked for, 178x19 observed). Since both
            # CLIs hard-wrap to the pane width with real newlines, that quietly
            # mangles long JSON fields. `resize-window` sets this window to
            # `window-size manual`, which no attaching client can override.
            subprocess.run(
                [
                    "tmux",
                    "resize-window",
                    "-t",
                    wid,
                    "-x",
                    str(SESSION_WIDTH),
                    "-y",
                    str(SESSION_HEIGHT),
                ],
                check=True,
            )

        # Printed now rather than after the boot wait, so it can be pasted while the CLI
        # is still starting. `ignore-size` lets a watcher use copy mode without changing
        # the pinned pane dimensions the response capture relies on.
        self.log.log(
            f"attach: tmux attach -f ignore-size -t "
            f"{self.session_name}:{self.window_name}"
        )

        # Wait for the CLI to boot and settle its TUI, rather than guessing a fixed
        # delay.
        self._wait_idle(15.0, min_wait=1.5)
        self._is_initialized = True

    def _capture(self) -> str:
        res = subprocess.run(
            ["tmux", "capture-pane", "-p", "-t", self.target, "-S", "-40"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout

    def _capture_response(self) -> str:
        """The whole pane, wrapped lines rejoined — what an answer is extracted from.

        `-J` joins the CLI's terminal soft-wraps so JSON doesn't break at the pane
        width; `-S -` takes the full history rather than the 40 lines `_capture` reads.
        """
        res = subprocess.run(
            ["tmux", "capture-pane", "-p", "-J", "-t", self.target, "-S", "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout

    def _is_busy(self) -> bool:
        return bool(self.busy_regex.search(self._capture()))

    def _idle_confirmed(self) -> bool:
        """True only when the pane shows no busy marker *and* has stopped changing.

        Marker absence alone is not enough. Once the model starts streaming its
        answer the spinner line is gone, leaving the status line's "esc to interrupt"
        as the only marker — and that segment is itself hidden while a transient hint
        occupies the status line, which "paste again to expand" does for ~8s after
        every prompt we send. Both markers are therefore blind for the whole streaming
        phase of a turn that starts answering inside that window, and a turn still
        writing its answer was read as finished, capturing a pane that holds only the
        echoed prompt.

        A turn that is still running keeps redrawing the pane — a spinner frame, a
        ticking elapsed counter, streamed output — while a finished one is completely
        static, so requiring the capture to be byte-identical across the gap covers
        the blind spot without depending on any particular marker being rendered.
        """
        before = self._capture()
        if self.busy_regex.search(before):
            return False
        time.sleep(1.0)
        after = self._capture()
        return after == before and not self.busy_regex.search(after)

    def _wait_idle(self, timeout: float, *, min_wait: float = 0.0) -> bool:
        """Block until the pane is confirmed idle; False if `timeout` ran out first."""
        if min_wait:
            time.sleep(min_wait)
        deadline = time.monotonic() + timeout
        while True:
            if self._idle_confirmed():
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.5)

    def _interrupt_if_busy(self) -> None:
        """Cancel a still-running turn left over from a failed prior attempt.

        Without this, a retry pastes a new prompt into a pane that's still mid-turn:
        the CLI queues it behind the old turn instead of running it, and whichever
        response lands first gets captured for the wrong call.
        """
        if not self._is_busy():
            return
        subprocess.run(["tmux", "send-keys", "-t", self.target, "Escape"], check=True)
        # If the Escape didn't take, submitting anyway is the exact mix-up this guard
        # exists to prevent, so fail instead and let the retry respawn the window.
        if not self._wait_idle(15.0, min_wait=0.5):
            raise AgentError("pane stayed busy after the interrupt")

    @staticmethod
    def _iter_balanced_objects(text: str):
        """Yield each top-level `{...}` span in `text`, ignoring braces inside
        strings."""
        depth = 0
        in_string = False
        escape = False
        start = None
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : i + 1]
                    start = None

    def _extract(self, pane: str, nonce: str, required: list) -> dict | None:
        """This turn's answer, or None if `pane` does not hold a complete one yet.

        None covers both "still being written" and "this turn produced no answer" —
        indistinguishable from a single capture, which is why the caller polls this
        rather than treating one None as failure.
        """
        # Keep only what the pane gained after this turn's prompt was echoed. Without
        # this, a turn that produces no JSON of its own (a refusal, an error banner, a
        # turn read as finished too early) falls through to the previous turn's answer
        # still sitting on the visible pane and hands it back as this turn's — a stale
        # move replayed silently, with no failure recorded anywhere. Anchor on the first
        # occurrence, so a model that repeats the marker in its answer cannot truncate
        # its own output.
        anchor = pane.find(nonce)
        if anchor < 0:
            return None
        turn_output = pane[anchor + len(nonce) :]

        # strict=False: the CLI word-wraps long string values to the pane width by
        # printing real newlines (not a terminal soft-wrap tmux -J could rejoin), so
        # a captured string field can contain literal control characters that strict
        # JSON parsing would otherwise reject.
        # Every candidate is checked against the schema's required keys, because not all
        # of the JSON in a turn is its answer — a model that restates the schema or
        # sketches an example emits objects that parse fine, and parsing alone would
        # hand one of those back, which the caller then reads as a malformed response
        # rather than the failed turn it is. Rejecting them here turns that into a
        # retry.
        def accept(text: str) -> dict | None:
            try:
                obj = json.loads(text, strict=False)
            except json.JSONDecodeError:
                return None
            if not isinstance(obj, dict) or any(key not in obj for key in required):
                return None
            return obj

        # Prefer an explicit fenced block if the model produced one; try the latest
        # first, in case an earlier one belongs to draft/example text. A block still
        # being streamed has no closing fence yet, so it simply doesn't match — which
        # is exactly the "not finished" answer the caller is polling for.
        blocks = re.findall(r"```json\s*(.*?)\s*```", turn_output, re.DOTALL)
        for block in reversed(blocks):
            if (obj := accept(block)) is not None:
                return obj

        # Fallback: the model often just emits raw JSON with no fence. Scan for every
        # balanced top-level object — the echoed prompt's own schema literal is one
        # such object, so a naive "first match" would grab that instead — and try
        # them from the end, since the actual answer comes after the echoed prompt.
        for candidate in reversed(list(self._iter_balanced_objects(turn_output))):
            if (obj := accept(candidate)) is not None:
                return obj
        return None

    def _invoke(self, prompt: str, schema: dict, *, system: str | None) -> dict:
        self._ensure_session()
        # Cancel any turn a previous failed attempt left running, so this submission
        # can't queue up behind it and get its response mixed up with another turn's.
        self._interrupt_if_busy()

        # 1. Format prompt for interactive structured output
        # Unlike subprocess, we can't pin `--json-schema` dynamically per turn, so we
        # inject the schema explicitly and request a fenced codeblock for reliable
        # extraction. No literal ```json example here: that text gets pasted into the
        # pane and echoed back, and would otherwise be mistaken for the model's own
        # answer during extraction.
        full_prompt = system + "\n\n" if system else ""
        full_prompt += prompt
        full_prompt += (
            f"\n\nRespond strictly with valid JSON matching this schema:\n"
            f"{json.dumps(schema)}"
        )
        full_prompt += (
            "\nWrap your output in a fenced code block labeled json, "
            "and output nothing else."
        )
        # A nonce that ties extraction to this turn's own output. Both CLIs echo the
        # submitted prompt into the transcript verbatim, so it lands in the pane ahead
        # of the answer and everything before it can be dropped. It goes last, after
        # the schema, so the echoed schema literal is dropped along with it.
        nonce = uuid.uuid4().hex[:12]
        full_prompt += f"\n(ignore this turn marker: {nonce})"

        started = time.monotonic()

        # 2. Free the pane's scrollback. This is not on its own enough to isolate the
        # turn — `clear-history` frees the history but leaves the visible region
        # standing, so the tail of the previous turn (its answer included) is still
        # capturable afterwards, which is what the nonce above is for.
        subprocess.run(["tmux", "clear-history", "-t", self.target], check=True)

        # Send keys (Replicating `fm_tmux_submit_core`)
        # We use set-buffer/paste-buffer to avoid escaping issues typing raw text.
        # Large multi-line pastes land in the composer as a collapsed "[Pasted text
        # #1 +N lines]" placeholder, which needs a moment to register before Enter
        # will actually submit it.
        # Named buffer prevents collisions between concurrent turns across windows.
        buf_name = f"buf-{nonce}"
        subprocess.run(
            ["tmux", "set-buffer", "-b", buf_name, "--", full_prompt], check=True
        )
        subprocess.run(
            ["tmux", "paste-buffer", "-p", "-t", self.target, "-b", buf_name],
            check=True,
        )
        subprocess.run(["tmux", "delete-buffer", "-b", buf_name], check=False)
        time.sleep(1.0)  # Settle time for the paste to register

        # 3. Submit, retrying Enter only — never retyping, since a swallowed Enter
        # leaves our text in the composer and retyping would duplicate it (observed
        # empirically: a single Enter sent too soon after the paste is sometimes
        # dropped entirely, leaving the pane idle with the prompt never submitted).
        # The pane is guaranteed idle at this point (interrupted above if busy), so a
        # busy reading here can only mean this submission specifically was accepted.
        # Poll busy() frequently rather than once per Enter: a quick turn's busy
        # window can start and end within a single second, so checking only once,
        # a full second after each Enter, can miss it entirely and misreport an
        # accepted submission as dropped (observed on fast navigation turns whose
        # whole busy window was a couple of seconds).
        # The deadline stays generous because the cost of overshooting it is high —
        # a retry kills the pane out from under a submission that may have gone
        # through — while on the happy path the loop exits on the first poll.
        transition_deadline = time.monotonic() + 45.0
        retry_every = 2.0
        next_enter = time.monotonic()
        submitted = False
        while time.monotonic() < transition_deadline:
            if time.monotonic() >= next_enter:
                subprocess.run(
                    ["tmux", "send-keys", "-t", self.target, "Enter"], check=True
                )
                next_enter = time.monotonic() + retry_every
            if self._is_busy():
                submitted = True
                break
            time.sleep(0.2)
        if not submitted:
            raise AgentError("submission was not accepted (never went busy)")

        # 4. Wait for the turn to produce a complete answer.
        #
        # Not just for the pane to go idle: idle is "no busy marker and byte-identical
        # across a one-second gap", and a model that pauses longer than that partway
        # through writing its JSON satisfies it while its object is still half-written.
        # Capturing there failed the call on a truncated object even though the turn
        # went on to answer perfectly well. So the wait ends on an answer that actually
        # parses and carries the schema's required keys, and a pane that is idle but
        # not yet parseable is just a pane that isn't finished.
        #
        # Nor does it end the moment an answer first parses, which is not the same as
        # the turn being over — a model that writes one object, pauses, then writes a
        # corrected one would have the draft taken as its answer. So the pane must also
        # have stopped changing: an answer is accepted once it has survived
        # ANSWER_CONFIRM on a frozen pane, and extraction is last-wins, so a turn
        # that kept writing is
        # read at its final state. A turn that genuinely produces no answer would hold
        # this loop until the call timeout, so a pane frozen for the much longer
        # RESPONSE_SETTLE_GRACE with nothing extractable is taken as finished-and-failed
        # and handed to the retry. Any further output resets both clocks.
        required = schema.get("required", [])
        timeout_at = time.monotonic() + self.timeout
        answer: dict | None = None
        pane = ""
        frozen_since = time.monotonic()
        while True:
            if self._cancelled:
                raise AgentError("cancelled")
            if time.monotonic() > timeout_at:
                raise AgentError(f"timed out after {self.timeout:.0f}s")
            if self._idle_confirmed():
                latest = self._capture_response()
                if latest != pane:
                    pane, frozen_since = latest, time.monotonic()
                frozen_for = time.monotonic() - frozen_since
                candidate = self._extract(pane, nonce, required)
                if candidate is not None:
                    if frozen_for >= ANSWER_CONFIRM:
                        answer = candidate
                        break
                elif frozen_for > RESPONSE_SETTLE_GRACE:
                    break
            time.sleep(1.0)

        # 5. Record the call. A turn that answered nothing still ran and still spent
        # tokens, so it counts here exactly as a successful one does.
        self.stats["calls"] += 1
        self.stats["duration_s"] += time.monotonic() - started

        if answer is None:
            raise AgentError(
                "this turn's prompt never appeared in the pane"
                if nonce not in pane
                else f"no JSON object with keys {required} found in pane output"
            )
        return answer
