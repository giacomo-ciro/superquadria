"""`ClaudeSubprocessAgent` against a fake `claude` binary.

The fake emits the same newline-delimited JSON the real CLI does, so these
exercise the stream parsing, the validation ladder, and the timeout kill without
spending a single token.
"""

import json
import stat
import sys
import textwrap

import pytest

from maze_harness.agents import ClaudeCallError, ClaudeSubprocessAgent

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def fake_claude(tmp_path, body: str, name: str = "claude"):
    """Write an executable stand-in for the CLI that runs `body`."""
    path = tmp_path / name
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def emit(*events: dict) -> str:
    lines = "\n".join(json.dumps(e) for e in events)
    return f"""
        import sys
        sys.stdout.write({lines!r} + "\\n")
    """


RESULT_OK = {"type": "result", "subtype": "success", "is_error": False,
             "structured_output": {"ok": True}, "total_cost_usd": 0.01}


def agent(binary, **kwargs):
    return ClaudeSubprocessAgent(binary=binary, max_retries=0, timeout=15, **kwargs)


def test_returns_structured_output_from_the_terminal_event(tmp_path):
    binary = fake_claude(tmp_path, emit({"type": "system"}, {"type": "assistant"}, RESULT_OK))
    call = agent(binary)
    assert call.run("hi", SCHEMA) == {"ok": True}
    assert call.stats["calls"] == 1
    assert call.stats["cost_usd"] == pytest.approx(0.01)


def test_plain_text_diagnostics_are_kept_not_dropped(tmp_path):
    binary = fake_claude(tmp_path, """
        import sys
        sys.stdout.write("warning: node version is old\\n")
        sys.stdout.write('{"type":"result","subtype":"error","is_error":true}\\n')
    """)
    with pytest.raises(ClaudeCallError) as excinfo:
        agent(binary).run("hi", SCHEMA)
    assert "warning: node version is old" in excinfo.value.stdout_tail[0]
    assert "reported failure" in excinfo.value.message


def test_missing_structured_output_is_an_error(tmp_path):
    event = dict(RESULT_OK)
    event.pop("structured_output")
    binary = fake_claude(tmp_path, emit(event))
    with pytest.raises(ClaudeCallError, match="no structured_output"):
        agent(binary).run("hi", SCHEMA)


def test_missing_result_event_is_an_error(tmp_path):
    binary = fake_claude(tmp_path, emit({"type": "assistant"}))
    with pytest.raises(ClaudeCallError, match="without a terminal result event"):
        agent(binary).run("hi", SCHEMA)


def test_nonzero_exit_carries_code_and_stderr(tmp_path):
    binary = fake_claude(tmp_path, """
        import sys
        sys.stderr.write("credit balance too low\\n")
        sys.exit(3)
    """)
    with pytest.raises(ClaudeCallError) as excinfo:
        agent(binary).run("hi", SCHEMA)
    assert excinfo.value.exit_code == 3
    assert "credit balance too low" in excinfo.value.stderr
    assert "exitCode=3" in str(excinfo.value)


def test_a_late_result_event_wins_over_an_empty_one(tmp_path):
    early = {"type": "result", "subtype": "success", "is_error": False, "structured_output": None}
    binary = fake_claude(tmp_path, emit(early, RESULT_OK))
    assert agent(binary).run("hi", SCHEMA) == {"ok": True}


def test_timeout_kills_the_process_group(tmp_path):
    binary = fake_claude(tmp_path, """
        import sys, time
        sys.stdout.write('{"type":"system"}\\n'); sys.stdout.flush()
        time.sleep(60)
    """)
    call = ClaudeSubprocessAgent(binary=binary, max_retries=0, timeout=1)
    with pytest.raises(ClaudeCallError) as excinfo:
        call.run("hi", SCHEMA)
    assert excinfo.value.timed_out
    assert "timedOut=true" in str(excinfo.value)


def test_spawn_failure_is_reported(tmp_path):
    with pytest.raises(ClaudeCallError, match="failed to spawn"):
        agent(str(tmp_path / "does-not-exist")).run("hi", SCHEMA)


def test_retries_then_succeeds(tmp_path):
    # Fails once (marker file absent), succeeds on the retry.
    marker = tmp_path / "seen"
    binary = fake_claude(tmp_path, f"""
        import os, sys, json
        marker = {str(marker)!r}
        if not os.path.exists(marker):
            open(marker, "w").close()
            sys.exit(1)
        sys.stdout.write(json.dumps({RESULT_OK!r}) + "\\n")
    """)
    call = ClaudeSubprocessAgent(binary=binary, max_retries=1, retry_backoff=0, timeout=15)
    assert call.run("hi", SCHEMA) == {"ok": True}
    assert call.stats["retries"] == 1


def test_command_includes_the_pinned_schema_and_tool_allowlist(tmp_path):
    call = ClaudeSubprocessAgent(binary="claude", model="claude-opus-5")
    command = call._build_command("prompt text", SCHEMA, system="sys")
    assert command[:2] == ["claude", "--model"]
    assert "--json-schema" in command
    assert json.loads(command[command.index("--json-schema") + 1]) == SCHEMA
    assert command[command.index("--allowedTools") + 1] == "StructuredOutput"
    assert "--strict-mcp-config" in command      # no MCP servers in an unattended run
    assert "--dangerously-skip-permissions" in command
    assert command[command.index("--append-system-prompt") + 1] == "sys"

    lenient = ClaudeSubprocessAgent(skip_permissions=False)
    assert "--dangerously-skip-permissions" not in lenient._build_command("p", SCHEMA, None)
