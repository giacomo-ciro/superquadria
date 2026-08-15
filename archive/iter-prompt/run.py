"""Entrypoint for running the isolated room generation prompt iteration harness.

Usage:
    .venv/bin/python iter-prompt/run.py             # codex agent and 3D render
    .venv/bin/python iter-prompt/run.py --offline   # procedural fallback, no agent
    .venv/bin/python iter-prompt/run.py --concept guitar
    .venv/bin/python iter-prompt/run.py --no-render # prompt & parse test only
"""

from __future__ import annotations

import sys
from pathlib import Path

# Fix sys.path immediately before any other imports
ITER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ITER_DIR.parent
SRC_DIR = REPO_ROOT / "src"

# Remove iter-prompt from sys.path to prevent navigation.py shadowing src/navigation
while str(ITER_DIR) in sys.path:
    sys.path.remove(str(ITER_DIR))
while "" in sys.path:
    sys.path.remove("")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

# Now import main from script
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("iter_script", ITER_DIR / "script.py")
iter_script = importlib.util.module_from_spec(spec)
sys.modules["iter_script"] = iter_script
spec.loader.exec_module(iter_script)

if __name__ == "__main__":
    sys.exit(iter_script.main())
