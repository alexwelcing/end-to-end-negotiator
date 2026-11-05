"""Run a team on herdr if it is installed, otherwise as plain subprocesses.

herdr owns the agents' terminals and marks each pane working / blocked /
idle; agents can spawn panes and wait until another agent is blocked. We only
need: start a pane running a command in a workspace, and wait for it. When
``herdr`` is not on PATH we fall back to ``subprocess``, so the loop always
runs.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PaneResult:
    name: str
    returncode: int
    stdout: str
    stderr: str


def have_herdr() -> bool:
    return shutil.which("herdr") is not None


def run_pane(name: str, cmd: list[str], cwd: str | Path, timeout: int = 1800) -> PaneResult:
    """Launch one role. On herdr, `herdr run` is the CLI entry agents use to spawn panes;
    its exact flags live in https://herdr.dev/docs/socket-api/ and may differ by version,
    so we try it and fall back to a direct subprocess on any failure."""
    if have_herdr():
        try:
            r = subprocess.run(["herdr", "run", "--name", name, "--cwd", str(cwd), "--wait", "--", *cmd],
                               capture_output=True, text=True, timeout=timeout)
            if r.returncode in (0, 1):
                return PaneResult(name, r.returncode, r.stdout, r.stderr)
        except (subprocess.SubprocessError, OSError):
            pass
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return PaneResult(name, r.returncode, r.stdout, r.stderr)


def run_claude_role(name: str, prompt: str, cwd: str | Path, model: str = "claude-sonnet-5-5", timeout: int = 1800) -> PaneResult:
    """A role agent is a Claude Code process on the user's subscription (as in EAT)."""
    if shutil.which("claude") is None:
        return PaneResult(name, 127, "", "claude CLI not found")
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json", "--permission-mode", "acceptEdits"]
    return run_pane(name, cmd, cwd, timeout)
