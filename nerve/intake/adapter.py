"""Run intake through a harness agent and validate the result."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..spec import Oracle, load_spec
from . import prompts

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".rs", ".go", ".swift", ".kt", ".java", ".html", ".vue", ".svelte", ".ink"}


def _kind(path: Path, kind: str) -> str:
    if kind != "auto":
        return kind
    if path.suffix.lower() in IMAGE_EXT:
        return "image"
    if path.suffix.lower() in CODE_EXT:
        return "code"
    return "prose"


def build_prompt(source: str, kind: str = "auto") -> tuple[str, str]:
    p = Path(source)
    k = _kind(p, kind) if p.exists() else ("prose" if kind == "auto" else kind)
    g = prompts.SPEC_GRAMMAR
    if k == "image":
        return k, prompts.FROM_IMAGE.format(grammar=g, path=p.resolve())
    body = p.read_text() if p.exists() else source
    if k == "code":
        return k, prompts.FROM_CODE.format(grammar=g, path=p, body=body[:60_000])
    return k, prompts.FROM_PROSE.format(grammar=g, body=body[:60_000])


def _extract_yaml(text: str) -> str:
    m = re.search(r"```ya?ml\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def intake(source: str, kind: str = "auto", model: str = "claude", out: str = "spec.nerve", rounds: int = 3,
           claude_model: str | None = None) -> dict[str, Any]:
    k, prompt = build_prompt(source, kind)
    if model == "claude":
        if shutil.which("claude") is None:
            return {"error": "claude CLI not found; install Claude Code or pass --model dsh", "kind": k, "prompt_chars": len(prompt)}
        # Read is all the agent needs (to look at the screenshot); we run `nerve check` ourselves and loop.
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--allowedTools", "Read"]
        if claude_model:
            cmd += ["--model", claude_model]
    elif model == "dsh":
        # DeepSeek Harness headless profile (see harness/dsh/). `dsh --profile headless -p` mirrors claude -p.
        if shutil.which("dsh") is None:
            return {"error": "dsh not found; `npx @deepseek-ai/dsh` or see harness/dsh/README.md", "kind": k}
        cmd = ["dsh", "--profile", "headless", "-p", prompt]
    else:
        return {"error": f"unknown model adapter {model!r}"}
    last_err: list[str] = []
    cost = 0.0
    for r in range(rounds):
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        text = res.stdout
        try:
            j = json.loads(text); text = j.get("result", text); cost += float(j.get("total_cost_usd", 0) or 0)
        except (json.JSONDecodeError, AttributeError):
            pass
        Path(out).write_text(_extract_yaml(text) + "\n")
        try:
            sp = load_spec(out)
            probs = Oracle(sp).self_check()
        except Exception as e:  # noqa: BLE001
            probs = [f"parse: {e}"]
        if not probs:
            return {"ok": True, "kind": k, "spec": out, "name": sp.name, "states": sp.state_space_size(), "rounds": r + 1, "cost_usd": round(cost, 3)}
        last_err = probs[:10]
        cmd = cmd[:2] + [prompt + "\n\nYour previous spec failed `nerve check`:\n" + "\n".join(last_err) + "\nFix it."] + cmd[3:]
    return {"ok": False, "kind": k, "spec": out, "problems": last_err, "cost_usd": round(cost, 3)}
