"""Run the edge Worker locally on workerd (Cloudflare's open-source Workers runtime).

This is the same artefact that deploys to Cloudflare, executed on the same
runtime, on your machine. Used by the tests for parity through real HTTP and
by `nerve edge` for a local product with no Python on the serving path.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

EDGE = Path(__file__).resolve().parents[2] / "edge"


def available() -> bool:
    return (EDGE / "node_modules" / ".bin" / "workerd").exists() and shutil.which("node") is not None


def export_dir(spec_name: str) -> Path:
    return EDGE / "export" / spec_name


def bundle() -> Path:
    out = EDGE / "dist" / "index.js"
    subprocess.run(["npx", "esbuild", "src/index.ts", "--bundle", "--format=esm", f"--outfile={out}", "--log-level=warning"],
                   cwd=EDGE, check=True)
    return out


class Workerd:
    """Context manager: bundle, start workerd, wait for health, stop."""

    def __init__(self, port: int = 8788, export_root: str = "export"):
        self.port = port
        self.export_root = export_root        # relative to edge/; tests use their own so they never clobber a real export
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"

    def __enter__(self) -> "Workerd":
        bundle()
        (EDGE / ".do-storage").mkdir(exist_ok=True); (EDGE / self.export_root).mkdir(parents=True, exist_ok=True)
        cfg = (EDGE / "workerd.capnp").read_text().replace("127.0.0.1:8788", f"127.0.0.1:{self.port}")
        cfg = cfg.replace('disk = (path = "export"', f'disk = (path = "{self.export_root}"')
        if self.export_root != "export":
            cfg = cfg.replace('path = ".do-storage"', f'path = "{self.export_root}/.do-storage"')
            (EDGE / self.export_root / ".do-storage").mkdir(exist_ok=True)
        tmp = EDGE / f".workerd.{self.port}.capnp"; tmp.write_text(cfg)
        self.proc = subprocess.Popen(["npx", "workerd", "serve", tmp.name], cwd=EDGE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(100):
            try:
                urllib.request.urlopen(self.base + "/", timeout=0.5).read()
                return self
            except Exception:  # noqa: BLE001
                if self.proc.poll() is not None:
                    raise RuntimeError("workerd exited:\n" + (self.proc.stdout.read() if self.proc.stdout else ""))
                time.sleep(0.1)
        raise RuntimeError("workerd did not come up")

    def __exit__(self, *a) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for f in EDGE.glob(f".workerd.{self.port}.capnp"):
            f.unlink()

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def step(self, spec: str, state: dict, action: tuple[str, dict]) -> dict:
        return self.post(f"/{spec}/step", {"state": state, "action": action[0], "params": action[1]})
