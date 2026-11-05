"""Inference worker: the network *is* the running program.

Loads a BF16 checkpoint, applies any PULSE patches found in the patch
directory in order (verifying each digest), and exposes ``step``. A small
stdlib HTTP server makes it usable from the Cloudflare Worker in ``edge/`` or
from a browser directly. Sessions are keyed; state lives in the worker only
as a cache — the Durable Object at the edge is the source of truth.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from ..model import NerveEnergy
from ..spec import Spec, Oracle
from ..sync import Patch, apply_patch


class NerveRuntime:
    def __init__(self, spec: Spec, checkpoint: str | Path, patch_dir: str | Path | None = None):
        self.spec = spec
        self.oracle = Oracle(spec)                      # used only for *reporting* violations
        self.model = NerveEnergy.load(spec, str(checkpoint))
        self.applied: list[str] = []
        self.sessions: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        if patch_dir:
            self.sync(patch_dir)

    # ---- weights ------------------------------------------------------
    def sync(self, patch_dir: str | Path) -> int:
        """Apply every unseen patch, in lexical order. Bit-identical or refused."""
        n = 0
        for p in sorted(Path(patch_dir).glob("*.nrvp")):
            if p.name in self.applied:
                continue
            patch = Patch.load(str(p))
            sd = {k: v for k, v in self.model.jepa.state_dict().items()}
            new = apply_patch(sd, patch, verify=True)
            self.model.jepa.load_state_dict({k: v.to(torch.float32) if v.is_floating_point() else v for k, v in new.items()})
            self.applied.append(p.name)
            n += 1
        return n

    # ---- behaviour ----------------------------------------------------
    def step(self, state: dict[str, Any], action: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        nxt = self.model.step(state, action)
        E = self.model.energies(state, action)
        top = torch.topk(-E, k=min(3, len(E)))
        return {
            "state": nxt,
            "energy": float(E.min()),
            "alternatives": [{"state": self.model.enc.all_states()[int(i)], "energy": float(-e)} for e, i in zip(top.values[1:], top.indices[1:])],
            "violations": self.oracle.violations(state, action, nxt),
        }

    def session_step(self, sid: str, action: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        with self.lock:
            s = self.sessions.setdefault(sid, dict(self.spec.initial))
            out = self.step(s, action)
            self.sessions[sid] = out["state"]
        return out

    # ---- http ---------------------------------------------------------
    def serve(self, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
        rt = self

        class H(BaseHTTPRequestHandler):
            def _json(self, code: int, obj: Any) -> None:
                b = json.dumps(obj).encode()
                self.send_response(code); self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)

            def do_GET(self):
                if self.path == "/health":
                    return self._json(200, {"ok": True, "spec": rt.spec.name, "patches": rt.applied})
                if self.path == "/spec":
                    return self._json(200, {"name": rt.spec.name, "fields": {k: f.kind for k, f in rt.spec.fields.items()},
                                            "actions": list(rt.spec.actions), "initial": rt.spec.initial})
                if self.path.startswith("/session/"):
                    sid = self.path.split("/", 2)[2]
                    return self._json(200, {"state": rt.sessions.get(sid, dict(rt.spec.initial))})
                self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/step":
                    a = (body["action"], body.get("params", {}))
                    if "session" in body:
                        return self._json(200, rt.session_step(body["session"], a))
                    return self._json(200, rt.step(body["state"], a))
                if self.path == "/sync":
                    return self._json(200, {"applied": rt.sync(body["patch_dir"]), "patches": rt.applied})
                self._json(404, {"error": "not found"})

            def log_message(self, *a):  # quiet
                pass

        srv = ThreadingHTTPServer((host, port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
