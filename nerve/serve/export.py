"""Export a trained network for edge execution.

The compiled circuit plus the JEPA weights are emitted as JSON float arrays
the Cloudflare Worker in ``edge/`` executes directly — no Python, no origin.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..model import NerveEnergy
from ..spec import Spec


def export_edge(spec: Spec, model: NerveEnergy, out: str | Path) -> dict[str, Any]:
    enc = model.enc
    def arr(t: torch.Tensor) -> list:
        return [round(x, 6) for x in t.detach().float().reshape(-1).tolist()]
    def lin(m: torch.nn.Linear) -> dict:
        return {"w": arr(m.weight), "b": arr(m.bias), "out": m.out_features, "in": m.in_features}
    def mlp(seq: torch.nn.Sequential) -> list:
        return [lin(l) for l in seq if isinstance(l, torch.nn.Linear)]
    j = model.jepa
    doc = {
        "spec": spec.name,
        "fields": {k: {"kind": f.kind, "domain": f.domain()} for k, f in spec.fields.items()},
        "initial": spec.initial,
        "actions": [{"name": a[0], "params": a[1]} for a in enc.all_actions()],
        "states": enc.all_states(),
        "state_vectors": enc.all_state_vectors().tolist(),
        "jepa": {"d": j.cfg.d, "enc": mlp(j.enc), "tgt": mlp(j.tgt), "pred": mlp(j.pred), "act": arr(j.act.weight)},
        "circuit": None if model.circuit is None else {
            "ws": arr(model.circuit.ws), "wa": arr(model.circuit.wa), "wo": arr(model.circuit.wo),
            "bias": float(model.circuit.bias), "H": int(model.circuit.ws.shape[1]), "weight": model.w,
        },
        "state_dim": enc.state_dim, "action_dim": enc.action_dim,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, separators=(",", ":")))
    return {"path": str(out), "bytes": Path(out).stat().st_size}
