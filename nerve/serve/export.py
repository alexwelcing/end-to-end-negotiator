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
from ..sync.pulse import checkpoint_digest


def export_edge(spec: Spec, model: NerveEnergy, out: str | Path) -> dict[str, Any]:
    enc = model.enc
    def arr(t: torch.Tensor) -> list:
        return [round(x, 6) for x in t.detach().float().reshape(-1).tolist()]
    j = model.jepa
    sd = j.state_dict()
    # exact BF16 values (the checkpoint is BF16) so the Worker's bits match the trainer's digest
    tensors = {k: {"shape": list(v.shape), "data": v.to(torch.bfloat16).to(torch.float32).reshape(-1).tolist()}
               for k, v in sd.items()}
    doc = {
        "spec": spec.name,
        "fields": {k: {"kind": f.kind, "domain": f.domain()} for k, f in spec.fields.items()},
        "initial": spec.initial,
        "actions": [{"name": a[0], "params": a[1]} for a in enc.all_actions()],
        "states": enc.all_states(),
        "state_vectors": enc.all_state_vectors().tolist(),
        "jepa": {"d": j.cfg.d, "tensors": tensors, "digest": checkpoint_digest(sd)},
        "circuit": None if model.circuit is None else {
            "ws": arr(model.circuit.ws), "wa": arr(model.circuit.wa), "wo": arr(model.circuit.wo),
            "bias": float(model.circuit.bias), "H": int(model.circuit.ws.shape[1]), "weight": model.w,
        },
        "state_dim": enc.state_dim, "action_dim": enc.action_dim,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, separators=(",", ":")))
    return {"path": str(out), "bytes": Path(out).stat().st_size, "digest": doc["jepa"]["digest"]}


def write_patch_index(patch_dir: str | Path) -> list[str]:
    """patches/index.json: the ordered patch chain the Worker applies on load and on /sync."""
    d = Path(patch_dir); d.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in d.glob("*.nrvp"))
    (d / "index.json").write_text(json.dumps(names))
    return names
