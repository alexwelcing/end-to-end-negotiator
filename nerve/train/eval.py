"""Official eval: conformance of the network to the spec's oracle.

This is the single score the EAT-style supervisor optimises. It is computed
on three slices: transitions seen in the log, transitions deliberately held
out (never trained on), and the whole finite space.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ..spec import Oracle, Spec, conformance
from ..spec.encode import Encoder


def holdout_mask(spec: Spec, s: dict[str, Any], a: tuple[str, dict[str, Any]], frac: float, salt: str = "nerve") -> bool:
    """Deterministic: is this (state, action) pair in the held-out set?"""
    key = f"{salt}|{spec.state_key(s)}|{spec.action_key(a)}".encode()
    h = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big") / 2**32
    return h < frac


def evaluate(spec: Spec, step_fn, holdout_frac: float = 0.15, learned_only_fn=None) -> dict[str, Any]:
    oracle = Oracle(spec)
    enc = Encoder(spec)
    hole_actions = {h.action for h in spec.holes.values()}
    # the oracle only knows a hole's *fallback*, so holes are scored on their examples, not here
    allpairs = [(s, a) for s in enc.all_states() for a in enc.all_actions() if a[0] not in hole_actions]
    held = [(s, a) for s, a in allpairs if holdout_mask(spec, s, a, holdout_frac)]
    seen = [(s, a) for s, a in allpairs if not holdout_mask(spec, s, a, holdout_frac)]
    holes = bool(hole_actions)
    out = {
        "all": conformance(oracle, step_fn, allpairs).to_dict(),
        "seen": conformance(oracle, step_fn, seen).to_dict(),
        "heldout": conformance(oracle, step_fn, held).to_dict(),
    }
    if holes:
        # for holes the oracle only knows the fallback; the real truth is the examples
        ex = []
        for h in spec.holes.values():
            for e in h.examples:
                ex.append((e["state"], (h.action, {}), e["next"]))
        hit = sum(step_fn(s, a) == n for s, a, n in ex)
        out["hole_examples"] = {"n": len(ex), "exact": hit / max(len(ex), 1)}
    if learned_only_fn is not None:
        out["learned_only"] = conformance(oracle, learned_only_fn, allpairs).to_dict()
    # heldout: generalisation; all: full conformance (circuit + jepa); violations: safety;
    # learned_only: what the network itself knows, independent of the compiled circuit.
    score = 0.4 * out["heldout"]["exact"] + 0.2 * out["all"]["exact"] + 0.2 * (1 - out["all"]["violation_rate"])
    score += 0.2 * (out["learned_only"]["exact"] if "learned_only" in out else out["heldout"]["exact"])
    if holes:
        score = 0.8 * score + 0.2 * out["hole_examples"]["exact"]
    out["score"] = round(score, 4)
    return out
