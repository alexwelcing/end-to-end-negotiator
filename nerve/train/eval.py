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


def close(spec: Spec, pred: dict[str, Any], truth: dict[str, Any], tol: float = 0.01) -> bool:
    """Equality for continuous specs: discrete fields exact, continuous within tol * range."""
    for k, f in spec.fields.items():
        if f.continuous:
            if abs(float(pred[k]) - float(truth[k])) > tol * (f.hi - f.lo):
                return False
        elif pred[k] != truth[k]:
            return False
    return True


def _evaluate_continuous(spec: Spec, step_fn, n_samples: int, tol: float, learned_only_fn=None) -> dict[str, Any]:
    import random
    rng = random.Random(123)
    oracle = Oracle(spec)
    n = ok = bad = 0
    per: dict[str, list[int]] = {}
    err = []
    for _ in range(n_samples):
        s = spec.sample_state(rng); a = spec.sample_action(rng)
        truth = oracle.step(s, a); pred = step_fn(s, a)
        hit = int(close(spec, pred, truth, tol)); n += 1; ok += hit
        per.setdefault(a[0], []).append(hit)
        bad += bool(oracle.violations(s, a, pred))
        for k in spec.continuous_fields():
            f = spec.fields[k]; err.append(abs(float(pred[k]) - float(truth[k])) / (f.hi - f.lo))
    c = {"n": n, "exact": ok / max(n, 1), "violation_rate": bad / max(n, 1),
         "per_action": {k: sum(v) / len(v) for k, v in per.items()},
         "mean_abs_err_frac": float(sum(err) / max(len(err), 1)), "tol": tol}
    out = {"all": c, "seen": c, "heldout": c, "learned_only": c, "continuous": True}
    out["score"] = round(0.8 * c["exact"] + 0.2 * (1 - c["violation_rate"]), 4)
    return out


def evaluate(spec: Spec, step_fn, holdout_frac: float = 0.15, learned_only_fn=None, n_samples: int = 1000,
             tol: float = 0.01) -> dict[str, Any]:
    if not spec.finite:
        return _evaluate_continuous(spec, step_fn, n_samples, tol, learned_only_fn)
    oracle = Oracle(spec)
    enc = Encoder(spec)
    hole_actions = {h.action for h in spec.holes.values()}
    # the oracle only knows a hole's *fallback*, so holes are scored on their examples, not here
    allpairs = [(s, a) for s in enc.all_states() for a in enc.all_actions() if a[0] not in hole_actions]
    if len(allpairs) < 20:
        holdout_frac = 0.0     # nothing to generalise from in a handful of transitions (e.g. toggle): score on all
    held = [(s, a) for s, a in allpairs if holdout_mask(spec, s, a, holdout_frac)]
    seen = [(s, a) for s, a in allpairs if not holdout_mask(spec, s, a, holdout_frac)]
    holes = bool(hole_actions)
    out = {
        "all": conformance(oracle, step_fn, allpairs).to_dict(),
        "seen": conformance(oracle, step_fn, seen).to_dict(),
        "heldout": conformance(oracle, step_fn, held).to_dict() if held else conformance(oracle, step_fn, allpairs).to_dict(),
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
