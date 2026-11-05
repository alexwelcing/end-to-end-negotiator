"""Conformance: does a model behave like the oracle on held-out transitions?"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable

from .oracle import Oracle


StepFn = Callable[[dict[str, Any], tuple[str, dict[str, Any]]], dict[str, Any]]


@dataclass
class Conformance:
    n: int
    exact: float                 # fraction of transitions matching the oracle exactly
    violation_rate: float        # fraction of predicted next-states breaking invariants/ensures
    per_action: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def conformance(oracle: Oracle, step: StepFn, cases: Iterable[tuple[dict[str, Any], tuple[str, dict[str, Any]]]]) -> Conformance:
    n = ok = bad = 0
    per: dict[str, list[int]] = {}
    for s, a in cases:
        truth = oracle.step(s, a)
        pred = step(s, a)
        n += 1
        hit = int(pred == truth)
        ok += hit
        per.setdefault(a[0], []).append(hit)
        if oracle.violations(s, a, pred):
            bad += 1
    return Conformance(n=n, exact=ok / max(n, 1), violation_rate=bad / max(n, 1),
                       per_action={k: sum(v) / len(v) for k, v in per.items()})
