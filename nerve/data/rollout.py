"""Generate training trajectories by operating the oracle.

Three policies, mirroring how a team of harness agents would explore a real
interface: random (breadth), coverage-seeking (visit every (state, action)
pair), and adversarial (hug the boundaries where invariants bite). Each
positive transition gets spec-derived negatives: candidate next states that
break an invariant or an ``ensures`` clause, plus plain wrong successors.
"""
from __future__ import annotations

import random
from typing import Any, Literal

from ..spec import Oracle, Spec
from .trajectories import EpisodeLog, Step

Policy = Literal["random", "coverage", "adversary", "exhaustive"]


def _perturb(spec: Spec, truth: dict, rng: random.Random) -> dict:
    """A near-miss successor for continuous specs: jitter continuous fields, sometimes flip a discrete one."""
    cand = dict(truth)
    for k in spec.continuous_fields():
        f = spec.fields[k]
        if rng.random() < 0.7:
            cand[k] = min(f.hi, max(f.lo, truth[k] + rng.gauss(0, 0.15) * (f.hi - f.lo)))
    if rng.random() < 0.3 and spec.discrete_fields():
        k = rng.choice(spec.discrete_fields()); cand[k] = spec.fields[k].sample(rng)
    return cand


def _negatives(spec: Spec, oracle: Oracle, s: dict, a: tuple, truth: dict, rng: random.Random, k: int) -> list[Step]:
    out: list[Step] = []
    space = spec.state_space() if spec.finite else None
    tried = 0
    while len(out) < k and tried < 10 * k:
        tried += 1
        cand = rng.choice(space) if space else _perturb(spec, truth, rng)
        if cand == truth:
            continue
        v = oracle.violations(s, a, cand)
        out.append(Step(state=dict(s), action=[a[0], dict(a[1])], next=dict(cand), ok=False,
                        violations=v, source="adversary" if v else "oracle"))
    return out


def exhaustive(spec: Spec, log: EpisodeLog, negatives: int = 3, seed: int = 0, max_pairs: int = 200_000) -> int:
    """Every (state, action) pair of a finite spec exactly once (plus negatives). Random policies leave
    gaps that look like generalisation failures; this removes the ambiguity. Held-out pairs are still
    excluded at training time by the hash mask, so the log can hold everything."""
    if not spec.finite:
        return 0
    rng = random.Random(seed)
    oracle = Oracle(spec)
    hole_actions = {h.action for h in spec.holes.values()}
    actions = [a for a in spec.action_instances() if a[0] not in hole_actions]
    states = spec.state_space()
    if len(states) * len(actions) > max_pairs:
        return 0
    n = 0
    for s in states:
        steps: list[Step] = []
        for a in actions:
            nxt = oracle.step(s, a)
            steps.append(Step(state=dict(s), action=[a[0], dict(a[1])], next=dict(nxt), ok=True, source="exhaustive"))
            steps += _negatives(spec, oracle, s, a, nxt, rng, negatives)
            n += 1
        log.extend(steps)
    return n


def rollout_dataset(spec: Spec, log: EpisodeLog, episodes: int = 50, horizon: int = 20,
                    policy: Policy = "coverage", negatives: int = 3, seed: int = 0) -> int:
    if policy == "exhaustive":
        return exhaustive(spec, log, negatives, seed)
    rng = random.Random(seed)
    oracle = Oracle(spec)
    actions = spec.action_instances()
    # the oracle only knows a hole's *fallback*; logging that as truth would contradict the examples
    hole_actions = {h.action for h in spec.holes.values()}
    seen: set[tuple] = set()
    n = 0
    for ep in range(episodes):
        if policy != "random" and rng.random() < 0.5:
            s = rng.choice(spec.state_space()) if spec.finite else spec.sample_state(rng)   # breadth
        else:
            s = dict(spec.initial)
        for _ in range(horizon):
            if not spec.finite:
                a = spec.sample_action(rng)
            elif policy == "coverage":
                unseen = [a for a in actions if (spec.state_key(s), spec.action_key(a)) not in seen]
                a = rng.choice(unseen or actions)
            elif policy == "adversary":
                # prefer actions whose guard is on the edge: sample a few, keep the one that
                # moves the state the least (boundary-hugging) or not at all
                pool = rng.sample(actions, min(4, len(actions)))
                a = min(pool, key=lambda x: sum(oracle.step(s, x)[k] != s[k] for k in spec.fields))
            else:
                a = rng.choice(actions)
            seen.add((spec.state_key(s), spec.action_key(a)))
            nxt = oracle.step(s, a)
            if a[0] in hole_actions:
                s = nxt
                continue
            steps = [Step(state=dict(s), action=[a[0], dict(a[1])], next=dict(nxt), ok=True, source="oracle", episode=ep)]
            steps += _negatives(spec, oracle, s, a, nxt, rng, negatives)
            log.extend(steps)
            n += 1
            s = nxt
    # ink `learn` holes: examples are ground truth the spec itself cannot express
    for h in spec.holes.values():
        for ex in h.examples:
            log.append(Step(state=dict(ex["state"]), action=[h.action, {}], next=dict(ex["next"]),
                            ok=True, source="hole-example"))
            n += 1
    return n
