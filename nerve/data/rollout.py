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

Policy = Literal["random", "coverage", "adversary"]


def _negatives(spec: Spec, oracle: Oracle, s: dict, a: tuple, truth: dict, rng: random.Random, k: int) -> list[Step]:
    out: list[Step] = []
    space = spec.state_space() if spec.finite else []
    tried = 0
    while len(out) < k and tried < 10 * k and space:
        tried += 1
        cand = rng.choice(space)
        if cand == truth:
            continue
        v = oracle.violations(s, a, cand)
        out.append(Step(state=dict(s), action=[a[0], dict(a[1])], next=dict(cand), ok=False,
                        violations=v, source="adversary" if v else "oracle"))
    return out


def rollout_dataset(spec: Spec, log: EpisodeLog, episodes: int = 50, horizon: int = 20,
                    policy: Policy = "coverage", negatives: int = 3, seed: int = 0) -> int:
    rng = random.Random(seed)
    oracle = Oracle(spec)
    actions = spec.action_instances()
    # the oracle only knows a hole's *fallback*; logging that as truth would contradict the examples
    hole_actions = {h.action for h in spec.holes.values()}
    seen: set[tuple] = set()
    n = 0
    for ep in range(episodes):
        if spec.finite and policy != "random" and rng.random() < 0.5:
            s = rng.choice(spec.state_space())      # start anywhere reachable-or-not: breadth
        else:
            s = dict(spec.initial)
        for _ in range(horizon):
            if policy == "coverage":
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
