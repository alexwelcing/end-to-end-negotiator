"""Reference interpreter for a spec.

The oracle is what the network must match. It is also the only piece of
"software" in the traditional sense, and it is derived mechanically from the
spec, never hand-written per interface.
"""
from __future__ import annotations

from typing import Any

from .expr import eval_expr
from .schema import Spec


class Violation(Exception):
    pass


class Oracle:
    def __init__(self, spec: Spec):
        self.spec = spec

    # ---- stepping -----------------------------------------------------
    def step(self, state: dict[str, Any], action: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        name, params = action
        t = self.spec.transitions[name]
        env = dict(state)
        env.update(params)
        if t._when is not None and not eval_expr(t._when, env, old=state):
            return dict(state)
        new = dict(state)
        for k, tree in t._updates.items():
            v = eval_expr(tree, env, old=state)
            f = self.spec.fields[k]
            if f.kind == "int" and isinstance(v, float) and v.is_integer():
                v = int(v)
            if f.kind == "enum":
                v = str(v)
            new[k] = f.canon(v)
        return new

    def rollout(self, actions: list[tuple[str, dict[str, Any]]], state: dict[str, Any] | None = None):
        s = dict(state or self.spec.initial)
        out = [s]
        for a in actions:
            s = self.step(s, a)
            out.append(s)
        return out

    # ---- checking -----------------------------------------------------
    def violations(self, state: dict[str, Any], action: tuple[str, dict[str, Any]] | None, new: dict[str, Any],
                   tol: float = 0.01) -> list[str]:
        """Which spec properties does (state -action-> new) break?

        Continuous fields are inferred by gradient descent, so a prediction within ``tol`` of the range of the
        true value is taken *as* the true value before ``ensures`` are checked; exact ``==`` on a float would
        otherwise flag 137.19 against 137.2. Discrete fields are always exact.
        """
        bad: list[str] = []
        cont = self.spec.continuous_fields()
        if cont and action is not None:
            truth = self.step(state, action)
            snapped = dict(new)
            for k in cont:
                f = self.spec.fields[k]
                if k in new and abs(float(new[k]) - float(truth[k])) <= tol * (f.hi - f.lo):
                    snapped[k] = truth[k]
            new = snapped
        for k, f in self.spec.fields.items():
            if k not in new or not f.contains(new[k]):
                bad.append(f"type:{k}={new.get(k)!r}")
        for src, tree in zip(self.spec.invariants, self.spec._invariants):
            try:
                if not eval_expr(tree, new, old=state):
                    bad.append(f"invariant:{src}")
            except Exception as e:  # noqa: BLE001
                bad.append(f"invariant-error:{src}:{e}")
        for e in self.spec.ensures:
            if e.after is None or (action is not None and e.after == action[0]):
                env = dict(new)
                if action is not None:
                    env.update(action[1])
                try:
                    if not eval_expr(e._holds, env, old=state):
                        bad.append(f"ensures:{e.after or '*'}:{e.holds}")
                except Exception as ex:  # noqa: BLE001
                    bad.append(f"ensures-error:{e.holds}:{ex}")
        return bad

    def self_check(self) -> list[str]:
        """Run the spec against itself over the whole finite state space.

        A spec whose own transitions break its own invariants is rejected
        before any training happens — the formal-verification analogue.
        """
        problems: list[str] = []
        if not self.spec.finite:
            import random
            rng = random.Random(0)
            for _ in range(2000):
                s = self.spec.sample_state(rng); a = self.spec.sample_action(rng)
                if self.violations(s, None, s):
                    continue
                n = self.step(s, a)
                for v in self.violations(s, a, n):
                    problems.append(f"{a[0]}{a[1] or ''} from {s}: {v}")
            return problems
        for s in self.spec.state_space():
            if self.violations(s, None, s) and self.violations(s, None, s)[0].startswith("invariant"):
                continue  # unreachable-by-construction states are allowed to violate invariants
            for a in self.spec.action_instances():
                n = self.step(s, a)
                for v in self.violations(s, a, n):
                    problems.append(f"{a[0]}{a[1] or ''} from {s}: {v}")
        return problems

    # ---- table --------------------------------------------------------
    def transition_table(self) -> dict[tuple, dict[tuple, tuple]]:
        """Full (state_key, action_key) -> next_state_key table for finite specs."""
        sp = self.spec
        table: dict[tuple, dict[tuple, tuple]] = {}
        for s in sp.state_space():
            row: dict[tuple, tuple] = {}
            for a in sp.action_instances():
                row[sp.action_key(a)] = sp.state_key(self.step(s, a))
            table[sp.state_key(s)] = row
        return table
