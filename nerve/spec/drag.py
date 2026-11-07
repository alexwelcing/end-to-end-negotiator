"""A drag as one minimisation per pointer frame (docs/dragology.md).

Dragology's operators each return, per frame, a preview state, a drop state and a *gap*: the distance from the
pointer to where the grabbed element lands in that state. ``closest`` is the argmin of the gap over candidate
states, ``between`` its convex relaxation, ``vary`` gradient descent on it. The gap is an energy, so here a drag is

    s'(p) = argmin over C(s, g) of   E_spec(s, g, s')  +  gap(render(s'), p)  +  E_mem

with the candidates C derived from the spec (every candidate is a successor the oracle reaches from ``s``; nothing
is hand-listed), ``E_spec`` zero on legal successors and infinite elsewhere (a trained network's energies can be
passed instead to break ties where the spec is silent), and ``E_mem`` the operator memory: stickiness as a bonus
on the incumbent branch, ``far`` as a two-state machine with hysteresis. Pure Python, no DOM: ``render`` is the
spec's declarative block of position expressions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from .expr import compile_expr, eval_expr
from .oracle import Oracle
from .schema import Future, Spec

Point = tuple[float, float]


def element_position(spec: Spec, element: str, state: dict[str, Any]) -> Point:
    """Where ``element`` is drawn in ``state`` according to the spec's ``render`` block."""
    r = spec.render[element]
    x = float(eval_expr(compile_expr(str(r["x"])), state)); y = float(eval_expr(compile_expr(str(r["y"])), state))
    return (x, y)


def gap(spec: Spec, element: str, state: dict[str, Any], pointer: Point) -> float:
    x, y = element_position(spec, element, state)
    return math.hypot(x - pointer[0], y - pointer[1])


@dataclass
class Candidate:
    state: dict[str, Any]
    action: tuple[str, dict[str, Any]] | None     # the action instance that realises it (None = stay)


@dataclass
class Frame:
    preview: dict[str, Any]
    drop: dict[str, Any]
    gap: float
    branch: Any                                   # which candidate value is active
    far: bool
    weights: dict[Any, float] = field(default_factory=dict)   # `between`: interpolation weights over values


class DragSession:
    """One gesture from ``start`` to release. ``energy`` optionally scores candidates (e.g. a trained network's
    learned-only energies) and is added to the gap; the oracle still decides legality."""

    def __init__(self, spec: Spec, future: Future, start: dict[str, Any],
                 energy: Callable[[dict[str, Any], tuple[str, dict[str, Any]] | None, dict[str, Any]], float] | None = None):
        if future.grab is None or future.grab not in spec.render:
            raise ValueError(f"future {future.name} needs `grab: <element>` with a `render` entry to have a geometry")
        self.spec, self.f, self.start = spec, future, dict(start)
        self.oracle = Oracle(spec)
        self.energy = energy
        self.last_branch: Any = None
        self.far = False
        self.frames: list[Frame] = []

    # ---- candidates: derived from the spec, never hand-listed --------------------------------------
    def candidates(self) -> dict[Any, Candidate]:
        """value of the dragged field -> the legal candidate reaching it (plus `stay`)."""
        s, f, sp = self.start, self.f, self.spec
        ft = sp.fields[f.field_name]
        values = list(f.values) if f.values is not None else (ft.domain() if not ft.continuous else [])
        out: dict[Any, Candidate] = {ft.canon(s[f.field_name]): Candidate(dict(s), None)}
        actions = [a for a in sp.action_instances() if f.action is None or a[0] == f.action]
        for v in values:
            if ft.canon(v) in out:
                continue
            for a in actions:
                n = self.oracle.step(s, a)
                if ft.canon(n[f.field_name]) == ft.canon(v) and not self.oracle.violations(s, a, n):
                    out[ft.canon(v)] = Candidate(n, a)
                    break
        return out

    def _score(self, c: Candidate, pointer: Point) -> float:
        e = gap(self.spec, self.f.grab, c.state, pointer)
        if self.energy is not None:
            e += float(self.energy(self.start, c.action, c.state))
        return e

    # ---- one frame -----------------------------------------------------------------------------------
    def frame(self, pointer: Point) -> Frame:
        f = self.f
        if f.kind == "vary":
            fr = self._vary(pointer)
        else:
            cands = self.candidates()
            scores = {v: self._score(c, pointer) for v, c in cands.items()}
            best = min(scores, key=scores.get)
            if f.stickiness and self.last_branch in scores and scores[best] + f.stickiness >= scores[self.last_branch]:
                best = self.last_branch                       # dragology `stickiness`: the incumbent keeps the branch
            weights: dict[Any, float] = {}
            if f.kind == "between" and len(cands) >= 2:
                weights = self._between_weights(cands, pointer)
            fr = Frame(preview=dict(cands[best].state), drop=dict(cands[best].state), gap=scores[best], branch=best,
                       far=False, weights=weights)
        # dragology `whenFar`: beyond `far` the gesture falls back to the start state; hysteresis at far / 1.2 far
        if f.far is not None:
            if not self.far and fr.gap > 1.2 * f.far:
                self.far = True
            elif self.far and fr.gap <= f.far:
                self.far = False
            if self.far:
                fr = Frame(preview=dict(self.start), drop=dict(self.start), gap=fr.gap, branch=None, far=True)
        self.last_branch = fr.branch
        self.frames.append(fr)
        return fr

    def _between_weights(self, cands: dict[Any, Candidate], pointer: Point) -> dict[Any, float]:
        """Convex weights over candidate values. Two candidates: the pointer's projection onto their segment,
        clamped (dragology interpolates the whole SVG with these; we only report them). More than two: inverse-gap
        weights, a stand-in for dragology's Delaunay / natural-neighbour scheme."""
        pos = {v: element_position(self.spec, self.f.grab, c.state) for v, c in cands.items()}
        if len(pos) == 2:
            (va, pa), (vb, pb) = pos.items()
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((pointer[0] - pa[0]) * dx + (pointer[1] - pa[1]) * dy) / L2))
            return {va: 1 - t, vb: t}
        inv = {v: 1.0 / (1e-6 + math.hypot(p[0] - pointer[0], p[1] - pointer[1])) for v, p in pos.items()}
        z = sum(inv.values())
        return {v: w / z for v, w in inv.items()}

    def _vary(self, pointer: Point) -> Frame:
        """Minimise the gap over a continuous field: coarse grid, then golden-section refinement, then the oracle
        realises the value through the gesture's action (which applies the spec's own clamps and guards)."""
        f, sp, s = self.f, self.spec, self.start
        ft = sp.fields[f.field_name]
        lo, hi = float(ft.lo), float(ft.hi)
        if f.action:
            act = sp.actions[f.action]
        else:   # the action with a float parameter of the field's kind, e.g. slider's `move(to)`
            act = next(a for a in sp.actions.values() if any(p.kind == "float" for p in a.params.values()))
        param = f.param or next(pk for pk, p in act.params.items() if p.kind == "float")

        def realise(u: float) -> tuple[dict[str, Any], tuple[str, dict[str, Any]]]:
            a = (act.name, {param: ft.canon(u) if ft.step else u})
            return self.oracle.step(s, a), a

        def score(u: float) -> float:
            n, a = realise(u)
            return gap(sp, f.grab, n, pointer) + (float(self.energy(s, a, n)) if self.energy else 0.0)

        grid = [lo + (hi - lo) * i / 64 for i in range(65)]
        best = min(grid, key=score)
        a_, b_ = max(lo, best - (hi - lo) / 64), min(hi, best + (hi - lo) / 64)
        g = (math.sqrt(5) - 1) / 2
        for _ in range(40):
            c1, c2 = b_ - g * (b_ - a_), a_ + g * (b_ - a_)
            if score(c1) < score(c2):
                b_ = c2
            else:
                a_ = c1
        u = (a_ + b_) / 2
        n, a = realise(u)
        return Frame(preview=dict(n), drop=dict(n), gap=gap(sp, f.grab, n, pointer), branch=ft.canon(n[f.field_name]), far=False)

    # ---- release -------------------------------------------------------------------------------------
    def release(self) -> dict[str, Any]:
        """The drop state is the last frame's; it is a spec transition like any other and is checked as one."""
        drop = self.frames[-1].drop if self.frames else dict(self.start)
        bad = self.oracle.violations(self.start, None, drop)
        if bad:
            raise ValueError(f"drag {self.f.name} would drop into a violating state: {bad}")
        return drop


def drag(spec: Spec, gesture: str, start: dict[str, Any], path: list[Point], energy=None) -> tuple[list[Frame], dict[str, Any]]:
    """Run a whole gesture along a pointer path; returns the frames and the committed state."""
    sess = DragSession(spec, spec.futures[gesture], start, energy)
    for p in path:
        sess.frame(p)
    return sess.frames, sess.release()
