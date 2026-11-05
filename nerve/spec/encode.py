"""Featurize states and actions for the network, and enumerate candidates.

Finite fields become one-hots, floats become a normalized scalar plus a
one-hot over their step grid. Actions are one-hot over concrete instances.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .schema import Spec


class Encoder:
    def __init__(self, spec: Spec):
        self.spec = spec
        self._state_slots: list[tuple[str, int, list[Any]]] = []
        off = 0
        for k, f in spec.fields.items():
            dom = f.domain()
            self._state_slots.append((k, off, dom))
            off += len(dom)
        self.state_dim = off
        self._actions = spec.action_instances()
        self._action_index = {spec.action_key(a): i for i, a in enumerate(self._actions)}
        self.action_dim = len(self._actions)
        self._states = spec.state_space()
        self._state_index = {spec.state_key(s): i for i, s in enumerate(self._states)}

    # ---- states -------------------------------------------------------
    def state(self, s: dict[str, Any]) -> np.ndarray:
        x = np.zeros(self.state_dim, dtype=np.float32)
        for k, off, dom in self._state_slots:
            v = s[k]
            f = self.spec.fields[k]
            if f.kind == "float":
                # snap to grid for the one-hot, keep exactness for finite grids
                j = int(round((v - f.lo) / f.step))
            else:
                j = dom.index(v)
            x[off + j] = 1.0
        return x

    def states(self, ss: list[dict[str, Any]]) -> np.ndarray:
        return np.stack([self.state(s) for s in ss]) if ss else np.zeros((0, self.state_dim), np.float32)

    def decode_state(self, x: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, off, dom in self._state_slots:
            j = int(np.argmax(x[off:off + len(dom)]))
            out[k] = dom[j]
        return out

    def state_id(self, s: dict[str, Any]) -> int:
        return self._state_index[self.spec.state_key(s)]

    def all_states(self) -> list[dict[str, Any]]:
        return self._states

    def all_state_vectors(self) -> np.ndarray:
        return self.states(self._states)

    # ---- actions ------------------------------------------------------
    def action(self, a: tuple[str, dict[str, Any]]) -> np.ndarray:
        x = np.zeros(self.action_dim, dtype=np.float32)
        x[self.action_id(a)] = 1.0
        return x

    def action_id(self, a: tuple[str, dict[str, Any]]) -> int:
        return self._action_index[self.spec.action_key(a)]

    def all_actions(self) -> list[tuple[str, dict[str, Any]]]:
        return self._actions

    # ---- candidates (dragology futures) -------------------------------
    def candidates(self, s: dict[str, Any], a: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Candidate next states to score. For finite specs: every state.

        Futures narrow this to the states a gesture may lead to; the full
        space is the conservative default and is what we evaluate against.
        """
        return self._states
