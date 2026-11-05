"""Featurize states and actions for the network, and enumerate candidates.

Finite fields become one-hots; gridded floats a one-hot over their step grid.
*Continuous* fields (float with a range, no step) become a normalised scalar
plus Fourier features, differentiable in torch so inference can descend the
energy with respect to them. Actions are one-hot over discrete instances,
with continuous params as extra features.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .schema import Spec

N_FOURIER = 4
CONT_WIDTH = 1 + 2 * N_FOURIER


class Encoder:
    def __init__(self, spec: Spec):
        self.spec = spec
        self._state_slots: list[tuple[str, int, list[Any]]] = []     # discrete fields
        off = 0
        for k, f in spec.fields.items():
            if f.continuous:
                continue
            dom = f.domain()
            self._state_slots.append((k, off, dom))
            off += len(dom)
        self.disc_dim = off
        self.cont_fields = spec.continuous_fields()
        self.cont_ranges = [(spec.fields[k].lo, spec.fields[k].hi) for k in self.cont_fields]
        # ordinal fields: ints and gridded floats with more than two values also get a normalised value + Fourier
        # features beside their one-hot, so "same value" and "+1" are compositional for cells never seen in training
        self.ord_fields = [k for k, off_, dom in self._state_slots
                           if spec.fields[k].kind in ("int", "float") and len(dom) > 2]
        self._ord_proj = np.zeros((self.disc_dim, len(self.ord_fields)), np.float32)   # one-hot -> normalised value
        for j, k in enumerate(self.ord_fields):
            off_, dom = next((o, d) for kk, o, d in self._state_slots if kk == k)
            lo, hi = float(dom[0]), float(dom[-1])
            for i, v in enumerate(dom):
                self._ord_proj[off_ + i, j] = (float(v) - lo) / (hi - lo)
        self.ord_dim = len(self.ord_fields) * CONT_WIDTH
        self.state_dim = self.disc_dim + len(self.cont_fields) * CONT_WIDTH + self.ord_dim
        self._actions = spec.action_instances()
        self._action_index = {spec.action_key(a): i for i, a in enumerate(self._actions)}
        self.action_dim = len(self._actions)
        self.cont_params = [(a.name, pk) for a in spec.actions.values() for pk in a.continuous_params()]
        self.action_cont_dim = len(self.cont_params)
        if spec.finite:
            self._states = spec.state_space()
            self._state_index = {spec.state_key(s): i for i, s in enumerate(self._states)}
        else:
            self._states = spec.discrete_space()
            self._state_index = {}

    # ---- states -------------------------------------------------------
    def discrete(self, s: dict[str, Any]) -> np.ndarray:
        x = np.zeros(self.disc_dim, dtype=np.float32)
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

    def cont_norm(self, s: dict[str, Any]) -> np.ndarray:
        """Continuous field values normalised to [0, 1]."""
        return np.array([(s[k] - lo) / (hi - lo) for k, (lo, hi) in zip(self.cont_fields, self.cont_ranges)], np.float32)

    def cont_denorm(self, u: np.ndarray) -> dict[str, float]:
        return {k: float(lo + float(x) * (hi - lo)) for (k, (lo, hi)), x in zip(zip(self.cont_fields, self.cont_ranges), u)}

    @staticmethod
    def cont_feats_np(u: np.ndarray) -> np.ndarray:
        out = [u]
        for k in range(1, N_FOURIER + 1):
            out += [np.sin(2 * math.pi * k * u), np.cos(2 * math.pi * k * u)]
        return np.stack(out, -1).reshape(*u.shape[:-1], -1).astype(np.float32)

    @staticmethod
    def cont_feats_torch(u):
        import torch
        out = [u]
        for k in range(1, N_FOURIER + 1):
            out += [torch.sin(2 * math.pi * k * u), torch.cos(2 * math.pi * k * u)]
        return torch.stack(out, -1).reshape(*u.shape[:-1], -1)

    def state(self, s: dict[str, Any]) -> np.ndarray:
        d = self.discrete(s)
        parts = [d]
        if self.cont_fields:
            parts.append(self.cont_feats_np(self.cont_norm(s)))
        if self.ord_fields:
            parts.append(self.cont_feats_np(d @ self._ord_proj))
        return np.concatenate(parts).astype(np.float32) if len(parts) > 1 else d

    def state_torch(self, disc, cont_norm=None):
        """Differentiable featurisation: disc [.., disc_dim] one-hots, cont_norm [.., n_cont] in [0,1]."""
        import torch
        parts = [disc]
        if self.cont_fields:
            parts.append(self.cont_feats_torch(cont_norm))
        if self.ord_fields:
            parts.append(self.cont_feats_torch(disc @ torch.as_tensor(self._ord_proj, dtype=disc.dtype, device=disc.device)))
        return torch.cat(parts, -1) if len(parts) > 1 else disc

    def states(self, ss: list[dict[str, Any]]) -> np.ndarray:
        return np.stack([self.state(s) for s in ss]) if ss else np.zeros((0, self.state_dim), np.float32)

    def decode_state(self, x: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, off, dom in self._state_slots:
            j = int(np.argmax(x[off:off + len(dom)]))
            out[k] = dom[j]
        if self.cont_fields:
            out.update(self.cont_denorm(x[self.disc_dim:self.disc_dim + len(self.cont_fields) * CONT_WIDTH:CONT_WIDTH]))
        return out

    def state_id(self, s: dict[str, Any]) -> int:
        return self._state_index[self.spec.state_key(s)]

    def all_states(self) -> list[dict[str, Any]]:
        """Finite specs: every state. Continuous specs: every discrete combination (continuous fields absent)."""
        return self._states

    def all_state_vectors(self) -> np.ndarray:
        """Finite specs: full feature vectors. Continuous specs: one-hot part only (features are built in torch)."""
        if self.cont_fields:
            return np.stack([self.discrete(s) for s in self._states])
        return self.states(self._states)

    def layout(self) -> dict:
        """What the edge Worker needs to rebuild `state()` bit for bit."""
        return {"disc_dim": self.disc_dim, "n_fourier": N_FOURIER,
                "ordinal": [{"field": k, "offset": next(o for kk, o, d in self._state_slots if kk == k),
                             "values_norm": [float(self._ord_proj[next(o for kk, o, d in self._state_slots if kk == k) + i, j])
                                             for i in range(len(next(d for kk, o, d in self._state_slots if kk == k)))]}
                            for j, k in enumerate(self.ord_fields)]}

    # ---- actions ------------------------------------------------------
    def action(self, a: tuple[str, dict[str, Any]]) -> np.ndarray:
        x = np.zeros(self.action_dim, dtype=np.float32)
        x[self.action_id(a)] = 1.0
        return x

    def action_id(self, a: tuple[str, dict[str, Any]]) -> int:
        return self._action_index[self.spec.action_key(a)]

    def action_cont(self, a: tuple[str, dict[str, Any]]) -> np.ndarray:
        """Normalised continuous params of this action (0 for params of other actions)."""
        x = np.zeros(self.action_cont_dim, dtype=np.float32)
        name, params = a
        for i, (an, pk) in enumerate(self.cont_params):
            if an == name and pk in params:
                f = self.spec.actions[an].params[pk]
                x[i] = (params[pk] - f.lo) / (f.hi - f.lo)
        return x

    def all_actions(self) -> list[tuple[str, dict[str, Any]]]:
        return self._actions

    # ---- candidates (dragology futures) -------------------------------
    def candidates(self, s: dict[str, Any], a: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Candidate next states to score. For finite specs: every state.

        Futures narrow this to the states a gesture may lead to; the full
        space is the conservative default and is what we evaluate against.
        """
        return self._states
