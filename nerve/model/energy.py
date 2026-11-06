"""The full energy: compiled circuit (exact where the spec is explicit) plus
the learned JEPA (holes and generalisation). Inference = argmin over
dragology-style candidate futures. For continuous fields the candidates are a
continuum: we enumerate the discrete part and descend the energy with respect
to the continuous part from a handful of seeds (dragology's ``vary``)."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..spec import Spec
from ..spec.circuit import CompiledCircuit, compile_circuit
from ..spec.encode import Encoder
from .jepa import JEPAConfig, NerveJEPA


class NerveEnergy:
    def __init__(self, spec: Spec, jepa: NerveJEPA, circuit: CompiledCircuit | None, circuit_weight: float = 1000.0):
        """``circuit_weight`` is large on purpose: the circuit is exact by construction, so where it speaks it wins.
        Where it is silent (``learn`` holes) its energy is constant and the JEPA decides."""
        self.spec = spec
        self.enc = Encoder(spec)
        self.jepa = jepa
        self.circuit = circuit
        self.w = circuit_weight
        self._cand = torch.from_numpy(self.enc.all_state_vectors())
        self.cont_steps = 60          # gradient steps on continuous fields
        self.cont_lr = 0.05

    @classmethod
    def build(cls, spec: Spec, cfg: JEPAConfig | None = None, compile: bool = True) -> "NerveEnergy":
        enc = Encoder(spec)
        cfg = cfg or JEPAConfig(state_dim=enc.state_dim, action_dim=enc.action_dim, action_cont_dim=enc.action_cont_dim,
                                ord_dim=len(enc.ord_fields))
        circuit = None
        if compile and spec.finite:
            try:
                circuit = compile_circuit(spec, exclude_actions={h.action for h in spec.holes.values()})
            except ValueError:
                circuit = None
        return cls(spec, NerveJEPA(cfg), circuit)

    def _a_cont(self, a) -> torch.Tensor | None:
        return torch.from_numpy(self.enc.action_cont(a))[None] if self.enc.action_cont_dim else None

    def energies(self, s: dict[str, Any], a: tuple[str, dict[str, Any]]) -> torch.Tensor:
        if not self.spec.finite:
            return self._continuous(s, a)[1]
        xs = torch.from_numpy(self.enc.state(s))[None]
        ai = torch.tensor([self.enc.action_id(a)])
        with torch.no_grad():
            E = self.jepa.energy_all(xs, ai, self._cand, self._a_cont(a))[0]
            if self.circuit is not None:
                xa = torch.from_numpy(self.enc.action(a))[None]
                E = E + self.w * self.circuit.energy_all(xs, xa, self._cand)[0]
        return E

    def step(self, s: dict[str, Any], a: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        if not self.spec.finite:
            return self._continuous(s, a)[0]
        E = self.energies(s, a)
        return dict(self.enc.all_states()[int(E.argmin())])

    def step_learned_only(self, s: dict[str, Any], a: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        if not self.spec.finite:
            return self._continuous(s, a)[0]
        xs = torch.from_numpy(self.enc.state(s))[None]
        ai = torch.tensor([self.enc.action_id(a)])
        with torch.no_grad():
            E = self.jepa.energy_all(xs, ai, self._cand, self._a_cont(a))[0]
        return dict(self.enc.all_states()[int(E.argmin())])

    def _continuous(self, s: dict[str, Any], a: tuple[str, dict[str, Any]]) -> tuple[dict[str, Any], torch.Tensor]:
        """Argmin over (discrete combos) x (continuous values found by gradient descent from seeds).

        Seeds per continuous field: the current value, the action's continuous params, lo, hi, mid, and a few
        random points. Each (discrete combo, seed) descends the energy in parallel; the global minimum wins.
        Returns (next_state, per-discrete-combo minimum energies).
        """
        enc = self.enc
        xs = torch.from_numpy(enc.state(s))[None]
        ai = torch.tensor([enc.action_id(a)])
        with torch.no_grad():
            zh = self.jepa.predict(xs, ai, self._a_cont(a))                     # [1, d]
        disc = self._cand                                                       # [D, disc_dim]
        D = disc.shape[0]
        cur = enc.cont_norm(s)
        seeds = [cur, np.zeros_like(cur), np.ones_like(cur), np.full_like(cur, 0.5)]
        for (an, pk) in enc.cont_params:
            if an == a[0] and pk in a[1]:
                f = self.spec.actions[an].params[pk]
                seeds.append(np.full_like(cur, (a[1][pk] - f.lo) / (f.hi - f.lo)))
        rng = np.random.default_rng(0)
        seeds += [rng.random(cur.shape).astype(np.float32) for _ in range(6)]
        S = len(seeds)
        u = torch.tensor(np.stack(seeds), dtype=torch.float32)                  # [S, n_cont]
        u = u[None].expand(D, S, -1).reshape(D * S, -1).clone().requires_grad_(True)
        dd = disc[:, None, :].expand(D, S, -1).reshape(D * S, -1)
        opt = torch.optim.Adam([u], lr=self.cont_lr)
        for _ in range(self.cont_steps):
            opt.zero_grad()
            E = self.jepa.energy_grad(zh, enc.state_torch(dd, u))
            E.sum().backward()
            opt.step()
            with torch.no_grad():
                u.clamp_(0.0, 1.0)
        with torch.no_grad():
            E = self.jepa.energy_grad(zh, enc.state_torch(dd, u)).reshape(D, S)
            best = int(E.min(1).values.argmin()); j = int(E[best].argmin())
            out = dict(enc.all_states()[best]) if enc.disc_dim else {}
            out.update(enc.cont_denorm(u.reshape(D, S, -1)[best, j].numpy()))
            return out, E.min(1).values

    # ---- persistence (BF16 checkpoints, PULSE-ready) --------------------
    def save(self, path: str) -> None:
        torch.save({
            "spec": self.spec.name,
            "jepa_cfg": self.jepa.config_dict(),
            "jepa": {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in self.jepa.state_dict().items()},
            "circuit": None if self.circuit is None else {k: v for k, v in self.circuit.state_dict().items()},
            "circuit_weight": self.w,
        }, path)

    @classmethod
    def load(cls, spec: Spec, path: str) -> "NerveEnergy":
        ck = torch.load(path, map_location="cpu")
        jepa = NerveJEPA(JEPAConfig(**ck["jepa_cfg"]))
        jepa.load_state_dict({k: v.to(torch.float32) if v.is_floating_point() else v for k, v in ck["jepa"].items()})
        jepa.eval()
        circuit = None
        if ck["circuit"] is not None:
            c = ck["circuit"]
            circuit = CompiledCircuit(c["ws"], c["wa"], c["wo"], float(c["bias"]))
        return cls(spec, jepa, circuit, ck.get("circuit_weight", 1000.0))
