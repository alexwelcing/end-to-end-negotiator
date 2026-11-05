"""The full energy: compiled circuit (exact where the spec is explicit) plus
the learned JEPA (holes and generalisation). Inference = argmin over
dragology-style candidate futures."""
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

    @classmethod
    def build(cls, spec: Spec, cfg: JEPAConfig | None = None, compile: bool = True) -> "NerveEnergy":
        enc = Encoder(spec)
        cfg = cfg or JEPAConfig(state_dim=enc.state_dim, action_dim=enc.action_dim)
        circuit = None
        if compile and spec.finite:
            try:
                circuit = compile_circuit(spec, exclude_actions={h.action for h in spec.holes.values()})
            except ValueError:
                circuit = None
        return cls(spec, NerveJEPA(cfg), circuit)

    def energies(self, s: dict[str, Any], a: tuple[str, dict[str, Any]]) -> torch.Tensor:
        xs = torch.from_numpy(self.enc.state(s))[None]
        ai = torch.tensor([self.enc.action_id(a)])
        with torch.no_grad():
            E = self.jepa.energy_all(xs, ai, self._cand)[0]
            if self.circuit is not None:
                xa = torch.from_numpy(self.enc.action(a))[None]
                E = E + self.w * self.circuit.energy_all(xs, xa, self._cand)[0]
        return E

    def step(self, s: dict[str, Any], a: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        E = self.energies(s, a)
        return dict(self.enc.all_states()[int(E.argmin())])

    def step_learned_only(self, s: dict[str, Any], a: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        xs = torch.from_numpy(self.enc.state(s))[None]
        ai = torch.tensor([self.enc.action_id(a)])
        with torch.no_grad():
            E = self.jepa.energy_all(xs, ai, self._cand)[0]
        return dict(self.enc.all_states()[int(E.argmin())])

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
