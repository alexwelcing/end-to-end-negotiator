"""Compile a finite spec into exact network weights — no training.

For a finite state space S and action set A the transition function is a
table T: S x A -> S. We realise it as a two-layer ReLU network whose weights
are *computed*, in the spirit of qwenGPU (a compiler fills the weights) and
ink's ``emit-circuit`` (finite-type functions lowered to exact ReLU nets):

    h  = relu(onehot(s) . Ws + onehot(a) . Wa - n)     one unit per (s, a) pair,
                                                       n = number of state fields
    y  = h . Wo                                        onehot(T(s, a))
    E(s, a, s') = 1 - <onehot(s'), y> / n              0 on the true successor, >0 elsewhere

This is the deterministic core of the interface, exact by construction. The
learned JEPA model sits beside it and covers ``learn`` holes and anything the
spec leaves open. Memory is |S|*|A| hidden units, so we only compile when
that is modest; the learned model carries larger interfaces alone.
"""
from __future__ import annotations

import numpy as np
import torch

from .encode import Encoder
from .oracle import Oracle
from .schema import Spec


class CompiledCircuit(torch.nn.Module):
    def __init__(self, ws: torch.Tensor, wa: torch.Tensor, wo: torch.Tensor, bias: float):
        super().__init__()
        self.register_buffer("ws", ws)   # [state_dim, H]
        self.register_buffer("wa", wa)   # [action_dim, H]
        self.register_buffer("wo", wo)   # [H, state_dim]
        self.register_buffer("bias", torch.tensor(float(bias)))   # = number of state fields

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return the one-hot successor prediction y, shape [B, state_dim]."""
        h = torch.relu(s @ self.ws + a @ self.wa - self.bias)
        return h @ self.wo

    def energy(self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """E(s, a, s') = 1 - <onehot(s'), y>, per row."""
        y = self.forward(s, a)
        return 1.0 - (y * s_next).sum(-1) / self.bias

    def energy_all(self, s: torch.Tensor, a: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Energy of every candidate: s,a [B,.], cand [C,state_dim] -> [B,C]."""
        y = self.forward(s, a)
        return 1.0 - (y @ cand.T) / self.bias


def compile_circuit(spec: Spec, max_hidden: int = 200_000, exclude_actions: set[str] | None = None) -> CompiledCircuit:
    """``exclude_actions`` (e.g. ``learn`` holes) get all-zero output rows, so their
    energy is constant over candidates: the circuit says nothing and the learned
    model decides."""
    if not spec.finite:
        raise ValueError("only finite specs can be compiled to an exact circuit")
    enc = Encoder(spec)
    oracle = Oracle(spec)
    states = enc.all_states()
    actions = enc.all_actions()
    H = len(states) * len(actions)
    if H > max_hidden:
        raise ValueError(f"circuit too large to compile ({H} hidden units > {max_hidden})")
    ws = np.zeros((enc.state_dim, H), np.float32)
    wa = np.zeros((enc.action_dim, H), np.float32)
    wo = np.zeros((H, enc.state_dim), np.float32)
    nf = len(spec.fields)
    k = 0
    for s in states:
        xs = enc.state(s)
        for a in actions:
            xa = enc.action(a)
            # one unit fires iff all nf state one-hots and the action one-hot are on:
            # each matched field adds 1 (nf total), the action adds 1, bias is -nf -> relu(1).
            # any mismatch gives at most nf - nf = 0.
            ws[:, k] = xs
            wa[:, k] = xa
            if not exclude_actions or a[0] not in exclude_actions:
                wo[k, :] = enc.state(oracle.step(s, a))
            k += 1
    return CompiledCircuit(torch.from_numpy(ws), torch.from_numpy(wa), torch.from_numpy(wo), bias=float(nf))
