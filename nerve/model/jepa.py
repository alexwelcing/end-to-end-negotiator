"""Action-conditioned JEPA whose prediction error is the energy.

    z   = f(s)                     online encoder
    z'  = f_ema(s')                target encoder (EMA, no gradient)
    ẑ'  = g(z, a)                  predictor
    E(s, a, s') = || ẑ' - z' ||²

The network never reconstructs pixels or DOM; it predicts in representation
space (JEPA) and the interface state is read out by scoring candidates
(energy-based inference). Collapse is prevented by the contrastive term over
candidates and a light VICReg variance term.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class JEPAConfig:
    state_dim: int
    action_dim: int
    d: int = 64
    hidden: int = 128
    ema: float = 0.99
    tau: float = 0.1            # InfoNCE temperature
    margin: float = 1.0         # hinge margin for spec-violating negatives
    var_weight: float = 0.05    # VICReg-style variance regulariser


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class NerveJEPA(nn.Module):
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = _mlp(cfg.state_dim, cfg.hidden, cfg.d)
        self.act = nn.Embedding(cfg.action_dim, cfg.d)
        self.pred = _mlp(2 * cfg.d, cfg.hidden, cfg.d)
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters():
            p.requires_grad_(False)

    # ---- pieces -------------------------------------------------------
    def predict(self, s: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        z = self.enc(s)
        return self.pred(torch.cat([z, self.act(a_idx)], -1))

    @torch.no_grad()
    def target(self, s_next: torch.Tensor) -> torch.Tensor:
        return self.tgt(s_next)

    def energy(self, s: torch.Tensor, a_idx: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        return ((self.predict(s, a_idx) - self.target(s_next)) ** 2).sum(-1)

    def energy_all(self, s: torch.Tensor, a_idx: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """[B, C] energies of every candidate next state."""
        zh = self.predict(s, a_idx)                       # [B, d]
        zc = self.target(cand)                            # [C, d]
        return torch.cdist(zh, zc) ** 2

    @torch.no_grad()
    def ema_update(self) -> None:
        m = self.cfg.ema
        for p_t, p_o in zip(self.tgt.parameters(), self.enc.parameters()):
            p_t.mul_(m).add_(p_o.detach(), alpha=1 - m)

    # ---- loss ---------------------------------------------------------
    def loss(self, s: torch.Tensor, a_idx: torch.Tensor, pos_idx: torch.Tensor, cand: torch.Tensor,
             neg: torch.Tensor | None = None, neg_s: torch.Tensor | None = None,
             neg_a: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """s,a -> true candidate index pos_idx among `cand`; optional explicit negatives."""
        E = self.energy_all(s, a_idx, cand)                                   # [B, C]
        nce = F.cross_entropy(-E / self.cfg.tau, pos_idx)                      # argmin-energy = oracle
        jepa = E.gather(1, pos_idx[:, None]).mean()                            # pull true successor to 0
        out = {"nce": nce, "jepa": jepa}
        total = nce + 0.1 * jepa
        if neg is not None and len(neg):
            e_neg = self.energy(neg_s, neg_a, neg)
            e_pos = E.gather(1, pos_idx[:, None]).squeeze(1)
            # negatives come paired with their positive by construction (same s, a)
            hinge = F.relu(self.cfg.margin + e_pos.mean() - e_neg).mean()
            out["hinge"] = hinge
            total = total + hinge
        z = self.enc(s)
        std = torch.sqrt(z.var(0) + 1e-4)
        var = F.relu(1 - std).mean()
        out["var"] = var
        total = total + self.cfg.var_weight * var
        out["total"] = total
        return out

    def config_dict(self) -> dict:
        return asdict(self.cfg)
