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
    action_cont_dim: int = 0    # number of continuous action params (0 for finite specs)
    ord_dim: int = 0            # number of ordinal fields (ints / gridded floats) read out linearly from the latent
    ord_weight: float = 1.0     # weight of the ordinal readout auxiliary loss


def sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances [N, M] with no square root (finite gradient everywhere, unlike cdist at 0)."""
    d = (a * a).sum(-1)[:, None] + (b * b).sum(-1)[None, :] - 2.0 * a @ b.T
    return d.clamp_min(0.0)


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class NerveJEPA(nn.Module):
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = _mlp(cfg.state_dim, cfg.hidden, cfg.d)
        self.act = nn.Embedding(cfg.action_dim, cfg.d)
        # continuous action params (e.g. `move to=137.2`) enter through a linear map added to the action embedding
        self.act_cont = nn.Linear(cfg.action_cont_dim, cfg.d, bias=False) if cfg.action_cont_dim else None
        self.pred = _mlp(2 * cfg.d, cfg.hidden, cfg.d)
        # ordinal readout: a linear map from latent to each ordinal field's normalised value. Trained on both the
        # encoder's latent and the *predicted* next latent, it makes "+1" a translation in latent space, so cells
        # never seen in training (inc at the top of the range) follow the same rule as the ones that were.
        self.ord_head = nn.Linear(cfg.d, cfg.ord_dim) if cfg.ord_dim else None
        self.tgt = copy.deepcopy(self.enc)
        for p in self.tgt.parameters():
            p.requires_grad_(False)

    # ---- pieces -------------------------------------------------------
    def predict(self, s: torch.Tensor, a_idx: torch.Tensor, a_cont: torch.Tensor | None = None) -> torch.Tensor:
        z = self.enc(s)
        za = self.act(a_idx)
        if self.act_cont is not None and a_cont is not None:
            za = za + self.act_cont(a_cont)
        return self.pred(torch.cat([z, za], -1))

    @torch.no_grad()
    def target(self, s_next: torch.Tensor) -> torch.Tensor:
        return self.tgt(s_next)

    def energy(self, s: torch.Tensor, a_idx: torch.Tensor, s_next: torch.Tensor, a_cont: torch.Tensor | None = None) -> torch.Tensor:
        return ((self.predict(s, a_idx, a_cont) - self.target(s_next)) ** 2).sum(-1)

    def energy_all(self, s: torch.Tensor, a_idx: torch.Tensor, cand: torch.Tensor, a_cont: torch.Tensor | None = None) -> torch.Tensor:
        """[B, C] energies of every candidate next state."""
        zh = self.predict(s, a_idx, a_cont)               # [B, d]
        zc = self.target(cand)                            # [C, d]
        return sqdist(zh, zc)

    def energy_grad(self, zh: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Energy with gradient flowing into s_next (for descent over continuous fields). zh is a fixed prediction."""
        return ((zh - self.tgt(s_next)) ** 2).sum(-1)

    @torch.no_grad()
    def ema_update(self) -> None:
        m = self.cfg.ema
        for p_t, p_o in zip(self.tgt.parameters(), self.enc.parameters()):
            p_t.mul_(m).add_(p_o.detach(), alpha=1 - m)

    # ---- loss ---------------------------------------------------------
    def loss(self, s: torch.Tensor, a_idx: torch.Tensor, pos_idx: torch.Tensor, cand: torch.Tensor,
             neg: torch.Tensor | None = None, neg_s: torch.Tensor | None = None,
             neg_a: torch.Tensor | None = None, ord_s: torch.Tensor | None = None,
             ord_next: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """s,a -> true candidate index pos_idx among `cand`; optional explicit negatives; optional ordinal targets."""
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
        # variance regulariser needs >= 2 samples: var() of one row is NaN and poisons the whole run
        var = F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean() if z.shape[0] > 1 else z.new_zeros(())
        out["var"] = var
        total = total + self.cfg.var_weight * var
        if self.ord_head is not None and ord_next is not None:
            zh = self.predict(s, a_idx)
            aux = F.mse_loss(self.ord_head(zh), ord_next) + F.mse_loss(self.ord_head(z), ord_s)
            out["ord"] = aux
            total = total + self.cfg.ord_weight * aux
        out["total"] = total
        return out

    def loss_continuous(self, s: torch.Tensor, a_idx: torch.Tensor, a_cont: torch.Tensor, s_next: torch.Tensor,
                        negs: torch.Tensor) -> dict[str, torch.Tensor]:
        """For specs with continuous fields there is no finite candidate set. Contrast the true successor
        against in-batch successors and K sampled near-miss perturbations: negs is [B, K, state_dim]."""
        zh = self.predict(s, a_idx, a_cont)                          # [B, d]
        zp = self.target(s_next)                                     # [B, d]
        B, K, D = negs.shape
        zn = self.target(negs.reshape(B * K, D)).reshape(B, K, -1)   # [B, K, d]
        e_pos = ((zh - zp) ** 2).sum(-1)                             # [B]
        e_batch = sqdist(zh, zp)                                     # [B, B], diagonal = e_pos
        e_neg = ((zh[:, None, :] - zn) ** 2).sum(-1)                 # [B, K]
        logits = -torch.cat([e_batch, e_neg], 1) / self.cfg.tau
        nce = F.cross_entropy(logits, torch.arange(B, device=s.device))
        z = self.enc(s)
        var = F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean() if z.shape[0] > 1 else z.new_zeros(())
        total = nce + 0.1 * e_pos.mean() + self.cfg.var_weight * var
        return {"nce": nce, "jepa": e_pos.mean(), "var": var, "total": total}

    def config_dict(self) -> dict:
        return asdict(self.cfg)
