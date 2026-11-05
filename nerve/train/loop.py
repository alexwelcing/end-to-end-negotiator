"""Train the JEPA energy from the episode log. A run is a pure function of
(spec, log, config, seed): no hidden state anywhere else."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data import EpisodeLog
from ..model import JEPAConfig, NerveEnergy
from ..spec import Spec
from ..spec.encode import Encoder
from .eval import evaluate, holdout_mask


@dataclass
class TrainConfig:
    epochs: int = 60
    batch: int = 128
    lr: float = 3e-3
    d: int = 64
    hidden: int = 128
    ema: float = 0.99
    tau: float = 0.1
    holdout: float = 0.15
    seed: int = 0
    compile_circuit: bool = True
    hole_weight: int = 20          # oversample hole examples (there are very few)
    log_every: int = 10
    init: str | None = None        # warm-start JEPA weights from a checkpoint (continued training)
    device: str = "auto"           # auto | cpu | cuda | mps


def _tensors(spec: Spec, enc: Encoder, log: EpisodeLog, cfg: TrainConfig):
    S, A, P, NS, NA, N = [], [], [], [], [], []
    for st in log:
        a = (st.action[0], st.action[1])
        if holdout_mask(spec, st.state, a, cfg.holdout):
            continue
        if st.ok:
            reps = cfg.hole_weight if st.source == "hole-example" else 1
            for _ in range(reps):
                S.append(enc.state(st.state)); A.append(enc.action_id(a)); P.append(enc.state_id(st.next))
        else:
            NS.append(enc.state(st.state)); NA.append(enc.action_id(a)); N.append(enc.state(st.next))
    t = lambda x, dt=torch.float32: torch.tensor(np.array(x), dtype=dt)
    return (t(S), t(A, torch.long), t(P, torch.long)), (t(NS), t(NA, torch.long), t(N))


def _tensors_continuous(spec: Spec, enc: Encoder, log: EpisodeLog, k_neg: int = 16, seed: int = 0):
    """Continuous specs: positives as (s, a_idx, a_cont, s_next); K near-miss negatives per positive built
    from the logged negatives plus fresh perturbations, so the contrastive set is never empty."""
    import random
    from ..data.rollout import _perturb
    rng = random.Random(seed)
    S, A, AC, NX, NEG = [], [], [], [], []
    logged_negs: dict[tuple, list[dict]] = {}
    pos: list = []
    for st in log:
        a = (st.action[0], st.action[1])
        if st.ok:
            pos.append((st.state, a, st.next))
        else:
            logged_negs.setdefault((json.dumps(st.state, sort_keys=True), json.dumps(a, sort_keys=True)), []).append(st.next)
    for s, a, n in pos:
        negs = list(logged_negs.get((json.dumps(s, sort_keys=True), json.dumps(a, sort_keys=True)), []))
        while len(negs) < k_neg:
            negs.append(_perturb(spec, n, rng))
        negs = negs[:k_neg]
        S.append(enc.state(s)); A.append(enc.action_id(a)); AC.append(enc.action_cont(a)); NX.append(enc.state(n))
        NEG.append(np.stack([enc.state(x) for x in negs]))
    t = lambda x, dt=torch.float32: torch.tensor(np.array(x), dtype=dt)
    return t(S), t(A, torch.long), t(AC), t(NX), t(NEG)


def _device(cfg: "TrainConfig") -> torch.device:
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(spec: Spec, log: EpisodeLog, out_dir: str | Path, cfg: TrainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    enc = Encoder(spec)
    model = NerveEnergy.build(spec, JEPAConfig(enc.state_dim, enc.action_dim, d=cfg.d, hidden=cfg.hidden,
                                               ema=cfg.ema, tau=cfg.tau, action_cont_dim=enc.action_cont_dim),
                              compile=cfg.compile_circuit and spec.finite)
    if not spec.finite:
        return _train_continuous(spec, log, out, cfg, enc, model)
    if cfg.init:
        prev = NerveEnergy.load(spec, cfg.init)
        model.jepa.load_state_dict(prev.jepa.state_dict())
    (S, A, P), (NS, NA, N) = _tensors(spec, enc, log, cfg)
    dev = _device(cfg)
    model.jepa.to(dev)
    S, A, P, NS, NA, N = (x.to(dev) for x in (S, A, P, NS, NA, N))
    cand = torch.from_numpy(enc.all_state_vectors()).to(dev)
    opt = torch.optim.AdamW(model.jepa.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)
    n = len(S)
    history: list[dict[str, Any]] = []
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.jepa.train()
        perm = torch.randperm(n, device=dev)
        tot = {"total": 0.0, "nce": 0.0}
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            neg = None
            if len(NS):
                j = torch.randint(0, len(NS), (len(idx),), device=dev)
                neg = (N[j], NS[j], NA[j])
            losses = model.jepa.loss(S[idx], A[idx], P[idx], cand,
                                     neg=None if neg is None else neg[0],
                                     neg_s=None if neg is None else neg[1],
                                     neg_a=None if neg is None else neg[2])
            opt.zero_grad(); losses["total"].backward(); opt.step(); model.jepa.ema_update()
            tot["total"] += losses["total"].item() * len(idx); tot["nce"] += losses["nce"].item() * len(idx)
        sched.step()
        rec = {"epoch": ep, "loss": tot["total"] / n, "nce": tot["nce"] / n}
        if ep % cfg.log_every == 0 or ep == cfg.epochs - 1:
            model.jepa.eval(); model.jepa.cpu()
            ev = evaluate(spec, model.step, cfg.holdout, learned_only_fn=model.step_learned_only)
            model.jepa.to(dev)
            rec.update({"heldout": ev["heldout"]["exact"], "all": ev["all"]["exact"],
                        "learned_only": ev["learned_only"]["exact"], "score": ev["score"]})
            print(f"  ep {ep:3d} loss {rec['loss']:.4f} nce {rec['nce']:.4f} | exact all {rec['all']:.3f} "
                  f"heldout {rec['heldout']:.3f} learned-only {rec['learned_only']:.3f} score {rec['score']:.3f}")
        history.append(rec)
    return _finish(spec, model, out, cfg, history, t0, n, len(NS))


def _finish(spec, model, out, cfg, history, t0, n, n_neg):
    model.jepa.eval(); model.jepa.cpu()
    model.save(str(out / "model.pt"))
    # official numbers come from the checkpoint that ships (BF16), not the FP32 training copy
    shipped = NerveEnergy.load(spec, str(out / "model.pt"))
    final = evaluate(spec, shipped.step, cfg.holdout, learned_only_fn=shipped.step_learned_only)
    final["train_pairs"] = int(n); final["negatives"] = int(n_neg); final["seconds"] = round(time.time() - t0, 1)
    final["params"] = sum(p.numel() for p in model.jepa.parameters())
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=1))
    (out / "history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in history))
    (out / "eval.json").write_text(json.dumps(final, indent=1))
    return final


def _train_continuous(spec, log, out, cfg, enc, model):
    if cfg.init:
        model.jepa.load_state_dict(NerveEnergy.load(spec, cfg.init).jepa.state_dict())
    S, A, AC, NX, NEG = _tensors_continuous(spec, enc, log, seed=cfg.seed)
    dev = _device(cfg)
    model.jepa.to(dev)
    S, A, AC, NX, NEG = (x.to(dev) for x in (S, A, AC, NX, NEG))
    opt = torch.optim.AdamW(model.jepa.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)
    n = len(S); history = []; t0 = time.time()
    for ep in range(cfg.epochs):
        model.jepa.train(); perm = torch.randperm(n, device=dev); tot = 0.0; nce = 0.0
        for i in range(0, n, cfg.batch):
            idx = perm[i:i + cfg.batch]
            if len(idx) < 2:
                continue
            losses = model.jepa.loss_continuous(S[idx], A[idx], AC[idx], NX[idx], NEG[idx])
            opt.zero_grad(); losses["total"].backward(); opt.step(); model.jepa.ema_update()
            tot += losses["total"].item() * len(idx); nce += losses["nce"].item() * len(idx)
        sched.step()
        rec = {"epoch": ep, "loss": tot / n, "nce": nce / n}
        if ep % cfg.log_every == 0 or ep == cfg.epochs - 1:
            model.jepa.eval(); model.jepa.cpu()
            ev = evaluate(spec, model.step, cfg.holdout, n_samples=200)
            model.jepa.to(dev)
            rec.update({"all": ev["all"]["exact"], "score": ev["score"]})
            print(f"  ep {ep:3d} loss {rec['loss']:.4f} nce {rec['nce']:.4f} | within-tol {rec['all']:.3f} "
                  f"violations {ev['all']['violation_rate']:.3f} score {rec['score']:.3f}")
        history.append(rec)
    return _finish(spec, model, out, cfg, history, t0, n, int(NEG.shape[0] * NEG.shape[1]))
