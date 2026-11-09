"""Train the neural field on a RenderSet; evaluate PSNR on held-out states; render at higher resolution."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..spec import Spec
from ..spec.encode import Encoder
from .dataset import RenderSet
from .field import FieldConfig, NeuralField, grid_coords


@dataclass
class RenderTrainConfig:
    steps: int = 4000
    batch_states: int = 8
    pixels: int = 4096           # pixels per state per step
    lr: float = 1e-3
    cond: int = 128
    hidden: int = 256
    layers: int = 6
    octaves: int = 10
    seed: int = 0
    device: str = "auto"
    eval_every: int = 500


def _device(cfg):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def evaluate(model: NeuralField, S: torch.Tensor, imgs: torch.Tensor, idx: np.ndarray, H: int, W: int) -> dict[str, float]:
    model.eval()
    ps = []
    for i in idx:
        ps.append(psnr(model.render(S[i], H, W), imgs[i]))
    model.train()
    return {"psnr_mean": float(np.mean(ps)) if ps else float("nan"), "psnr_min": float(np.min(ps)) if ps else float("nan"), "n": int(len(idx))}


def train_render(spec: Spec, rs: RenderSet, out_dir: str | Path, cfg: RenderTrainConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RenderTrainConfig()
    torch.manual_seed(cfg.seed)
    dev = _device(cfg)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    enc = Encoder(spec)
    H, W = rs.hw
    S = torch.tensor(np.stack([enc.state(s) for s in rs.states]), dtype=torch.float32, device=dev)   # [N, state_dim]
    imgs = torch.tensor(rs.images, dtype=torch.float32, device=dev) / 255.0                            # [N, H, W, 3]
    train_idx, held_idx = rs.split()
    coords = grid_coords(H, W, dev).reshape(-1, 2)                                                       # [H*W, 2]
    model = NeuralField(FieldConfig(enc.state_dim, cfg.cond, cfg.hidden, cfg.layers, cfg.octaves, aspect=W / H)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr * 0.02)
    t0 = time.time(); hist = []
    tr = torch.tensor(train_idx, device=dev)
    for step in range(cfg.steps):
        bi = tr[torch.randint(0, len(tr), (cfg.batch_states,), device=dev)]
        pi = torch.randint(0, H * W, (cfg.batch_states, cfg.pixels), device=dev)
        xy = coords[pi]                                                                                   # [B, P, 2]
        target = imgs[bi].reshape(len(bi), H * W, 3).gather(1, pi[..., None].expand(-1, -1, 3))         # [B, P, 3]
        pred = model(S[bi], xy)
        loss = (pred - target).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            ev_tr = evaluate(model, S, imgs, train_idx[:8], H, W); ev_he = evaluate(model, S, imgs, held_idx, H, W)
            rec = {"step": step, "l1": loss.item(), "train_psnr": ev_tr["psnr_mean"], "heldout_psnr": ev_he["psnr_mean"], "heldout_min": ev_he["psnr_min"]}
            hist.append(rec); print(f"  step {step:5d} l1 {rec['l1']:.4f} | PSNR train {rec['train_psnr']:.1f} held-out {rec['heldout_psnr']:.1f} (min {rec['heldout_min']:.1f})", flush=True)
    model.eval().cpu()
    final = {"train": evaluate(model, S.cpu(), imgs.cpu(), train_idx, H, W), "heldout": evaluate(model, S.cpu(), imgs.cpu(), held_idx, H, W),
             "seconds": round(time.time() - t0, 1), "params": sum(p.numel() for p in model.parameters()), "hw": [H, W]}
    torch.save({"cfg": model.config_dict(), "state": model.state_dict(), "spec": spec.name}, out / "field.pt")
    (out / "history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hist))
    (out / "eval.json").write_text(json.dumps(final, indent=1)); (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return final


def load_field(path: str | Path) -> NeuralField:
    ck = torch.load(path, map_location="cpu")
    m = NeuralField(FieldConfig(**ck["cfg"])); m.load_state_dict(ck["state"]); m.eval()
    return m
