"""The engine's shader (crates/nerve/src/shader.rs) as a torch module, for GPU training.

Architecture and parameter names match the Rust `Shader` exactly, so a checkpoint written by
`export_rust` loads with `Shader::load` and draws the same picture (gelu is the erf form, as
`gelu_erf`). Training data come from `nerve rasters` (the engine's own oracle at `scale` times the
scene's raster); the Python side never rasterises anything.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RustShaderConfig:
    w: int
    h: int
    state_dim: int
    cond: int = 128
    hidden: int = 128
    layers: int = 4
    octaves: int = 6                                   # top octave ~4 px of the 4x raster; 8 aliased at off-lattice sizes
    kind: str = "mlp"                                  # "mlp" | "grid"
    grid_levels: list = None                           # [[gw, gh], ...] in cells, coarse to fine
    grid_channels: int = 0

    def __post_init__(self):
        if self.grid_levels is None:
            self.grid_levels = []

    @property
    def aspect(self) -> float:
        return self.w / self.h

    @property
    def grid_dim(self) -> int:
        return len(self.grid_levels) * self.grid_channels if self.kind == "grid" else 0

    @classmethod
    def grid(cls, w: int, h: int, state_dim: int, bases=(16, 64), channels: int = 8, hidden: int = 64, layers: int = 2, octaves: int = 4, cond: int = 128):
        aspect = w / h
        return cls(w, h, state_dim, cond, hidden, layers, octaves, "grid", [[max(2, round(b * aspect)), max(2, b)] for b in bases], channels)


class RustShader(nn.Module):
    def __init__(self, cfg: RustShaderConfig):
        super().__init__()
        self.cfg = cfg
        self.cond = nn.ModuleDict({"0": nn.Linear(cfg.state_dim, 256), "2": nn.Linear(256, cfg.cond)})
        self.grids = nn.ParameterList([nn.Parameter(torch.empty(gw * gh, cfg.grid_channels).uniform_(-0.1, 0.1)) for gw, gh in cfg.grid_levels]) if cfg.kind == "grid" else None
        self.inp = nn.Linear(2 + 4 * cfg.octaves + cfg.grid_dim, cfg.hidden)
        self.layers = nn.ModuleList([nn.Linear(cfg.hidden, cfg.hidden) for _ in range(cfg.layers)])
        self.film = nn.ModuleList([nn.Linear(cfg.cond, 2 * cfg.hidden) for _ in range(cfg.layers)])
        self.out = nn.Linear(cfg.hidden, 3)
        self.register_buffer("freqs", (2.0 ** torch.arange(cfg.octaves)) * math.pi)

    def features(self, xy: torch.Tensor) -> torch.Tensor:
        xy = xy * torch.tensor([self.cfg.aspect, 1.0], device=xy.device, dtype=xy.dtype)
        ang = xy[..., None, :] * self.freqs[:, None]                     # [B, P, O, 2]
        b, p = xy.shape[0], xy.shape[1]
        return torch.cat([xy, torch.sin(ang).reshape(b, p, -1), torch.cos(ang).reshape(b, p, -1)], -1)

    def film_params(self, x: torch.Tensor) -> list[torch.Tensor]:
        c = F.gelu(self.cond["2"](F.gelu(self.cond["0"](x))))
        return [f(c) for f in self.film]

    def grid_features(self, xy: torch.Tensor) -> torch.Tensor:
        """Bilinear features of every level at xy [B, P, 2] -> [B, P, levels * channels]; cell i sits at i/(g-1)."""
        b, p = xy.shape[0], xy.shape[1]
        feats = []
        for g, (gw, gh) in zip(self.grids, self.cfg.grid_levels):
            u = ((xy[..., 0] + 1) * 0.5).clamp(0, 1) * (gw - 1)
            v = ((xy[..., 1] + 1) * 0.5).clamp(0, 1) * (gh - 1)
            x0 = u.floor().clamp(max=gw - 2); y0 = v.floor().clamp(max=gh - 2)
            fx, fy = u - x0, v - y0
            x0, y0 = x0.long(), y0.long()
            def corner(dx, dy):
                return g[(y0 + dy) * gw + x0 + dx]                                       # [B, P, C]
            f = (corner(0, 0) * ((1 - fx) * (1 - fy))[..., None] + corner(1, 0) * (fx * (1 - fy))[..., None]
                 + corner(0, 1) * ((1 - fx) * fy)[..., None] + corner(1, 1) * (fx * fy)[..., None])
            feats.append(f)
        return torch.cat(feats, -1)

    def forward(self, x: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """x [B, state_dim], xy [B, P, 2] -> [B, P, 3] unclamped."""
        gb = self.film_params(x)
        f = self.features(xy)
        if self.grids is not None:
            f = torch.cat([f, self.grid_features(xy)], -1)
        h = F.gelu(self.inp(f))
        nh = self.cfg.hidden
        for lin, p in zip(self.layers, gb):
            g, b = p[:, :nh], p[:, nh:]
            h = F.gelu(lin(h)) * (1 + g[:, None, :]) + b[:, None, :]
        return self.out(h)

    @staticmethod
    def grid(w: int, h: int, device=None) -> torch.Tensor:
        ys = (2 * (torch.arange(h, device=device) + 0.5) / h - 1)
        xs = (2 * (torch.arange(w, device=device) + 0.5) / w - 1)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], -1).reshape(-1, 2)                   # row-major, (x, y)

    @torch.no_grad()
    def draw(self, x: torch.Tensor, w: int, h: int, chunk: int = 65536) -> torch.Tensor:
        """[3, h, w] in 0..1 for one state vector x [state_dim]."""
        g = self.grid(w, h, x.device)
        out = []
        for i in range(0, g.shape[0], chunk):
            out.append(self.forward(x[None], g[None, i:i + chunk])[0])
        return torch.cat(out).clamp(0, 1).reshape(h, w, 3).permute(2, 0, 1)

    def export_rust(self, spec_name: str, out_dir: str | Path, train_cfg: dict[str, Any], report: dict[str, Any]) -> None:
        from safetensors.torch import save_file
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        sd = {k.replace("grids.", "grid."): v.detach().float().cpu().contiguous() for k, v in self.state_dict().items() if k != "freqs"}
        save_file(sd, str(out / "shader.safetensors"))
        meta = {"spec": spec_name, "kind": "shader", "shader": asdict(self.cfg), "train": train_cfg, "eval": report}
        (out / "shader.json").write_text(json.dumps(meta, indent=1))


class Rasters:
    """What `nerve rasters` wrote: states (as JSON), state vectors, and CHW float rasters."""

    def __init__(self, root: str | Path):
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text())
        n, w, h, d = self.meta["n"], self.meta["w"], self.meta["h"], self.meta["state_dim"]
        self.images = np.fromfile(root / "rasters.f32", dtype="<f4").reshape(n, 3, h, w)
        self.x = np.fromfile(root / "x.f32", dtype="<f4").reshape(n, d)        # the engine's own state encoding
        self.states = self.meta["states"]
        self.w, self.h = w, h

    def config(self, **kw) -> RustShaderConfig:
        if kw.pop("kind", "mlp") == "grid":
            return RustShaderConfig.grid(self.w, self.h, self.meta["state_dim"], **{k: v for k, v in kw.items() if k in ("bases", "channels", "hidden", "layers", "octaves", "cond")})
        return RustShaderConfig(self.w, self.h, self.meta["state_dim"], **kw)


@dataclass
class ShaderTrain:
    steps: int = 20000
    batch_states: int = 8
    pixels: int = 4096
    lr: float = 1e-3
    seed: int = 0
    log_every: int = 1000
    device: str = "auto"


def train_shader(spec_name: str, rs: Rasters, cfg: RustShaderConfig, t: ShaderTrain, out_dir: str | Path) -> dict[str, Any]:
    dev = torch.device(t.device if t.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(t.seed)
    model = RustShader(cfg).to(dev)
    X = torch.tensor(rs.x, dtype=torch.float32, device=dev)
    imgs = torch.tensor(rs.images, device=dev).permute(0, 2, 3, 1).reshape(len(rs.x), -1, 3)   # [n, h*w, 3]
    grid = RustShader.grid(rs.w, rs.h, dev)
    opt = torch.optim.Adam(model.parameters(), lr=t.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t.steps, eta_min=t.lr * 0.02)
    n, npx = len(rs.x), rs.w * rs.h
    b = min(t.batch_states, n)
    t0 = time.time(); last = float("nan")
    for step in range(t.steps):
        si = torch.randint(0, n, (b,), device=dev)
        pi = torch.randint(0, npx, (b, t.pixels), device=dev)
        # a random point inside each sampled pixel's footprint (the oracle is area-averaged, so the target holds
        # across the pixel); pixel centres alone taught the lattice, not the picture, and aliased at other sizes
        jitter = (torch.rand(b, t.pixels, 2, device=dev) - 0.5) * torch.tensor([2.0 / rs.w, 2.0 / rs.h], device=dev)
        pred = model(X[si], grid[pi] + jitter)
        y = imgs[si].gather(1, pi[..., None].expand(-1, -1, 3))
        diff = pred - y
        loss = (diff ** 2).mean() + 0.3 * diff.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        last = loss.item()
        if t.log_every and (step + 1) % t.log_every == 0:
            print(f"step {step + 1:>6}  loss {last:.5f}", flush=True)
    model.eval()
    maes = []
    for i in range(n):
        maes.append((model.draw(X[i], rs.w, rs.h) - torch.tensor(rs.images[i], device=dev)).abs().mean().item())
    report = {"n": n, "mae": float(np.mean(maes)), "worst_mae": float(np.max(maes)), "w": rs.w, "h": rs.h, "loss": last,
              "seconds": round(time.time() - t0, 1), "trained_on": str(dev), "params": sum(p.numel() for p in model.parameters()), "steps": t.steps}
    model.cpu().export_rust(spec_name, out_dir, asdict(t), report)
    return report
