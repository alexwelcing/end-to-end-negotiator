"""Conditional neural field renderer: (state, pixel coordinate) -> colour.

    c      = cond(state_features)                      what to draw
    f(x,y) = Fourier features of the coordinate        where
    rgb    = sigmoid( FiLM-MLP(f ; c) )                 colour at that point

Because it is queried per coordinate it renders at any resolution with no resampling, and the multi-octave
Fourier features keep edges and glyphs sharp. The state enters only through FiLM modulation of each layer, so
the renderer shares the behaviour network's featurisation and generalises to states it never saw rendered
(a knob at an unseen position, a digit in an unseen context). Training samples random pixels, so memory does
not scale with image size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FieldConfig:
    state_dim: int
    cond: int = 128            # conditioning latent
    hidden: int = 256
    layers: int = 6
    octaves: int = 10          # Fourier octaves on (x, y): highest frequency 2^(octaves-1) cycles per half-image
    aspect: float = 1.0        # width / height, so a pixel is square in coordinate space


class NeuralField(nn.Module):
    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        self.cond = nn.Sequential(nn.Linear(cfg.state_dim, 256), nn.GELU(), nn.Linear(256, cfg.cond), nn.GELU())
        in_dim = 2 + 4 * cfg.octaves
        self.inp = nn.Linear(in_dim, cfg.hidden)
        self.layers = nn.ModuleList([nn.Linear(cfg.hidden, cfg.hidden) for _ in range(cfg.layers)])
        self.film = nn.ModuleList([nn.Linear(cfg.cond, 2 * cfg.hidden) for _ in range(cfg.layers)])
        self.out = nn.Linear(cfg.hidden, 3)
        self.register_buffer("freqs", (2.0 ** torch.arange(cfg.octaves)) * math.pi)

    def encode_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """xy [..., 2] in [-1, 1] (x scaled by aspect) -> Fourier features."""
        xy = xy * torch.tensor([self.cfg.aspect, 1.0], device=xy.device)
        ang = xy[..., None, :] * self.freqs[:, None]                          # [..., O, 2]
        return torch.cat([xy, torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2)], -1)

    def forward(self, state: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """state [B, state_dim], xy [B, P, 2] -> rgb [B, P, 3] in [0, 1]."""
        c = self.cond(state)                                                   # [B, cond]
        h = F.gelu(self.inp(self.encode_xy(xy)))                               # [B, P, hidden]
        for lin, film in zip(self.layers, self.film):
            g, b = film(c).chunk(2, -1)                                        # [B, hidden] each
            h = F.gelu(lin(h)) * (1 + g[:, None, :]) + b[:, None, :]
        return torch.sigmoid(self.out(h))

    @torch.no_grad()
    def render(self, state: torch.Tensor, height: int, width: int, chunk: int = 65536) -> torch.Tensor:
        """Full image [H, W, 3] for one state vector, at any resolution."""
        ys = torch.linspace(-1, 1, height, device=state.device)
        xs = torch.linspace(-1, 1, width, device=state.device)
        grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)   # [H*W, 2] (x, y)
        out = []
        for i in range(0, grid.shape[0], chunk):
            out.append(self.forward(state[None], grid[None, i:i + chunk])[0])
        return torch.cat(out).reshape(height, width, 3)

    def config_dict(self) -> dict:
        return asdict(self.cfg)


def grid_coords(height: int, width: int, device=None) -> torch.Tensor:
    ys = torch.linspace(-1, 1, height, device=device); xs = torch.linspace(-1, 1, width, device=device)
    return torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1)              # [H, W, 2]
