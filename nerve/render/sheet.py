"""Contact sheet: ground truth vs the field's render, at native and 2x resolution, train and held-out states."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..spec import Spec
from ..spec.encode import Encoder
from .dataset import RenderSet
from .field import NeuralField


def _label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img); d.rectangle([0, 0, img.width, 18], fill=(20, 24, 36)); d.text((6, 3), text, fill=(232, 234, 240)); return img


def contact_sheet(spec: Spec, rs: RenderSet, model: NeuralField, out: str | Path, n_train: int = 3, n_held: int = 3,
                  upscale: int = 2, gt2x: dict[int, np.ndarray] | None = None) -> dict[str, Any]:
    enc = Encoder(spec); H, W = rs.hw
    tr, he = rs.split()
    picks = [(int(i), "train") for i in tr[:n_train]] + [(int(i), "HELD-OUT") for i in he[:n_held]]
    cols = 3 if gt2x is None else 4
    cell_w, cell_h = W, H
    sheet = Image.new("RGB", (cols * (cell_w + 8) + 8, len(picks) * (cell_h + 8) + 8), (11, 13, 18))
    rows = []
    for r, (i, kind) in enumerate(picks):
        s = torch.from_numpy(enc.state(rs.states[i]))
        gt = Image.fromarray(rs.images[i])
        r1 = Image.fromarray((model.render(s, H, W).numpy() * 255).round().astype(np.uint8))
        r2 = Image.fromarray((model.render(s, H * upscale, W * upscale).numpy() * 255).round().astype(np.uint8))
        psnr1 = 10 * np.log10(1 / max(((np.asarray(gt) / 255 - np.asarray(r1) / 255) ** 2).mean(), 1e-10))
        cells = [_label(gt.copy(), f"{kind} · ground truth · {rs.states[i]}"), _label(r1, f"network · {W}x{H} · PSNR {psnr1:.1f} dB"),
                 _label(r2.resize((W, H), Image.LANCZOS), f"network · {W*upscale}x{H*upscale} (shown downscaled)")]
        if gt2x is not None and i in gt2x:
            g2 = gt2x[i]; r2a = np.asarray(r2) / 255; psnr2 = 10 * np.log10(1 / max(((g2 / 255 - r2a) ** 2).mean(), 1e-10))
            cells.append(_label(Image.fromarray(g2).resize((W, H), Image.LANCZOS), f"ground truth {W*upscale}x{H*upscale} · PSNR vs network {psnr2:.1f} dB"))
        for c, im in enumerate(cells):
            sheet.paste(im, (8 + c * (cell_w + 8), 8 + r * (cell_h + 8)))
        rows.append({"state": rs.states[i], "kind": kind, "psnr": round(float(psnr1), 2)})
    Path(out).parent.mkdir(parents=True, exist_ok=True); sheet.save(out)
    return {"sheet": str(out), "rows": rows}


def crop_compare(spec: Spec, rs: RenderSet, model: NeuralField, idx: int, box: tuple[int, int, int, int], out: str | Path, upscale: int = 4) -> str:
    """Zoom into a box (x0, y0, x1, y1): ground truth (nearest-upscaled) vs the field rendered natively at `upscale`x."""
    enc = Encoder(spec); H, W = rs.hw
    s = torch.from_numpy(enc.state(rs.states[idx]))
    x0, y0, x1, y1 = box
    gt = Image.fromarray(rs.images[idx]).crop(box).resize(((x1 - x0) * upscale, (y1 - y0) * upscale), Image.NEAREST)
    full = model.render(s, H * upscale, W * upscale).numpy()
    hi = Image.fromarray((full[y0 * upscale:y1 * upscale, x0 * upscale:x1 * upscale] * 255).round().astype(np.uint8))
    sheet = Image.new("RGB", (gt.width * 2 + 24, gt.height + 16 + 20), (11, 13, 18))
    sheet.paste(_label(gt, f"screenshot pixels, {upscale}x nearest"), (8, 8)); sheet.paste(_label(hi, f"network rendered natively at {upscale}x"), (gt.width + 16, 8))
    Path(out).parent.mkdir(parents=True, exist_ok=True); sheet.save(out); return str(out)
