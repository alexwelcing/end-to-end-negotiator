"""(state, image) pairs: render every state of a finite spec with its ground-truth renderer under Playwright.

Images are stored as uint8 arrays at a base resolution; the neural field is trained on random pixel coordinates,
so the same set serves any output resolution. A deterministic hash holds out a fraction of states entirely.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..spec import Spec, Oracle


class RenderSet:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.images = np.load(self.root / "images.npy")                 # [N, H, W, 3] uint8
        self.states: list[dict[str, Any]] = self.meta["states"]
        self.held = np.array(self.meta["held"], dtype=bool)

    @property
    def hw(self) -> tuple[int, int]:
        return self.images.shape[1], self.images.shape[2]

    def split(self):
        return np.flatnonzero(~self.held), np.flatnonzero(self.held)


def heldout_state(spec: Spec, s: dict[str, Any], frac: float, salt: str = "render") -> bool:
    h = int.from_bytes(hashlib.blake2b(f"{salt}|{spec.state_key(s)}".encode(), digest_size=4).digest(), "big") / 2**32
    return h < frac


def build(spec: Spec, render_html, out: str | Path, width: int = 560, height: int = 360, scale: int = 1,
          holdout: float = 0.2, only_valid: bool = True) -> dict[str, Any]:
    """Render every (valid) state. `render_html(state) -> html`. Writes images.npy + meta.json."""
    import subprocess, tempfile
    from PIL import Image
    oracle = Oracle(spec)
    states = [s for s in spec.state_space()
              if not only_valid or not any(v.startswith("invariant") for v in oracle.violations(s, None, s))]
    tmp = Path(tempfile.mkdtemp(dir=".", prefix=".shoot-"))
    for i, s in enumerate(states):
        (tmp / f"{i:05d}.html").write_text(render_html(s))
    shoot = Path(__file__).with_name("shoot.mjs")
    subprocess.run(["node", str(shoot), str(tmp), str(width), str(height), str(scale)], check=True, cwd=Path.cwd())
    imgs, held = [], []
    for i, s in enumerate(states):
        imgs.append(np.asarray(Image.open(tmp / f"{i:05d}.png").convert("RGB")))
        held.append(heldout_state(spec, s, holdout))
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", np.stack(imgs))
    (out / "meta.json").write_text(json.dumps({"spec": spec.name, "states": states, "held": held, "width": width,
                                               "height": height, "scale": scale}))
    return {"states": len(states), "held_out": int(sum(held)), "shape": list(np.stack(imgs).shape), "out": str(out)}
