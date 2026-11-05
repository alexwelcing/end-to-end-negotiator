"""PULSE-style sparse value patching for trainer -> inference weight sync.

After Miahi & Belilovsky, "Understanding and Exploiting Weight Update
Sparsity for Communication-Efficient Distributed RL" (arXiv:2602.03839):

* compute-visibility gate  G(θ, s) = { i : cast_bf16(θ_i) != cast_bf16(θ_i - s_i) }
  — only updates that change the BF16 bits an inference worker can see are sent;
* a patch carries changed *indices* (delta-encoded varints) and their new
  *absolute* BF16 values, never arithmetic deltas, so chained patches
  reconstruct the trainer's weights bit-identically;
* the payload is compressed with zstd level 1 by default;
* decode is a plain memory copy, no floating point arithmetic;
* every patch carries a blake2b of the full post-patch checkpoint, so an
  inference worker can verify it is bit-identical to the trainer.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import zstandard as zstd

MAGIC = b"NRVP"            # nerve patch
VERSION = 1


def _varint(xs: Iterable[int]) -> bytes:
    out = bytearray()
    for x in xs:
        x = int(x)
        while True:
            b = x & 0x7F
            x >>= 7
            if x:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def _unvarint(buf: bytes, n: int) -> np.ndarray:
    out = np.empty(n, dtype=np.int64)
    i = 0
    for k in range(n):
        shift = 0
        x = 0
        while True:
            b = buf[i]; i += 1
            x |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        out[k] = x
    return out


def _bf16_bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy().reshape(-1)


def checkpoint_digest(sd: dict[str, torch.Tensor]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for k in sorted(sd):
        h.update(k.encode())
        v = sd[k]
        h.update(_bf16_bits(v).tobytes() if v.is_floating_point() else v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


@dataclass
class Patch:
    data: bytes
    n_changed: int
    n_total: int
    digest: str                    # of the post-patch checkpoint

    @property
    def sparsity(self) -> float:
        return 1.0 - self.n_changed / max(self.n_total, 1)

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(self.data)

    @classmethod
    def load(cls, path: str) -> "Patch":
        with open(path, "rb") as fh:
            data = fh.read()
        hdr = _read_header(data)
        return cls(data, hdr["n_changed"], hdr["n_total"], hdr["digest"])


def _read_header(data: bytes) -> dict:
    assert data[:4] == MAGIC, "not a nerve patch"
    (hlen,) = struct.unpack("<I", data[5:9])
    return json.loads(data[9:9 + hlen])


def sparsity(prev: dict[str, torch.Tensor], new: dict[str, torch.Tensor]) -> float:
    ch = tot = 0
    for k, v in new.items():
        if not v.is_floating_point():
            continue
        a, b = _bf16_bits(prev[k]), _bf16_bits(v)
        ch += int((a != b).sum()); tot += a.size
    return 1.0 - ch / max(tot, 1)


def encode_patch(prev: dict[str, torch.Tensor], new: dict[str, torch.Tensor], level: int = 1) -> Patch:
    """Sparse Value Patching (Algorithm 1): I = {i : W_t(i) != W_{t-1}(i)}, V = W_t[I]."""
    body = io.BytesIO()
    meta: dict[str, dict] = {}
    n_changed = n_total = 0
    for k in sorted(new):
        v = new[k]
        if not v.is_floating_point():
            if k not in prev or not torch.equal(prev[k], v):
                raw = v.detach().contiguous().numpy().tobytes()
                meta[k] = {"kind": "dense", "dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape), "len": len(raw)}
                body.write(raw)
            continue
        a = _bf16_bits(prev[k]) if k in prev else np.zeros(v.numel(), np.int16)
        b = _bf16_bits(v)
        n_total += b.size
        idx = np.flatnonzero(a != b)
        n_changed += idx.size
        if idx.size == 0:
            continue
        deltas = np.diff(idx, prepend=-1) - 0          # delta-encode indices (first delta = idx[0]+1 - 1)
        deltas[0] = idx[0]
        iv = _varint(deltas)
        vals = b[idx].tobytes()
        meta[k] = {"kind": "sparse", "shape": list(v.shape), "n": int(idx.size), "ilen": len(iv), "vlen": len(vals)}
        body.write(iv); body.write(vals)
    digest = checkpoint_digest(new)
    header = json.dumps({"version": VERSION, "n_changed": n_changed, "n_total": n_total, "digest": digest,
                         "level": level, "tensors": meta}, separators=(",", ":")).encode()
    payload = zstd.ZstdCompressor(level=level).compress(body.getvalue())
    data = MAGIC + bytes([VERSION]) + struct.pack("<I", len(header)) + header + payload
    return Patch(data, n_changed, n_total, digest)


def apply_patch(prev: dict[str, torch.Tensor], patch: Patch | bytes, verify: bool = True) -> dict[str, torch.Tensor]:
    """Decode: W_t <- W_{t-1}; W_t[I] <- V. Memory copy only; returns BF16 tensors."""
    data = patch.data if isinstance(patch, Patch) else patch
    hdr = _read_header(data)
    hlen = 9 + len(json.dumps(hdr, separators=(",", ":")).encode()) if False else 9 + struct.unpack("<I", data[5:9])[0]
    body = zstd.ZstdDecompressor().decompress(data[hlen:], max_output_size=1 << 31)
    out: dict[str, torch.Tensor] = {}
    pos = 0
    for k, v in prev.items():
        out[k] = v.detach().to(torch.bfloat16).clone() if v.is_floating_point() else v.detach().clone()
    for k, m in hdr["tensors"].items():
        if m["kind"] == "dense":
            raw = body[pos:pos + m["len"]]; pos += m["len"]
            out[k] = torch.from_numpy(np.frombuffer(raw, dtype=getattr(np, m["dtype"])).copy()).reshape(m["shape"])
            continue
        idx = np.cumsum(_unvarint(body[pos:pos + m["ilen"]], m["n"])); pos += m["ilen"]
        vals = np.frombuffer(body[pos:pos + m["vlen"]], dtype=np.int16); pos += m["vlen"]
        base = out[k].contiguous().view(torch.int16).numpy().reshape(-1).copy() if k in out else np.zeros(int(np.prod(m["shape"])), np.int16)
        base[idx] = vals
        out[k] = torch.from_numpy(base).view(torch.bfloat16).reshape(m["shape"])
    if verify and checkpoint_digest(out) != hdr["digest"]:
        raise ValueError("patch digest mismatch: reconstructed weights are not bit-identical to the trainer")
    return out
