// PULSE sparse value patches, decoded in the Worker. Mirrors nerve/sync/pulse.py byte for byte:
//   "NRVP" | version u8 | header_len u32le | header JSON | zstd(body)
// body per sparse tensor: varint index deltas (first = absolute) then n BF16 values (int16 LE).
// Decode is a memory copy; the SHA-256 of the post-patch weights must equal header.digest or we roll back.
import { decompress } from "fzstd";

export interface Tensor { shape: number[]; data: Float32Array }
export interface PatchHeader {
  version: number; n_changed: number; n_total: number; digest: string; level: number;
  tensors: Record<string, { kind: "sparse"; shape: number[]; n: number; ilen: number; vlen: number } | { kind: "dense"; dtype: string; shape: number[]; len: number }>;
}

const bf16ToF32 = (() => {
  const buf = new DataView(new ArrayBuffer(4));
  return (bits: number) => { buf.setUint32(0, (bits & 0xffff) << 16, true); return buf.getFloat32(0, true); };
})();

const f32ToBf16 = (() => {
  const buf = new DataView(new ArrayBuffer(4));
  return (x: number) => { buf.setFloat32(0, x, true); return (buf.getUint32(0, true) >>> 16) & 0xffff; };
})();

export function parseHeader(data: Uint8Array): { header: PatchHeader; payload: Uint8Array } {
  if (String.fromCharCode(...data.subarray(0, 4)) !== "NRVP") throw new Error("not a nerve patch");
  const hlen = new DataView(data.buffer, data.byteOffset + 5, 4).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(data.subarray(9, 9 + hlen))) as PatchHeader;
  return { header, payload: data.subarray(9 + hlen) };
}

function unvarint(buf: Uint8Array, pos: number, n: number): { out: Float64Array; pos: number } {
  const out = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    let shift = 0, x = 0;
    for (;;) { const b = buf[pos++]; x += (b & 0x7f) * 2 ** shift; if (!(b & 0x80)) break; shift += 7; }
    out[k] = x;
  }
  return { out, pos };
}

/** SHA-256 over sorted names, each followed by its BF16 bits (little-endian). Same as checkpoint_digest in Python. */
export async function digest(tensors: Record<string, Tensor>): Promise<string> {
  const names = Object.keys(tensors).sort();
  const enc = new TextEncoder();
  let total = 0;
  const parts: Uint8Array[] = [];
  for (const k of names) {
    const kb = enc.encode(k); parts.push(kb); total += kb.length;
    const d = tensors[k].data; const vb = new Uint8Array(d.length * 2); const dv = new DataView(vb.buffer);
    for (let i = 0; i < d.length; i++) dv.setUint16(i * 2, f32ToBf16(d[i]), true);
    parts.push(vb); total += vb.length;
  }
  const all = new Uint8Array(total); let o = 0;
  for (const p of parts) { all.set(p, o); o += p.length; }
  const h = await crypto.subtle.digest("SHA-256", all);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Apply a patch in place. Throws (and restores the previous bits) if the digest does not match. */
export async function applyPatch(tensors: Record<string, Tensor>, data: Uint8Array): Promise<PatchHeader> {
  const { header, payload } = parseHeader(data);
  const body = decompress(payload);
  const backup: Record<string, Float32Array> = {};
  let pos = 0;
  try {
    for (const [k, m] of Object.entries(header.tensors)) {
      if (m.kind === "dense") throw new Error(`dense (non-float) tensor ${k} is not supported in the Worker`);
      const t = tensors[k];
      if (!t) throw new Error(`patch refers to unknown tensor ${k}`);
      backup[k] = Float32Array.from(t.data);
      const r = unvarint(body, pos, m.n); pos = r.pos;
      const vals = new DataView(body.buffer, body.byteOffset + pos, m.vlen); pos += m.vlen;
      let idx = 0;
      for (let i = 0; i < m.n; i++) { idx += r.out[i]; t.data[idx] = bf16ToF32(vals.getUint16(i * 2, true)); }
    }
    const got = await digest(tensors);
    if (got !== header.digest) throw new Error(`patch digest mismatch: ${got} != ${header.digest}`);
    return header;
  } catch (e) {
    for (const [k, b] of Object.entries(backup)) tensors[k].data.set(b);
    throw e;
  }
}
