// The forward pass. This is the entire "program": energy over candidate futures, argmin.
import { applyPatch, digest, type Tensor } from "./pulse";

export interface Linear { w: Float32Array; b: Float32Array; out: number; in: number }
export interface EdgeModel {
  spec: string;
  fields: Record<string, { kind: string; domain: unknown[] }>;
  initial: Record<string, unknown>;
  actions: { name: string; params: Record<string, unknown> }[];
  states: Record<string, unknown>[];
  state_vectors: number[][];
  jepa: { d: number; tensors: Record<string, { shape: number[]; data: number[] }>; digest: string;
          fact?: { weight: number; own_scale: number; fields: { offset: number; n: number; ordinal: boolean }[] } | null };
  circuit: { ws: number[]; wa: number[]; wo: number[]; bias: number; H: number; weight: number } | null;
  state_dim: number;
  action_dim: number;
  layout?: { disc_dim: number; n_fourier: number; ordinal: { field: string; offset: number; values_norm: number[] }[] };
}

const gelu = (x: number) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));

function linear(l: Linear, x: Float32Array): Float32Array {
  const y = new Float32Array(l.out);
  for (let o = 0; o < l.out; o++) {
    let s = l.b[o];
    const row = o * l.in;
    for (let i = 0; i < l.in; i++) s += l.w[row + i] * x[i];
    y[o] = s;
  }
  return y;
}

function mlp(layers: Linear[], x: Float32Array): Float32Array {
  let h = x;
  for (let i = 0; i < layers.length; i++) {
    h = linear(layers[i], h);
    if (i < layers.length - 1) for (let j = 0; j < h.length; j++) h[j] = gelu(h[j]);
  }
  return h;
}

export class Nerve {
  private candZ: Float32Array[] | null = null;
  m: EdgeModel;
  tensors: Record<string, Tensor> = {};        // live weights, patched in place by PULSE
  enc: Linear[]; tgt: Linear[]; pred: Linear[]; act: Float32Array;
  applied: string[] = [];
  constructor(m: EdgeModel) {
    this.m = m;
    for (const [k, t] of Object.entries(m.jepa.tensors)) this.tensors[k] = { shape: t.shape, data: Float32Array.from(t.data) };
    const lin = (p: string): Linear => {
      const w = this.tensors[`${p}.weight`], b = this.tensors[`${p}.bias`];
      return { w: w.data, b: b.data, out: w.shape[0], in: w.shape[1] };
    };
    const mlp = (p: string) => [0, 2, 4].map((i) => lin(`${p}.${i}`));   // Sequential(Linear, GELU, Linear, GELU, Linear)
    this.enc = mlp("enc"); this.tgt = mlp("tgt"); this.pred = mlp("pred"); this.act = this.tensors["act.weight"].data;
    const F = m.jepa.fact;
    if (F) {
      this.factNet = F.fields.map((_, f) => [lin(`fact.net.${f}.0`), lin(`fact.net.${f}.2`)]);
      this.factOwn = F.fields.map((_, f) => lin(`fact.own.${f}`));
      this.factMasks = this.tensors["fact.masks"].data;
    }
  }
  factNet: Linear[][] = [];
  factOwn: Linear[] = [];
  factMasks: Float32Array = new Float32Array(0);

  /** Factored energy of every candidate: -sum_f log p_f(cand_f). Per field: an MLP over [action embedding ;
   *  state with the field's own value masked] gives absolute logits (and, for ordinal fields, shift logits
   *  indexed by offset from the current value); a small own-value term from the fused latent is added. */
  private factored(zh: Float32Array, xs: Float32Array, za: Float32Array): Float32Array {
    const F = this.m.jepa.fact!;
    const C = this.m.state_vectors.length;
    const E = new Float32Array(C);
    const D = this.m.state_dim;
    F.fields.forEach((fl, f) => {
      let i = 0;
      for (let k = 0; k < fl.n; k++) if (xs[fl.offset + k] > 0.5) { i = k; break; }
      const inp = new Float32Array(za.length + D);
      inp.set(za, 0);
      for (let k = 0; k < D; k++) inp[za.length + k] = xs[k] * this.factMasks[f * D + k];
      const h = mlp(this.factNet[f], inp);                       // [n + (2n-1 if ordinal)]
      const own = linear(this.factOwn[f], zh);
      const lg = new Float32Array(fl.n);
      for (let j = 0; j < fl.n; j++) lg[j] = h[j] + F.own_scale * own[j];
      if (fl.ordinal) for (let j = 0; j < fl.n; j++) lg[j] += h[fl.n + (j - i + fl.n - 1)];
      let mx = -Infinity; for (let j = 0; j < fl.n; j++) mx = Math.max(mx, lg[j]);
      let z = 0; for (let j = 0; j < fl.n; j++) z += Math.exp(lg[j] - mx);
      const lse = mx + Math.log(z);
      for (let c = 0; c < C; c++) {
        const v = this.m.state_vectors[c];
        let jc = 0; for (let k = 0; k < fl.n; k++) if (v[fl.offset + k] > 0.5) { jc = k; break; }
        E[c] -= lg[jc] - lse;
      }
    });
    return E;
  }

  /** Apply a PULSE patch (verified bit-identical by digest) and drop cached target encodings. */
  async patch(name: string, bytes: Uint8Array): Promise<void> {
    await applyPatch(this.tensors, bytes);
    this.applied.push(name);
    this.candZ = null;
  }

  digest(): Promise<string> { return digest(this.tensors); }

  stateVector(s: Record<string, unknown>): Float32Array {
    const x = new Float32Array(this.m.state_dim);
    let off = 0;
    const idx: Record<string, number> = {};
    for (const [k, f] of Object.entries(this.m.fields)) {
      const j = f.domain.findIndex((v) => v === s[k]);
      if (j < 0) throw new Error(`state.${k}=${JSON.stringify(s[k])} not in type`);
      x[off + j] = 1; idx[k] = j;
      off += f.domain.length;
    }
    // ordinal features (same order and layout as Encoder.cont_feats_np): u, sin(2πku), cos(2πku) for k=1..n
    const L = this.m.layout;
    if (L) for (const o of L.ordinal) {
      const u = o.values_norm[idx[o.field]];
      x[off++] = u;
      for (let k = 1; k <= L.n_fourier; k++) { x[off++] = Math.sin(2 * Math.PI * k * u); x[off++] = Math.cos(2 * Math.PI * k * u); }
    }
    return x;
  }

  actionIndex(name: string, params: Record<string, unknown>): number {
    const i = this.m.actions.findIndex((a) => a.name === name && JSON.stringify(sortObj(a.params)) === JSON.stringify(sortObj(params)));
    if (i < 0) throw new Error(`unknown action ${name} ${JSON.stringify(params)}`);
    return i;
  }

  private targets(): Float32Array[] {
    if (!this.candZ) this.candZ = this.m.state_vectors.map((v) => mlp(this.tgt, Float32Array.from(v)));
    return this.candZ;
  }

  /** Energy of every candidate future for (state, action). */
  energies(s: Record<string, unknown>, action: string, params: Record<string, unknown> = {}): Float32Array {
    const xs = this.stateVector(s);
    const ai = this.actionIndex(action, params);
    const d = this.m.jepa.d;
    const z = mlp(this.enc, xs);
    const za = new Float32Array(2 * d);
    za.set(z, 0);
    za.set(this.act.subarray(ai * d, (ai + 1) * d), d);
    const zh = mlp(this.pred, za);
    const tg = this.targets();
    const E = new Float32Array(tg.length);
    for (let c = 0; c < tg.length; c++) {
      let e = 0;
      for (let i = 0; i < d; i++) { const t = zh[i] - tg[c][i]; e += t * t; }
      E[c] = e;
    }
    if (this.m.jepa.fact) {
      const Ef = this.factored(zh, xs, za.subarray(d, 2 * d));
      for (let c = 0; c < E.length; c++) E[c] += this.m.jepa.fact.weight * Ef[c];
    }
    const C = this.m.circuit;
    if (C) {
      // h = relu(xs·ws + onehot(a)·wa - bias); y = h·wo; E += weight * (1 - <cand, y>/bias)
      const y = new Float32Array(this.m.state_dim);
      for (let h = 0; h < C.H; h++) {
        let act = -C.bias + C.wa[ai * C.H + h];
        for (let i = 0; i < this.m.state_dim; i++) if (xs[i]) act += C.ws[i * C.H + h];
        if (act > 0) for (let o = 0; o < this.m.state_dim; o++) y[o] += act * C.wo[h * this.m.state_dim + o];
      }
      for (let c = 0; c < E.length; c++) {
        let dot = 0;
        const v = this.m.state_vectors[c];
        for (let i = 0; i < v.length; i++) dot += v[i] * y[i];
        E[c] += C.weight * (1 - dot / C.bias);
      }
    }
    return E;
  }

  step(s: Record<string, unknown>, action: string, params: Record<string, unknown> = {}) {
    const E = this.energies(s, action, params);
    let best = 0;
    for (let i = 1; i < E.length; i++) if (E[i] < E[best]) best = i;
    return { state: this.m.states[best], energy: E[best] };
  }
}

function sortObj(o: Record<string, unknown>) {
  return Object.fromEntries(Object.entries(o ?? {}).sort(([a], [b]) => (a < b ? -1 : 1)));
}
