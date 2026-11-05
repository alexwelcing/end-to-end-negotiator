// The forward pass. This is the entire "program": energy over candidate futures, argmin.
export interface Linear { w: number[]; b: number[]; out: number; in: number }
export interface EdgeModel {
  spec: string;
  fields: Record<string, { kind: string; domain: unknown[] }>;
  initial: Record<string, unknown>;
  actions: { name: string; params: Record<string, unknown> }[];
  states: Record<string, unknown>[];
  state_vectors: number[][];
  jepa: { d: number; enc: Linear[]; tgt: Linear[]; pred: Linear[]; act: number[] };
  circuit: { ws: number[]; wa: number[]; wo: number[]; bias: number; H: number; weight: number } | null;
  state_dim: number;
  action_dim: number;
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
  constructor(m: EdgeModel) { this.m = m; }

  stateVector(s: Record<string, unknown>): Float32Array {
    const x = new Float32Array(this.m.state_dim);
    let off = 0;
    for (const [k, f] of Object.entries(this.m.fields)) {
      const j = f.domain.findIndex((v) => v === s[k]);
      if (j < 0) throw new Error(`state.${k}=${JSON.stringify(s[k])} not in type`);
      x[off + j] = 1;
      off += f.domain.length;
    }
    return x;
  }

  actionIndex(name: string, params: Record<string, unknown>): number {
    const i = this.m.actions.findIndex((a) => a.name === name && JSON.stringify(sortObj(a.params)) === JSON.stringify(sortObj(params)));
    if (i < 0) throw new Error(`unknown action ${name} ${JSON.stringify(params)}`);
    return i;
  }

  private targets(): Float32Array[] {
    if (!this.candZ) this.candZ = this.m.state_vectors.map((v) => mlp(this.m.jepa.tgt, Float32Array.from(v)));
    return this.candZ;
  }

  /** Energy of every candidate future for (state, action). */
  energies(s: Record<string, unknown>, action: string, params: Record<string, unknown> = {}): Float32Array {
    const xs = this.stateVector(s);
    const ai = this.actionIndex(action, params);
    const d = this.m.jepa.d;
    const z = mlp(this.m.jepa.enc, xs);
    const za = new Float32Array(2 * d);
    za.set(z, 0);
    za.set(this.m.jepa.act.slice(ai * d, (ai + 1) * d), d);
    const zh = mlp(this.m.jepa.pred, za);
    const tg = this.targets();
    const E = new Float32Array(tg.length);
    for (let c = 0; c < tg.length; c++) {
      let e = 0;
      for (let i = 0; i < d; i++) { const t = zh[i] - tg[c][i]; e += t * t; }
      E[c] = e;
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
