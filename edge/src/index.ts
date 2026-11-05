// Nerve at the edge: the network runs inside the Worker; session state lives in a Durable Object;
// weights and PULSE patches come from R2. There is no application code anywhere in this path —
// only a spec-derived network and the forward pass in nn.ts.
import { Nerve, type EdgeModel } from "./nn";

export interface Env {
  // R2 bucket in production (wrangler.toml); a workerd `disk` service locally (workerd.capnp). Same keys either way.
  WEIGHTS: R2Bucket | Fetcher;
  SESSIONS: DurableObjectNamespace;
  ORIGIN: string;
}

const models = new Map<string, Nerve>();

/** Read a weights object by key from whichever store this Worker was bound to. */
async function getWeights(env: Env, key: string): Promise<Response | null> {
  const w = env.WEIGHTS as any;
  if (typeof w.get === "function" && typeof w.fetch !== "function") {
    const obj = await (w as R2Bucket).get(key);
    return obj ? new Response(obj.body) : null;
  }
  const r = await (w as Fetcher).fetch(`http://weights/${key}`);
  return r.ok ? r : null;
}

async function load(env: Env, spec: string): Promise<Nerve> {
  const cached = models.get(spec);
  if (cached) return cached;
  const obj = await getWeights(env, `${spec}/edge.json`);
  if (!obj) throw new Error(`no weights for ${spec}; run: nerve export specs/${spec}.nerve --model runs/${spec}/v2/model.pt --out edge/export/${spec}/edge.json`);
  const n = new Nerve((await obj.json()) as EdgeModel);
  models.set(spec, n);
  return n;
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    const [, spec, verb, sid] = url.pathname.split("/");
    if (!spec) return json({ nerve: "edge", usage: "/<spec>/step  /<spec>/session/<id>  /<spec>/weights" });
    try {
      if (verb === "weights") {
        // Serve the current edge export and patch feed so browsers can run the network locally too.
        const key = url.searchParams.get("patch") ? `${spec}/patches/${url.searchParams.get("patch")}` : `${spec}/edge.json`;
        const obj = await getWeights(env, key);
        return obj ? new Response(obj.body, { headers: { "content-type": key.endsWith(".json") ? "application/json" : "application/octet-stream" } }) : json({ error: "not found" }, 404);
      }
      if (verb === "step" && req.method === "POST") {
        const body = (await req.json()) as { state: Record<string, unknown>; action: string; params?: Record<string, unknown> };
        if (env.ORIGIN) return fetch(`${env.ORIGIN}/step`, { method: "POST", body: JSON.stringify(body), headers: { "content-type": "application/json" } });
        const n = await load(env, spec);
        return json(n.step(body.state, body.action, body.params ?? {}));
      }
      if (verb === "session" && sid) {
        const id = env.SESSIONS.idFromName(`${spec}:${sid}`);
        return env.SESSIONS.get(id).fetch(new Request(`https://do/${spec}`, req));
      }
      if (verb === "reload") { models.delete(spec); return json({ reloaded: spec }); }
      return json({ error: "not found" }, 404);
    } catch (e) {
      return json({ error: String(e) }, 500);
    }
  },
};

/** One interface session: its state is the only mutable thing in the system. */
export class NerveSession {
  ctx: DurableObjectState; env: Env;
  constructor(ctx: DurableObjectState, env: Env) { this.ctx = ctx; this.env = env; }

  async fetch(req: Request): Promise<Response> {
    const spec = new URL(req.url).pathname.slice(1);
    const n = await load(this.env, spec);
    let state = (await this.ctx.storage.get<Record<string, unknown>>("state")) ?? n.m.initial;
    if (req.method === "POST") {
      const body = (await req.json()) as { action: string; params?: Record<string, unknown> };
      const out = n.step(state, body.action, body.params ?? {});
      state = out.state;
      await this.ctx.storage.put("state", state);
      return json({ state, energy: out.energy });
    }
    if (req.method === "DELETE") { await this.ctx.storage.delete("state"); return json({ state: n.m.initial }); }
    return json({ state });
  }
}

const json = (o: unknown, status = 200) => new Response(JSON.stringify(o), { status, headers: { "content-type": "application/json" } });
