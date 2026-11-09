// The shared machine room (docs/machine-room.md, M4): one Durable Object per room holds the authoritative
// machine state and ticks it with the same WebAssembly core the pages use, on an alarm at the room's rate.
// Viewers connect over WebSockets (hibernation API), receive every tick with its checks, and send the
// clock's controls, grabs and drops. A grab detaches a block for everyone: its deliveries queue in the room
// until the drop, which arrives as the external action the drag committed to. Everything else on this
// Worker is a static asset from `dist/`.
import init, { Room } from "./dist/pkg/nerve.js";
import wasmModule from "./dist/pkg/nerve_bg.wasm";

const ROOM_PATH = /^\/room\/([a-z0-9_]+)\/([A-Za-z0-9_-]{1,32})\/?$/;
const MAX_RATE = 30;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/room/_health") {
      return new Response("ok", { headers: { "content-type": "text/plain" } });
    }
    const m = url.pathname.match(ROOM_PATH);
    if (m) {
      const id = env.ROOMS.idFromName(`${m[1]}/${m[2]}`);
      return env.ROOMS.get(id).fetch(request);
    }
    return env.ASSETS.fetch(request);
  },
};

let wasmReady = null;

/** The core, loaded once per isolate, with every spec and machine and the networks of `machine`'s blocks. */
async function loadRoom(env, machine) {
  if (!wasmReady) wasmReady = init({ module_or_path: wasmModule });
  await wasmReady;
  const get = async (path) => {
    const r = await env.ASSETS.fetch(new Request(`https://assets.local/${path}`));
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r;
  };
  const manifest = await (await get("manifest.json")).json();
  const room = new Room(1.0);
  for (const name of manifest.specs) room.add_spec(await (await get(`specs/${name}.nerve`)).text());
  for (const name of manifest.machines) room.add_machine(await (await get(`machines/${name}.nerve`)).text());
  const listing = JSON.parse(room.call("/api/machines", "")).find((x) => x.name === machine);
  if (!listing) throw new Error(`no machine ${machine}`);
  const types = [...new Set(listing.blocks.map((b) => b.type))].filter((t) => manifest.models.includes(t));
  await Promise.all(types.map(async (t) => {
    const [meta, weights] = await Promise.all([get(`models/${t}/model.json`).then((r) => r.text()), get(`models/${t}/model.safetensors`).then((r) => r.arrayBuffer())]);
    room.add_model(t, meta, new Uint8Array(weights));
  }));
  // Listed again so each block's `network` flag reflects the loaded networks.
  const fresh = JSON.parse(room.call("/api/machines", "")).find((x) => x.name === machine);
  return { room, listing: fresh };
}

/** The clock's input: the first parameterless input of a `clock` block, else the first parameterless input. */
function driveInputs(listing) {
  const simple = listing.inputs.filter((i) => !Object.keys(i.params).length);
  const clocks = simple.filter((i) => listing.blocks.find((b) => b.instance === i.block).type === "clock");
  return (clocks.length ? clocks : simple.slice(0, 1)).map((i) => ({ block: i.block, action: i.action, params: i.params }));
}

export class MachineRoom {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
    this.core = null;
    this.snap = null;
    this.ctx.blockConcurrencyWhile(async () => {
      this.snap = (await this.ctx.storage.get("snap")) || null;
    });
  }

  fresh(machine) {
    return { machine, tick: 0, state: this.core.listing.initial, pending: [], running: false, rate: 6, holds: {}, broken: [], stats: { ticks: 0, agree: 0, viol: 0, ms: 0 }, stopped: null };
  }

  /** The core and the snapshot, after a cold start or a wake from hibernation. */
  async ensure(machine) {
    machine = machine || this.snap?.machine;
    if (!machine) throw new Error("no machine for this room");
    if (!this.core) this.core = await loadRoom(this.env, machine);
    if (!this.snap || this.snap.machine !== machine) {
      this.snap = this.fresh(machine);
      await this.save();
    }
    // Holds belong to live connections only.
    const live = new Set(this.ctx.getWebSockets().map((ws) => ws.deserializeAttachment()?.id));
    for (const id of Object.keys(this.snap.holds)) if (!live.has(id)) delete this.snap.holds[id];
  }

  async save() {
    await this.ctx.storage.put("snap", this.snap);
  }

  viewers() {
    return this.ctx.getWebSockets().length;
  }

  broadcast(msg) {
    const text = JSON.stringify(msg);
    for (const ws of this.ctx.getWebSockets()) {
      try { ws.send(text); } catch (_) { /* a closing socket */ }
    }
  }

  status() {
    return { type: "status", running: this.snap.running, rate: this.snap.rate, holds: this.snap.holds, broken: this.snap.broken || [], hands: this.snap.hands || 0, viewers: this.viewers(), tick: this.snap.tick, stats: this.snap.stats };
  }

  async fetch(request) {
    const url = new URL(request.url);
    const m = url.pathname.match(ROOM_PATH);
    if (!m) return new Response("not a room", { status: 404 });
    try {
      await this.ensure(m[1]);
    } catch (err) {
      return new Response(`the room could not load: ${err.message}`, { status: 500 });
    }
    if (request.headers.get("Upgrade") !== "websocket") {
      return Response.json({ room: m[2], machine: this.snap.machine, tick: this.snap.tick, running: this.snap.running, viewers: this.viewers(), state: this.snap.state });
    }
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    const id = crypto.randomUUID().slice(0, 8);
    this.ctx.acceptWebSocket(server);
    // The attachment survives hibernation: a woken instance learns the machine from it.
    server.serializeAttachment({ id, machine: m[1] });
    server.send(JSON.stringify({
      type: "welcome", id, room: m[2], machine: this.core.listing, state: this.snap.state, pending: this.snap.pending,
      tick: this.snap.tick, running: this.snap.running, rate: this.snap.rate, holds: this.snap.holds, stats: this.snap.stats,
      stopped: this.snap.stopped, viewers: this.viewers(),
    }));
    this.broadcast({ type: "viewers", count: this.viewers() });
    return new Response(null, { status: 101, webSocket: client });
  }

  /** One tick by the networks: the held blocks stay put, their deliveries queue, and everyone sees the result. */
  async tick(external = driveInputs(this.core.listing)) {
    const detached = [...new Set(Object.values(this.snap.holds))];
    const t0 = Date.now();
    const r = JSON.parse(this.core.room.call(`/api/machine/${this.snap.machine}/tick`, JSON.stringify({ state: this.snap.state, external, pending: this.snap.pending, detached, broken: this.snap.broken || [] })));
    r.ms = Date.now() - t0;
    const s = this.snap;
    s.tick += 1;
    s.state = r.state;
    s.pending = r.pending;
    s.stats.ticks += 1;
    if (r.agrees) s.stats.agree += 1;
    s.stats.viol += r.violations.length;
    s.stats.ms = s.stats.ms ? 0.9 * s.stats.ms + 0.1 * r.ms : r.ms;
    if (!r.agrees || r.violations.length) {
      // The visible signature of the design: the machine stops at the first failed check, for everyone.
      s.running = false;
      s.stopped = { tick: s.tick, steps: r.steps, violations: r.violations, agrees: r.agrees };
    }
    await this.save();
    this.broadcast({ type: "tick", tick: s.tick, running: s.running, holds: s.holds, broken: s.broken || [], hands: s.hands || 0, stats: s.stats, viewers: this.viewers(), ...r });
  }

  async alarm() {
    if (!this.snap) return;
    try {
      await this.ensure(this.snap.machine);
    } catch (_) {
      return;
    }
    if (!this.snap.running) return;
    if (this.viewers() === 0) {
      // Nobody is watching: the clock stops rather than ticking an empty room forever.
      this.snap.running = false;
      await this.save();
      return;
    }
    await this.tick();
    if (this.snap.running) await this.ctx.storage.setAlarm(Date.now() + 1000 / this.snap.rate);
  }

  async webSocketMessage(ws, raw) {
    let msg;
    try { msg = JSON.parse(raw); } catch (_) { return; }
    const { id, machine } = ws.deserializeAttachment() || {};
    try {
      await this.ensure(machine);
    } catch (err) {
      ws.send(JSON.stringify({ type: "error", message: `the room could not load: ${err.message}` }));
      return;
    }
    const s = this.snap;
    // The room's scoreboard counts every hand on the machine: grabs, keys and the input buttons.
    if (["grab", "key", "external"].includes(msg.type)) s.hands = (s.hands || 0) + 1;
    switch (msg.type) {
      case "run":
        s.running = true;
        s.stopped = null;
        await this.save();
        this.broadcast(this.status());
        await this.ctx.storage.setAlarm(Date.now());
        break;
      case "pause":
        s.running = false;
        await this.save();
        this.broadcast(this.status());
        break;
      case "step":
        await this.tick();
        break;
      case "external": {
        // One of the machine's declared inputs, applied by hand (dropping the ball); parameters are not taken.
        const block = String(msg.block), action = String(msg.action);
        const allowed = this.core.listing.inputs.some((i) => i.block === block && i.action === action && !Object.keys(i.params).length);
        if (!allowed) {
          ws.send(JSON.stringify({ type: "error", message: `${block}.${action} is not an input of this machine` }));
          break;
        }
        await this.tick([{ block, action, params: {} }]);
        break;
      }
      case "break": {
        // A visitor breaks a random networked block: its network runs as a corrupted copy from now on.
        const can = this.core.listing.blocks.filter((b) => b.network && !(s.broken || []).includes(b.instance)).map((b) => b.instance);
        if (can.length) {
          s.broken = [...(s.broken || []), can[Math.floor(Math.random() * can.length)]];
          await this.ctx.storage.put("snap", s);
        }
        this.broadcast(this.status());
        break;
      }
      case "heal":
        s.broken = [];
        s.stopped = null;
        await this.ctx.storage.put("snap", s);
        this.broadcast(this.status());
        break;
      case "key": {
        // A key as the machine document binds it, resolved by the core over the room's state right now.
        const r = JSON.parse(this.core.room.call(`/api/machine/${s.machine}/binding`, JSON.stringify({ state: s.state, key: String(msg.key) })));
        if (!r) break;
        if (r.verb) {
          await this.webSocketMessage(ws, JSON.stringify({ type: r.verb }));
          break;
        }
        await this.tick([{ block: r.block, action: r.action, params: r.params }]);
        break;
      }
      case "rate":
        s.rate = Math.min(MAX_RATE, Math.max(1, Number(msg.rate) || 6));
        await this.save();
        this.broadcast(this.status());
        break;
      case "reset": {
        const holds = s.holds;
        this.snap = this.fresh(s.machine);
        this.snap.holds = holds;
        await this.save();
        this.broadcast({ type: "reset", state: this.snap.state, ...this.status() });
        break;
      }
      case "grab": {
        const block = String(msg.block);
        const holder = Object.entries(s.holds).find(([, b]) => b === block)?.[0];
        if (holder && holder !== id) {
          ws.send(JSON.stringify({ type: "error", message: `${block} is held by someone else` }));
          break;
        }
        s.holds[id] = block;
        await this.save();
        this.broadcast(this.status());
        break;
      }
      case "drop": {
        const block = String(msg.block);
        if (s.holds[id] !== block) break;
        delete s.holds[id];
        // The drop is an external action to the composed spec; the deliveries held meanwhile land first.
        const external = msg.action ? [{ block, action: String(msg.action), params: msg.params || {} }] : [];
        if (external.length || s.pending.length) {
          try {
            await this.tick(external);
          } catch (err) {
            await this.save();
            ws.send(JSON.stringify({ type: "error", message: `drop refused: ${err.message}` }));
            this.broadcast(this.status());
          }
        } else {
          await this.save();
          this.broadcast(this.status());
        }
        break;
      }
      default:
        ws.send(JSON.stringify({ type: "error", message: `unknown message ${msg.type}` }));
    }
  }

  async webSocketClose(ws) {
    await this.release(ws);
  }

  async webSocketError(ws) {
    await this.release(ws);
  }

  /** A viewer left: whatever it held is let go (its queued deliveries land on the next tick). */
  async release(ws) {
    const { id, machine } = ws.deserializeAttachment() || {};
    if (!this.snap && machine) {
      try { await this.ensure(machine); } catch (_) { return; }
    }
    if (this.snap && id && this.snap.holds[id]) {
      delete this.snap.holds[id];
      await this.save();
    }
    if (this.snap) this.broadcast(this.status());
  }
}
