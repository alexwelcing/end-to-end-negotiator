// Smoke-test and parity-check the WebAssembly build against the native lab, in Node, without a browser.
//
//   node scripts/wasm_smoke.mjs [--dist site/dist] [--native http://127.0.0.1:8787]
//
// Loads site/dist as the page would (the module, every spec, machine and served network), then asks the
// wasm `Room` the same questions scripts/ci_smoke.py asks the server. With --native, every answer about
// state is compared with the running server's: the step of every finite interface from its initial
// state for every action instance, a drag of every gesture of the reorder list, and 100 ticks of the chain.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const arg = (name, dflt) => { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : dflt; };
const DIST = path.resolve(arg("--dist", "site/dist"));
const NATIVE = arg("--native", null);
const TICKS = 100;

const read = rel => fs.readFileSync(path.join(DIST, rel));
const text = rel => read(rel).toString("utf8");

const { default: init, Room } = await import(pathToFileURL(path.join(DIST, "pkg", "nerve.js")).href);
await init({ module_or_path: read("pkg/nerve_bg.wasm") });
const manifest = JSON.parse(text("manifest.json"));
const room = new Room(1.0);
for (const name of manifest.specs) room.add_spec(text(`specs/${name}.nerve`));
for (const name of manifest.machines) room.add_machine(text(`machines/${name}.nerve`));
for (const name of manifest.models) room.add_model(name, text(`models/${name}/model.json`), new Uint8Array(read(`models/${name}/model.safetensors`)));
for (const name of manifest.pictures || []) room.add_picture(name, text(`models/${name}/picture.json`), new Uint8Array(read(`models/${name}/picture.safetensors`)));
const wasm = (p, body) => JSON.parse(room.call(p, body === undefined ? "" : JSON.stringify(body)));

async function native(p, body) {
  const r = await fetch(NATIVE + p, body === undefined ? {} : { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json();
  if (j && j.error) throw new Error(`${p}: ${j.error}`);
  return j;
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
let compared = 0, failures = 0;
async function check(label, p, body, pick) {
  const w = wasm(p, body);
  if (!NATIVE) return w;
  const n = await native(p, body);
  compared += 1;
  if (!same(pick(w), pick(n))) {
    failures += 1;
    console.log(`DIFF ${label}\n  wasm:   ${JSON.stringify(pick(w))}\n  native: ${JSON.stringify(pick(n))}`);
  }
  return w;
}

// The listings.
const specs = wasm("/api/specs");
const withNet = specs.filter(s => s.model).map(s => s.name);
console.log(`${specs.length} interfaces, ${withNet.length} with a network`);
if (withNet.length !== manifest.models.length) throw new Error("not every served network loaded");
const machines = wasm("/api/machines");
const chain = machines.find(m => m.name === "chain");
if (!chain || !chain.blocks.every(b => b.network)) throw new Error("the chain or one of its networks is missing");

// Every finite interface: one step per action instance from the initial state (parameters from the domains).
let steps = 0;
for (const s of specs.filter(s => s.finite)) {
  for (const a of s.actions) {
    const combos = a.params.reduce((acc, p) => acc.flatMap(c => (p.domain || []).map(v => ({ ...c, [p.name]: v }))), [{}]).slice(0, 12);
    for (const params of combos) {
      const r = await check(`${s.name}.${a.name}(${JSON.stringify(params)})`, `/api/${s.name}/step`, { state: s.initial, action: a.name, params },
        x => ({ oracle: x.oracle, network: x.network, verdict: x.verdict, agrees: x.agrees }));
      if (r.violations && r.violations.length) throw new Error(`${s.name}.${a.name}: ${r.violations.join("; ")}`);
      steps += 1;
    }
  }
}
console.log(`${steps} steps answered${NATIVE ? " and compared" : ""}`);

// Every render network draws its interface's initial state close to the oracle's drawing.
const PIXEL_ERROR = 0.05;
let drawn = 0;
for (const s of specs.filter(s => s.picture)) {
  const p = wasm(`/api/${s.name}/picture`, { state: s.initial });
  if (!p.network || !(p.mae < PIXEL_ERROR)) throw new Error(`${s.name}: picture ${p.network ? `error ${p.mae}` : "not drawn by the network"}`);
  drawn += 1;
}
if (drawn !== (manifest.pictures || []).length) throw new Error("not every render network loaded");
console.log(`${drawn} pictures drawn by their networks within ${PIXEL_ERROR} of the oracle`);

// Every gesture of the reorder list: open, one frame at a landing spot, release.
const reorder = specs.find(s => s.name === "reorder");
for (const g of reorder.futures) {
  const d = wasm(`/api/reorder/drag`, { gesture: g.name, state: reorder.initial, preview: true });
  const target = d.candidates[d.candidates.length - 1];
  const open = wasm(`/api/reorder/drag`, { gesture: g.name, state: reorder.initial });
  const fr = wasm(`/api/drag/${open.session}/frame`, { x: target.x, y: target.y });
  // Sessions are per process, so the release is not routed through `check`; the native drag is replayed below.
  const rel = wasm(`/api/drag/${open.session}/release`, { x: target.x, y: target.y });
  if (NATIVE) {
    // The native server has its own session ids: open and drive the same drag there, compare the drop.
    const nopen = await native(`/api/reorder/drag`, { gesture: g.name, state: reorder.initial });
    await native(`/api/drag/${nopen.session}/frame`, { x: target.x, y: target.y });
    const nrel = await native(`/api/drag/${nopen.session}/release`, { x: target.x, y: target.y });
    compared += 1;
    if (!same(nrel.state, rel.state)) { failures += 1; console.log(`DIFF drag ${g.name}: wasm ${JSON.stringify(rel.state)} native ${JSON.stringify(nrel.state)}`); }
  }
  if (!same(fr.branch, target.value) || rel.violations.length) throw new Error(`reorder ${g.name}: frame ${JSON.stringify(fr.branch)} release ${JSON.stringify(rel)}`);
}
console.log(`${reorder.futures.length} drags released${NATIVE ? " and compared" : ""}`);

// The chain, by its networks, with the composed oracle agreeing at every tick; native and wasm in lockstep.
let state = chain.initial;
const advance = [{ block: "clock", action: "advance" }];
for (let i = 0; i < TICKS; i++) {
  const t = await check(`chain tick ${i + 1}`, `/api/machine/chain/tick`, { state, external: advance }, x => ({ state: x.state, agrees: x.agrees, violations: x.violations, steps: x.steps.map(s => s.block + "." + s.action) }));
  if (!t.agrees || t.violations.length) throw new Error(`tick ${i + 1}: ${JSON.stringify(t)}`);
  state = t.state;
}
console.log(`${TICKS} chain ticks agree with the composed oracle${NATIVE ? " and with the native server" : ""}`);

// The ball machine, when its rail network is served: drop the ball, then tick at frame rate, timing the
// continuous blocks' step heads. Continuous states are compared within the machine's tolerance (1% of the
// rail's range), since the wasm and native builds round their float arithmetic differently.
const ball = machines.find(m => m.name === "ball");
if (ball && ball.blocks.every(b => b.network)) {
  const TOL = 2.48;
  const near = (a, b) => typeof a === "number" && typeof b === "number" ? Math.abs(a - b) <= TOL
    : Array.isArray(a) && Array.isArray(b) ? a.length === b.length && a.every((x, i) => near(x, b[i]))
    : a && b && typeof a === "object" && typeof b === "object" ? Object.keys(a).length === Object.keys(b).length && Object.keys(a).every(k => near(a[k], b[k]))
    : same(a, b);
  let bs = wasm("/api/machine/ball/tick", { state: ball.initial, external: [{ block: "r1", action: "enter" }] }).state;
  let ns = NATIVE ? (await native("/api/machine/ball/tick", { state: ball.initial, external: [{ block: "r1", action: "enter" }] })).state : null;
  let ms = 0, agreed = 0;
  for (let i = 0; i < TICKS; i++) {
    const t0 = performance.now();
    const t = wasm("/api/machine/ball/tick", { state: bs, external: advance });
    ms += performance.now() - t0;
    if (t.violations.length) throw new Error(`ball tick ${i + 1}: ${JSON.stringify(t.violations)}`);
    agreed += t.agrees ? 1 : 0;
    bs = t.state;
    if (NATIVE) {
      const n = await native("/api/machine/ball/tick", { state: ns, external: advance });
      ns = n.state;
      compared += 1;
      if (!near(bs, ns)) { failures += 1; console.log(`DIFF ball tick ${i + 1}\n  wasm:   ${JSON.stringify(bs)}\n  native: ${JSON.stringify(ns)}`); }
    }
  }
  if (agreed < TICKS) throw new Error(`ball: only ${agreed}/${TICKS} ticks within tolerance of the composed oracle`);
  console.log(`${TICKS} ball ticks within tolerance of the composed oracle, ${(ms / TICKS).toFixed(2)} ms per tick in wasm (three rails stepped by their head)`);
}

if (NATIVE) console.log(`parity: ${compared} answers compared, ${failures} differ`);
if (failures) process.exit(1);
console.log("wasm smoke ok");
