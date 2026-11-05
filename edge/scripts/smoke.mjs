// Drive the Worker over HTTP (workerd or wrangler dev) and, if given Python's answers, check parity.
// usage: node scripts/smoke.mjs <base-url> <spec> [py_steps.json]
const [base, spec, pyPath] = process.argv.slice(2);
if (!base || !spec) { console.error("usage: node scripts/smoke.mjs <base-url> <spec> [py_steps.json]"); process.exit(1); }
const j = async (path, init) => { const r = await fetch(base + path, init); if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`); return r.json(); };
const post = (path, body) => j(path, { method: "POST", body: JSON.stringify(body), headers: { "content-type": "application/json" } });

// a session: state lives in the Durable Object, the Worker only computes
const sid = `smoke-${Date.now()}`;
await fetch(`${base}/${spec}/session/${sid}`, { method: "DELETE" });
const trace = [];
for (const a of [{ action: "inc" }, { action: "inc" }, { action: "set", params: { to: 7 } }, { action: "arm" }, { action: "inc" }]) {
  const r = await post(`/${spec}/session/${sid}`, a);
  trace.push({ ...a, state: r.state, energy: +r.energy.toFixed(3) });
}
const out = { base, spec, session: trace };

if (pyPath) {
  const { readFileSync } = await import("node:fs");
  const py = JSON.parse(readFileSync(pyPath, "utf8"));
  let same = 0, maxdE = 0;
  for (const c of py) {
    const r = await post(`/${spec}/step`, { state: c.s, action: c.a, params: c.p });
    if (JSON.stringify(r.state) === JSON.stringify(c.next)) same++;
    maxdE = Math.max(maxdE, Math.abs(r.energy - c.E));
  }
  out.parity = { pairs: py.length, same_successor: same, max_energy_diff: maxdE };
}
console.log(JSON.stringify(out));
