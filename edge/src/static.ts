// Browser build: the whole runtime (nn.ts) plus the play UI, no server. Weights are fetched as JSON next to the page.
import { Nerve, type EdgeModel } from "./nn";

declare global { interface Window { NerveStatic: typeof boot } }

async function boot(root: HTMLElement, specs: string[]) {
  const q = new URLSearchParams(location.search);
  const spec = q.get("spec") || specs[0];
  const el = (id: string) => root.querySelector<HTMLElement>("#" + id)!;
  root.innerHTML = `
  <h1 id="title"></h1><p class="sub">Every value below comes out of a neural network, not code: a click scores every candidate future and takes the lowest energy.
  <select id="spec"></select> <span id="who" class="pill"></span></p>
  <div class="grid"><div class="card"><h2>State</h2><div id="state" class="state"></div><p class="energy" id="energy"></p><div id="alts"></div>
  <div class="row" style="margin-top:10px"><button id="reset">reset</button></div></div>
  <div class="card"><h2>Actions</h2><div id="actions" class="actions"></div></div></div>
  <div class="card" style="margin-top:16px"><h2>Trace</h2><div id="log" class="log"></div></div>`;
  const sel = el("spec") as HTMLSelectElement;
  for (const s of specs) { const o = document.createElement("option"); o.value = s; o.textContent = s; if (s === spec) o.selected = true; sel.append(o); }
  sel.onchange = () => { location.search = "?spec=" + sel.value; };

  const m = (await (await fetch(`./${spec}.json`)).json()) as EdgeModel;
  const n = new Nerve(m);
  let state = m.initial as Record<string, unknown>, prev: Record<string, unknown> | null = null;
  el("title").textContent = spec;
  let params = 0; for (const t of Object.values(m.jepa.tensors)) params += t.data.length;
  el("who").textContent = `${m.states.length} states · ${m.actions.length} action instances · ${params} learned params · runs in this tab`;
  const fmt = (v: unknown) => typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : String(v);
  function render(E?: Float32Array, order?: number[]) {
    const st = el("state"); st.innerHTML = "";
    for (const k of Object.keys(m.fields)) { const b = document.createElement("b"); b.textContent = k; const s = document.createElement("span"); s.textContent = fmt(state[k]); s.className = "val" + (prev && prev[k] !== state[k] ? " changed" : ""); st.append(b, s); }
    el("energy").textContent = E && order ? `energy ${E[order[0]].toFixed(3)}` : "";
    el("alts").innerHTML = E && order ? order.slice(1, 3).map((i) => `<div class="alt">runner-up ${JSON.stringify(m.states[i])} · ${E[i].toFixed(2)}</div>`).join("") : "";
  }
  function act(name: string, p: Record<string, unknown>) {
    const E = n.energies(state, name, p); const order = [...E.keys()].sort((a, b) => E[a] - E[b]);
    prev = state; state = m.states[order[0]] as Record<string, unknown>; render(E, order);
    const d = document.createElement("div"); d.textContent = `${name}${Object.keys(p).length ? " " + JSON.stringify(p) : ""}  →  ${JSON.stringify(state)}  E=${E[order[0]].toFixed(2)}`; el("log").prepend(d);
  }
  const byName: Record<string, Record<string, unknown>[]> = {};
  for (const a of m.actions) (byName[a.name] = byName[a.name] || []).push(a.params);
  for (const [name, plist] of Object.entries(byName)) {
    const row = document.createElement("div"); row.className = "row"; const b = document.createElement("button"); b.textContent = name; row.append(b);
    const keys = Object.keys(plist[0]); const sels: Record<string, HTMLSelectElement> = {};
    for (const k of keys) { const s = document.createElement("select"); const seen = new Set<string>(); for (const p of plist) { const v = JSON.stringify(p[k]); if (seen.has(v)) continue; seen.add(v); const o = document.createElement("option"); o.value = v; o.textContent = `${k}=${p[k]}`; s.append(o); } sels[k] = s; row.append(s); }
    b.onclick = () => act(name, Object.fromEntries(keys.map((k) => [k, JSON.parse(sels[k].value)])));
    el("actions").append(row);
  }
  el("reset").onclick = () => { state = m.initial as Record<string, unknown>; prev = null; render(); el("log").innerHTML = ""; };
  render();
}
window.NerveStatic = boot;
