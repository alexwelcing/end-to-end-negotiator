// Check the Worker's forward pass against Python on every (state, action) pair.
import { readFileSync } from "node:fs";
import { Nerve } from "../src/nn.ts";   // run via esbuild bundle (fzstd import); see package.json "parity"
const [exportPath, pyPath] = process.argv.slice(2);
const n = new Nerve(JSON.parse(readFileSync(exportPath, "utf8")));
const py = JSON.parse(readFileSync(pyPath, "utf8")) as { s: any; a: string; p: any; next: any; E: number }[];
let ok = 0, maxdE = 0;
for (const c of py) {
  const r = n.step(c.s, c.a, c.p);
  if (JSON.stringify(r.state) === JSON.stringify(c.next)) ok++;
  maxdE = Math.max(maxdE, Math.abs(r.energy - c.E));
}
console.log(JSON.stringify({ pairs: py.length, same_successor: ok, max_energy_diff: maxdE }));
