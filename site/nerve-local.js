// The lab without a server: every API path the pages call is answered by the WebAssembly build of
// crates/nerve (`Room`), with the specs, machine documents and served networks fetched as static files.
// The pages wait on `window.nerveReady` and route `api()` through `window.nerveApi` when it exists.
import init, { Room } from "./pkg/nerve.js";

const status = msg => { const el = document.getElementById("who"); if (el) el.textContent = msg; };
const text = async url => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.text(); };
const bytes = async url => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };

try {
  status("loading the networks…");
  const [manifest] = await Promise.all([fetch("./manifest.json").then(r => r.json()), init()]);
  const room = new Room(1.0);
  for (const name of manifest.specs) room.add_spec(await text(`./specs/${name}.nerve`));
  for (const name of manifest.machines) room.add_machine(await text(`./machines/${name}.nerve`));
  // Every served network up front (about 400 KB each), so the listings carry each network's record.
  let done = 0;
  await Promise.all(manifest.models.map(async name => {
    const [meta, weights] = await Promise.all([text(`./models/${name}/model.json`), bytes(`./models/${name}/model.safetensors`)]);
    room.add_model(name, meta, weights);
    done += 1;
    status(`loading the networks… ${done}/${manifest.models.length}`);
  }));
  window.nerveApi = async (path, body) => JSON.parse(room.call(path, body === undefined ? "" : JSON.stringify(body)));
  window.nerveRoom = room;
  window.nerveBuilt = manifest.built;
} catch (err) {
  status(`failed to load: ${err.message}`);
  throw err;
} finally {
  if (window.nerveResolve) window.nerveResolve();
}
