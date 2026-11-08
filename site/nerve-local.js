// The lab without a server: every API path the pages call is answered by the WebAssembly build of
// crates/nerve (`Room`), with the specs, machine documents and served networks fetched as static files.
// The pages wait on `window.nerveReady` and route `api()` through `window.nerveApi` when it exists.
//
// The manifest is always revalidated; everything else is fetched under a URL that carries the build it
// came from, so the module and the networks can be cached for good and a new deploy is seen at once.

const status = msg => { const el = document.getElementById("who"); if (el) el.textContent = msg; };

try {
  status("loading the networks…");
  const manifest = await fetch("./manifest.json", { cache: "no-cache" }).then(r => r.json());
  const build = encodeURIComponent((manifest.built && manifest.built.commit) || manifest.built || "dev");
  const versioned = url => `${url}${url.includes("?") ? "&" : "?"}v=${build}`;
  const text = async url => { const r = await fetch(versioned(url)); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.text(); };
  const bytes = async url => { const r = await fetch(versioned(url)); if (!r.ok) throw new Error(`${url}: ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };
  const { default: init, Room } = await import(versioned("./pkg/nerve.js"));
  await init({ module_or_path: versioned("./pkg/nerve_bg.wasm") });
  const room = new Room(1.0);
  for (const name of manifest.specs) room.add_spec(await text(`./specs/${name}.nerve`));
  for (const name of manifest.machines) room.add_machine(await text(`./machines/${name}.nerve`));
  // Every served network up front (about 500 KB each), so the listings carry each network's record.
  let done = 0;
  await Promise.all(manifest.models.map(async name => {
    const [meta, weights] = await Promise.all([text(`./models/${name}/model.json`), bytes(`./models/${name}/model.safetensors`)]);
    room.add_model(name, meta, weights);
    done += 1;
    status(`loading the networks… ${done}/${manifest.models.length}`);
  }));
  // The render networks (1–2 MB each) load when an interface is first drawn: their records now, so the
  // listings can describe them, their weights on the first request for their picture.
  const pictures = manifest.pictures || [];
  const records = new Map(await Promise.all(pictures.map(async name => {
    const meta = await text(`./models/${name}/picture.json`);
    return [name, { meta, record: JSON.parse(meta) }];
  })));
  const loading = new Map();
  const loadPicture = name => {
    if (!loading.has(name)) {
      loading.set(name, (async () => {
        room.add_picture(name, records.get(name).meta, await bytes(`./models/${name}/picture.safetensors`));
      })().catch(err => { loading.delete(name); throw err; }));
    }
    return loading.get(name);
  };
  /** What the engine's listing would say about a render network, from its record, before its weights are in. */
  const describe = ({ record: r }) => ({ kind: "picture", w: r.picture.w, h: r.picture.h, params: (r.eval || {}).params || 0, device: "cpu", trained: { eval: r.eval, train: r.train } });
  const call = (path, body) => JSON.parse(room.call(path, body === undefined ? "" : JSON.stringify(body)));
  window.nerveApi = async (path, body) => {
    const picture = /^\/api\/([^/]+)\/picture$/.exec(path);
    if (picture && records.has(picture[1])) await loadPicture(picture[1]);
    const out = call(path, body);
    if (path === "/api/specs") return out.map(s => (s.picture || !records.has(s.name) ? s : { ...s, picture: describe(records.get(s.name)) }));
    return out;
  };
  window.nerveRoom = room;
  window.nerveBuilt = manifest.built;
} catch (err) {
  status(`failed to load: ${err.message}`);
  throw err;
} finally {
  if (window.nerveResolve) window.nerveResolve();
}
