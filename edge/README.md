# nerve-edge

The trained network runs **inside a Cloudflare Worker**. `src/nn.ts` is the whole runtime: a forward pass and an argmin.

## Local, on workerd

[workerd](https://github.com/cloudflare/workerd) is the open-source runtime Cloudflare Workers run on. The same bundle
runs here with no account:

```
npm install
python -m nerve edge ../specs/counter.nerve --model ../runs/counter/v2/model.pt --patches ../runs/counter/patches
#   -> exports to export/counter/{edge.json,patches/}, bundles with esbuild, serves on http://127.0.0.1:8788
curl -X POST http://127.0.0.1:8788/counter/session/alice -d '{"action":"inc"}'
npm run smoke                      # drives a session and prints the trace
```

`workerd.capnp` binds `WEIGHTS` to a read-only disk service over `export/` (R2 in production) and `SESSIONS` to a
Durable Object namespace persisted in `.do-storage/`, so sessions survive restarts. `src/index.ts` reads weights
through one function that accepts either binding. `tests/test_workerd.py` trains tiny models, starts workerd on a
spare port with its own export root, and asserts every transition matches Python over HTTP.

## Deploy to Cloudflare

```
nerve export specs/counter.nerve --model runs/counter/model.pt --out edge/export/counter.json
cd edge && npm i && npm run upload -- counter export/counter.json ../runs/counter/patches
npm run deploy
curl -X POST https://nerve-edge.<you>.workers.dev/counter/session/alice -d '{"action":"inc"}'
```

* `R2 nerve-weights/<spec>/edge.json` — compiled circuit + JEPA weights (from `nerve export`)
* `R2 nerve-weights/<spec>/patches/*.nrvp` — PULSE patches (from `nerve patch`); served at `/<spec>/weights?patch=...` for clients that apply them locally
* `Durable Object NerveSession` — one per interface session; holds only the state dict
* `ORIGIN` — optional: a Cloudflare Container or `nerve serve` for networks too large for the Worker CPU budget

Browsers can also fetch `/<spec>/weights` and run `nn.ts` locally (WebGPU is a drop-in for the inner loops), which is where `qwenGPU`'s "compiled layers in front of a model" idea points: ship the deterministic circuit to the client, keep training in the loop.
