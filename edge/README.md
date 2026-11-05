# nerve-edge

The trained network runs **inside a Cloudflare Worker**. `src/nn.ts` is the whole runtime: a forward pass and an argmin.

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
