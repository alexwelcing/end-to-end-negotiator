# Quickstart: play with it, train one, write one

Everything here is CPU-only and needs one tool: a Rust toolchain (https://rustup.rs). Python is optional (the
reference implementation and the experiment scripts); a GPU is optional (training is faster, not different).

## 1. Run the lab (about three minutes the first time, for the build)

```
git clone https://github.com/alexwelcing/nerve.git nerve
cd nerve
cargo run --release -p nerve -- serve
```

Open http://127.0.0.1:8787. Every tab is an interface with no code behind it: a spec in `specs/` says what is
allowed, and the trained network in `models/served/` runs it. Click an action and the network computes the next
state while the spec checks its answer; drag a handle and the states the spec allows compete to land under your
pointer. Each tab says what it demonstrates and what to try.

Or, with Docker and no Rust: `docker build -t nerve-lab . && docker run --rm -p 7860:7860 nerve-lab`, then
http://localhost:7860.

## 2. Train a network yourself

```
cargo run --release -p nerve -- train specs/hanoi.nerve --holdout 0 --epochs 300 --out runs/mine
```

Trains hanoi on every (state, action) cell of its spec and prints exactness and violations at the end (hanoi
takes a few minutes on a laptop CPU; `kanban` and `scheduler` take an hour or more). Then serve your model:

```
cargo run --release -p nerve -- serve --models runs/mine
```

On a CUDA GPU build with `cargo build --release -p nerve --features cuda` (Windows: `scripts\cargo-cuda.cmd
build --release -p nerve`, which sets up MSVC and CUDA 12.2) and add `--cuda` to `train`.

`nerve eval specs/hanoi.nerve runs/mine/hanoi` rescores a model; `--all-states` scores the network without the
spec's invariants masking its answers, `--json` prints one line for a log.

## 3. Write a spec

Copy `specs/toggle.nerve` (12 lines) or read `specs/counter.nerve`, then:

```
cargo run --release -p nerve -- check specs/yours.nerve     # types, invariants, ensures, futures, holes
cargo run --release -p nerve -- train specs/yours.nerve --holdout 0
```

`check` runs the spec against itself over its whole state space before anything is trained; a spec that breaks
its own rules never reaches training. The language is documented by example in `specs/` and in the paper
(`docs/paper/`); `render` and `futures` blocks give a spec a drag in the lab (`docs/dragology.md`).

## 4. Reproduce the numbers

`experiments/README.md` describes the pipeline; `experiments/round2/` holds the scripts, the registered
predictions and the append-only results log behind paper 2's tables. `python docs/paper2/make_tables.py`
regenerates the paper's numbers from that log; `make` in `docs/paper2` builds the PDF (latexmk or tectonic).

## The lab in the browser (no server)

The same two pages run on a WebAssembly build of the engine. Once:

```
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.114
```

then `cargo run --release -p nerve -- site` builds the module and assembles `site/dist`: the lab, the machine
room, the specs, the machines, the served networks and the render networks. Serve it with any static server,
for example `python -m http.server 8791 --directory site/dist`, and open `index.html` or `machine.html`.
`nerve smoke http://127.0.0.1:8787` asks a running server what CI needs to know (networks answer, the chain ticks
in agreement, pictures draw); `node scripts/wasm_smoke.mjs --dist site/dist --native http://127.0.0.1:8787` asks
the browser build the same and compares the answers. CI does all of it on every push.
The site is live at [https://nerve-machine-room.aw-ab5.workers.dev](https://nerve-machine-room.aw-ab5.workers.dev) (the machine room at `/machine`), served by a Cloudflare Worker that
holds `site/dist` as static assets (`site/wrangler.toml`). By hand, after `wrangler login`:
`wrangler deploy --config site/wrangler.toml`. `.github/workflows/pages.yml` does the same on every push to
`nerve` once the repository variable `CF_DEPLOY` and the secrets `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID` are set.

## Layout

| path | what |
|---|---|
| `specs/*.nerve` | the 20 interfaces, as specs (18 draggable ones, plus the digit and clock blocks machines are built from); `nerve gen` writes the ten generated ones from `crates/nerve/src/gen/` |
| `machines/*.nerve` | machines: blocks wired by events (odometer, odometer6, chain, ball, ball_drift); the lab serves them at `/machine` |
| `models/served/` | the trained network for each, chosen from the results log by `nerve serve-set` |
| `models/pictures/` | the render network for each (`nerve train-picture`): what the interface looks like, as weights |
| `crates/nerve/` | the Rust implementation: spec, oracle, encoder, energy model, drags, trainer, lab |
| `crates/nerve/web/index.html`, `machine.html` | the lab page and the machine room |
| `site/`, `scripts/build_site.py` | the static build of both pages on the WebAssembly core |
| `nerve/` | the Python reference implementation (paper 1), with the compiler to exact weights |
| `docs/paper/`, `docs/paper2/` | the papers; every number generated from committed logs |
| `docs/dragology-demos.md` | all 115 Dragology demos, ranked for Nerve; which are built |
