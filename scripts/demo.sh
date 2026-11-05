#!/usr/bin/env bash
# The whole thesis in one run, on a CPU, in about a minute.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
SPEC=${1:-counter}; R=runs/$SPEC; rm -rf "$R"; mkdir -p "$R"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "1. check: the spec is type-checked and run against its own invariants (zero problems or we stop)"
python -m nerve check specs/$SPEC.nerve
step "2. compile: explicit transitions become exact weights — no training, no code"
python -m nerve compile specs/$SPEC.nerve
step "3. rollout: harness policies operate the oracle; every transition + spec-derived negatives go to the append-only log"
python -m nerve rollout specs/$SPEC.nerve --log $R/episodes.v1.jsonl --episodes 40
step "4. train: the JEPA energy learns the dynamics (and the learn-holes the spec could only show by example)"
python -m nerve train specs/$SPEC.nerve --log $R/episodes.v1.jsonl --out $R/v1 --epochs 40 | tail -1
step "5. continue training at RL-regime lr, then PULSE-patch v1 -> v2 (sparse, bit-identical)"
python -m nerve train specs/$SPEC.nerve --log $R/episodes.v1.jsonl --out $R/v2 --epochs 2 --lr 3e-6 --init $R/v1/model.pt | tail -1
python -m nerve patch --prev $R/v1/model.pt --new $R/v2/model.pt --out $R/patches/0001.nrvp
step "6. run it: an inference worker loads v1, applies the patch, and *is* the interface"
python -m nerve step specs/$SPEC.nerve --model $R/v1/model.pt --patches $R/patches inc inc 'set:{"to":7}' arm inc nudge
step "7. export for the Cloudflare Worker (edge/): the forward pass in nn.ts is the entire runtime"
python -m nerve export specs/$SPEC.nerve --model $R/v2/model.pt --out edge/export/$SPEC.json
step "8. evolve: EAT-style supervisor emits TEAM_CONFIGs, the harness scores each with the official eval"
python -m nerve evolve specs/$SPEC.nerve --workdir $R/evo --iterations 2 --budget 1 | grep -v '^  ep'
