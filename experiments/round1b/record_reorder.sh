#!/usr/bin/env bash
# The reorder hold-out and valid-candidates sweep (2026-11-07), rescored into one log so the paper can cite it.
# The models were trained by the commands recorded in docs/dragology-demos.md; this script only scores them.
# usage: experiments/round1b/record_reorder.sh   (writes experiments/round1b/reorder_holdout.jsonl)
set -euo pipefail
cd "$(dirname "$0")/../.."
BIN="${NERVE_BIN:-runs/bin-cuda/nerve-cont.exe}"
OUT=experiments/round1b/reorder_holdout.jsonl
: > "$OUT"
for d in h15 f10-h0.5 f10-h0.7 f10-h0.85 f10-h0.9 f0-h0.15 f0-h0.5 f0-h0.7 valid-h0.5 valid-h0.7 valid-h0.85 valid-h0.9; do
  for mode in --all-states --valid-only; do
    "$BIN" eval specs/reorder.nerve "runs/reorder/$d/reorder" "$mode" --json \
      | python -I -c "import json,sys; r=json.loads(sys.stdin.read().strip().splitlines()[-1]); r['run']='$d'; r['mode']='network-only' if '$mode'=='--all-states' else 'spec-assisted'; print(json.dumps(r))" >> "$OUT"
  done
done
wc -l "$OUT"
