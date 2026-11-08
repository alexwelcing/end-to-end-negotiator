#!/usr/bin/env bash
# Train one model and record it.
# usage: experiments/round2/run.sh <spec> <tag> [extra `nerve train` flags...]
# Every run: local GPU, ${EPOCHS:-300} epochs, AdamW weight decay 0.05. The tag names the split; the seed is a
# flag, and each seed gets its own folder (runs/round2/<spec>/<tag>-s<seed>), so no run overwrites another's model.
set -euo pipefail
cd "$(dirname "$0")/../.."
spec="$1"; tag="$2"; shift 2
seed=0
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[$i]}" == --seed ]]; then seed="${args[$((i + 1))]}"; fi
done
BIN="${NERVE_BIN:-runs/bin-cuda/nerve-cont.exe}"
out="runs/round2/$spec/$tag-s$seed"
mkdir -p "$out"
"$BIN" train "specs/$spec.nerve" --cuda --epochs "${EPOCHS:-300}" --weight-decay 0.05 --log-every 100 --out "$out" "$@" > "$out/train.log" 2>&1
"$BIN" record "specs/$spec.nerve" "$out/$spec" "$tag"
