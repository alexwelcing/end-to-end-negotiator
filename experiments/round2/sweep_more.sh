#!/usr/bin/env bash
# Q5-Q6: the eight further demos. Finite specs: 15% held out (seed 0, the lab's model) and 50% held out
# (three seeds). The continuous spec: three seeds. Pass the specs to run, e.g. `sweep_more.sh todo sokoban`.
set -euo pipefail
cd "$(dirname "$0")"
for spec in "$@"; do
  if [[ "$spec" == linked_sliders ]]; then
    for seed in 0 1 2; do ./run.sh "$spec" "continuous" --seed "$seed"; done
    continue
  fi
  ./run.sh "$spec" "h0.15" --holdout 0.15 --seed 0
  for seed in 0 1 2; do ./run.sh "$spec" "h0.5" --holdout 0.5 --seed "$seed"; done
done
