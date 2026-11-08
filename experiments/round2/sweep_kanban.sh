#!/usr/bin/env bash
# Q4-Q5: Kanban, 360 boards among 20,736 slot tuples. Pass the hold-outs to run, e.g. `sweep_kanban.sh 0.5`.
set -euo pipefail
cd "$(dirname "$0")"
for h in "$@"; do
  for seed in 0 1 2; do
    ./run.sh kanban "h$h" --holdout "$h" --seed "$seed"
  done
done
if [[ " $* " == *" 0.5 "* ]]; then
  ./run.sh kanban "h0.5-saltb" --holdout 0.5 --salt b --seed 0
fi
