#!/usr/bin/env bash
# Q7: every demo trained on its whole spec (--holdout 0), through this pipeline, so every model the lab serves
# comes from a committed command. Finite specs on three seeds (does training reliably get there?), except the
# two largest on one. Continuous specs get round 1's longer recipe (1000 epochs, 8000 pairs), one seed.
# Pass the specs to run, e.g. `sweep_full.sh hanoi todo`.
set -euo pipefail
cd "$(dirname "$0")"
for spec in "$@"; do
  case "$spec" in
    linked_sliders|tracks|slider_continuous) EPOCHS=1000 ./run.sh "$spec" "full" --pairs 8000 --seed 0 ;;
    kanban|scheduler) ./run.sh "$spec" "full" --holdout 0 --seed 0 ;;
    *) for seed in 0 1 2; do ./run.sh "$spec" "full" --holdout 0 --seed "$seed"; done ;;
  esac
done
