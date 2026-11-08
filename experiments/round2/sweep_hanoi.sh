#!/usr/bin/env bash
# Q1-Q3: Hanoi's written rules and learned solver. 15 runs.
set -euo pipefail
cd "$(dirname "$0")"
for seed in 0 1 2; do
  for h in 0.15 0.5 0.7; do
    ./run.sh hanoi "h$h" --holdout "$h" --seed "$seed"
  done
  ./run.sh hanoi "h0.5-saltb" --holdout 0.5 --salt b --seed "$seed"
  # the whole region where the largest disk is on the middle peg: 27 of 81 positions
  ./run.sh hanoi "rule-d4mid" --holdout-rule "d4 == 1" --seed "$seed"
done
