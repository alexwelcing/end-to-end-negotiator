# Measurements paper 2 cites that are not in a results log

Recorded 2026-11-07 on the development machine (Windows 11, NVIDIA RTX A4500, driver 595.71, `nvidia-smi`).
Each entry gives the command or the observation it comes from so it can be redone.

## Finite step time (§4.1, §4.5)

`nerve eval specs/reorder.nerve runs/reorder/f10-h0.7/reorder --all-states`: 3,000 (state, action) cells, each
an argmin over all 3,125 position tuples, 2.48 s wall-clock including process start and model load, on the CPU
(the candidates' target embeddings are encoded once per set of weights). Under 1 ms per step.

## Continuous step time (§4.5)

`POST /api/tracks/step` against the lab serving `runs/cont-big/tracks` (1,000 epochs, 8,000 pairs), CPU:
44 ms for one step (eleven descent seeds × three discrete assignments, sixty Adam iterations). A full
decision field for tracks (7,200 pixels × three `vary` searches) took 35 s with the continuous network's term
and 1.1 s without, which is why the field omits it (`LearnedEnergy::cheap`).

## Drift on the continuous slider (§5.5)

Observed in the lab with the round-1 `slider_continuous` model (300 epochs, 4,000 pairs), clicking `move`
while not grabbed, where the spec says the value does not change: network value 0 → 0.59 → 1.41 over two
clicks (the page keeps the network's answer as the state). The 1,000-epoch model served since scores 100%
within 1% of range on fresh transitions; the drift is smaller but the mechanism is the same.

## Training time is contended wall-clock (§2.4)

`seconds` in each `model.json` is wall-clock for the run; the round-2 sweep ran up to five trainers on the one
GPU at a time (`experiments/round2/sweep_full.sh` chains were started in parallel), so the cost figure is an
upper bound on isolated cost and the ratios between runs are approximate.
