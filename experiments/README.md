# Experiments

Every experiment that produces a number we quote lives here, so it can be rerun from a clean checkout.

A round is a folder with:

- `README.md`: the questions, written **before** running, each with a prediction and what result would change
  our minds; then the results, appended after, next to the predictions.
- the scripts that ran it (training commands, sweeps), committed;
- `results.jsonl`: one line per (trained model, scoring mode), written by the scripts, never edited by hand;
- `nerve summarize` output pasted into the README, with mean ± sd over seeds.

Rules from round 1:

- At least three seeds for anything in a table, and a second hold-out split (salt) where a claim depends on it.
- Report the network on its own (`nerve eval --all-states`) next to the network with the spec's invariants
  (`nerve eval`, the default for new models). They answer different questions.
- Prefer structured hold-outs (`--holdout-rule`, a whole region of the state space) to random cells when the
  question is whether the network learned a rule.
- Trained models stay in the ignored `runs/`; their `model.json` records how they were trained and is never
  edited after the fact.
