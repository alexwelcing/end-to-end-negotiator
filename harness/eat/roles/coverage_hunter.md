# coverage_hunter
Owns: rollout. Every (state, action) pair must appear as a positive in `episodes.v1.jsonl`. Run
`python -m nerve rollout <spec> --log <log> --policy coverage random --episodes N`. Raise N where the previous
eval's per-action conformance was weakest.
