# GPU training on Hugging Face

Two ways to move training off the CPU; both use the same code path (`nerve.train.train`, `--device auto`).

## Jobs (recommended: pay per second, nothing left running)

```
export HF_TOKEN=...           # write + jobs scope; in a cloud session add it as an environment secret
pip install huggingface_hub
nerve train specs/negotiate.nerve --log runs/negotiate/episodes.v1.jsonl --out runs/negotiate/gpu \
    --epochs 600 --d 128 --remote hf --flavor a10g-small
```

`nerve train --remote hf` uploads the spec, the episode log and the config to the private dataset repo
`AlexWelcing/nerve-runs` under `runs/<spec>/<timestamp>/in/`, launches `hf/jobs/train_job.py` as a uv Job on the
requested flavor, waits, and downloads `model.pt`, `eval.json` and `history.jsonl` into `--out`. The job clones the
`nerve` branch, so what runs on the GPU is exactly what is pushed.

The EAT supervisor can send every iteration's training there: `nerve evolve ... --remote hf`.

## Space (persistent GPU)

A Docker Space on GPU hardware can run `nerve serve` or a training endpoint, but it bills by the hour while idle.
Use it for the serving side (a Container origin for the edge Worker) rather than for training.

Flavors worth knowing: `t4-small` (cheapest), `a10g-small`, `a10g-large`, `a100-large`. Nerve's networks are tiny;
the win is the thousands of candidate-energy evaluations per step and the 600-epoch grokking schedules, which a T4
does in minutes.
