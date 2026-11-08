"""Score one trained model and append the results to experiments/round2/results.jsonl.

usage: python experiments/round2/record.py <spec name> <model dir> <tag>

Finite specs are scored twice by `nerve eval --json`: with the spec's invariants (as trained, the default for
new models) and as the network on its own (`--all-states`). Continuous specs carry their score in model.json.
Lines are only ever appended, never edited.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

BIN = os.environ.get("NERVE_BIN", "runs/bin-cuda/nerve-cont.exe")
OUT = Path(os.environ.get("NERVE_RESULTS", Path(__file__).with_name("results.jsonl")))


def eval_line(spec: str, model: Path, *flags: str) -> dict:
    # Windows' CreateProcess needs an absolute path to launch a binary given relative to the repo.
    cmd = [str(Path(BIN).resolve()), "eval", f"specs/{spec}.nerve", str(model), "--json", *flags]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return json.loads(out[-1])


def main() -> None:
    spec, model, tag = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    meta = json.loads((model / "model.json").read_text(encoding="utf-8"))
    rows = []
    # Size and cost travel with every row, so a table can be built from the log alone.
    cost = {"params": meta["eval"].get("params"), "seconds": meta["eval"].get("seconds")}
    if meta.get("kind") == "continuous":
        e, t = meta["eval"], meta["train"]
        rows.append({"spec": spec, "model": str(model), "mode": "network", "seed": t["seed"], "holdout": None,
                     "salt": None, "rule": None, "epochs": t["epochs"], "heldout": e["all"], "all": e["all"],
                     "continuous": True, **cost})
    else:
        # Both modes are forced explicitly, whatever candidate set the model was trained with.
        trained = meta.get("candidates", "all")
        rows.append({**eval_line(spec, model, "--valid-only"), "mode": "spec-assisted", "trained_candidates": trained, **cost})
        rows.append({**eval_line(spec, model, "--all-states"), "mode": "network-only", "trained_candidates": trained, **cost})
    # One O_APPEND write per model, so concurrent sweeps cannot interleave lines.
    blob = "".join(json.dumps({"round": 2, "tag": tag, **r}) + "\n" for r in rows).encode("utf-8")
    fd = os.open(OUT, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0))
    try:
        os.write(fd, blob)
    finally:
        os.close(fd)
    for r in rows:
        h = r["heldout"]
        print(f"{spec:<16} {tag:<22} {r['mode']:<14} held-out {h['exact']:.4f} of {h['n']}")


if __name__ == "__main__":
    main()
