"""Build the lab's model set from round-2 results: models/served/<spec>/ for every spec trained on its full spec.

usage: python experiments/round2/serve_set.py

For each spec, the full-spec seed with the highest score (spec-assisted exact for finite specs, within-tolerance
for continuous ones; the lowest seed on ties) is copied unchanged to models/served/<spec>/, which is committed
(about half a megabyte per network) so that a fresh clone can run the lab without training. The choice and its
scores are written to experiments/round2/served.json, so the served set is always traceable to the run that
produced it. Model files are copied, never edited.
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
SERVED = ROOT / "models" / "served"


def main() -> None:
    latest: dict[tuple, dict] = {}
    for line in (HERE / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["spec"], r["tag"], r.get("seed"), r["mode"])] = r
    best: dict[str, dict] = {}
    for (spec, tag, seed, mode), r in sorted(latest.items(), key=lambda kv: str(kv[0])):
        if tag != "full" or mode not in ("spec-assisted", "network"):
            continue
        score = (r["all"]["exact"], -r["all"]["violation_rate"], -(seed or 0))
        if spec not in best or score > best[spec]["_score"]:
            best[spec] = {**r, "_score": score}
    if SERVED.exists():
        shutil.rmtree(SERVED)
    SERVED.mkdir(parents=True)
    manifest = {}
    for spec, r in sorted(best.items()):
        src = ROOT / r["model"]
        shutil.copytree(src, SERVED / spec)
        manifest[spec] = {"model": r["model"], "seed": r.get("seed"), "exact": r["all"]["exact"],
                          "violation_rate": r["all"]["violation_rate"], "cells": r["all"]["n"]}
        print(f"{spec:<18} seed {r.get('seed')}  exact {r['all']['exact']:.4f}  violations {r['all']['violation_rate']:.4f}  <- {r['model']}")
    (HERE / "served.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
