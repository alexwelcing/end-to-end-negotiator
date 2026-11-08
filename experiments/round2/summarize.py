"""Summarise experiments/round2/results.jsonl as markdown tables: mean ± sd over seeds per split and mode.

usage: python experiments/round2/summarize.py [spec ...]
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROWS = [json.loads(l) for l in Path(__file__).with_name("results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def split_name(r: dict) -> str:
    if r.get("continuous"):
        return "random pairs"
    if r.get("rule"):
        return f"rule `{r['rule']}`"
    if not r.get("holdout"):
        return "full spec"
    salt = "" if r.get("salt") in (None, "nerve") else f", salt {r['salt']}"
    return f"{r['holdout']:.0%} held out{salt}"


def r_is_continuous(r: dict) -> bool:
    return bool(r.get("continuous"))


def stat(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    m = statistics.mean(xs) * 100
    if len(xs) == 1:
        return f"{m:.1f}%"
    return f"{m:.1f} ± {statistics.stdev(xs) * 100:.1f}%"


def main() -> None:
    wanted = set(sys.argv[1:])
    # A rerun of the same (spec, tag, seed, mode) replaces the earlier row instead of counting twice.
    latest: dict[tuple, dict] = {}
    for r in ROWS:
        latest[(r["spec"], r["tag"], r.get("seed"), r["mode"])] = r
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in latest.values():
        if wanted and r["spec"] not in wanted:
            continue
        # Runs that differ in how they were trained are never pooled.
        groups[(r["spec"], split_name(r), r["mode"], r.get("epochs"), r.get("fact_weight"))].append(r)
    print("| spec | split | mode | seeds | held-out cells | held-out exact | violations | hole, unseen exact | hole, unseen violations |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (spec, split, mode, epochs, fw), rs in sorted(groups.items(), key=lambda kv: tuple(str(k) for k in kv[0])):
        he = [r.get("hole_examples") or {} for r in rs]
        cells = rs[0]["heldout"]["n"]
        # When the split selects nothing, "held-out" falls back to every cell: say so.
        cells_txt = f"{cells} (all)" if cells == rs[0]["all"]["n"] and not r_is_continuous(rs[0]) else str(cells)
        if epochs not in (None, 300) or fw not in (None, 10.0):
            split = f"{split} ({epochs} epochs, fact {fw})"
        print(f"| {spec} | {split} | {mode} | {len({r.get('seed') for r in rs})} | {cells_txt} | "
              f"{stat([r['heldout']['exact'] for r in rs])} | {stat([r['all']['violation_rate'] for r in rs])} | "
              f"{stat([(h.get('heldout') or {}).get('exact') for h in he])} | "
              f"{stat([(h.get('heldout') or {}).get('violation_rate') for h in he])} |")


if __name__ == "__main__":
    main()
