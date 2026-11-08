"""Generate every number in paper 2 from experiments/round2/results.jsonl.

usage: python docs/paper2/make_tables.py

Writes numbers.tex (macros), table_round2_rows.tex (one row per interface trained on its whole spec) and
fig_cost.tex (GPU seconds against cells). Rows are the latest per (spec, tag, seed, mode); only tag `full`
counts. Parameters and seconds come from each row, or from the run's model.json when an older row lacks them.
"""
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "experiments" / "round2" / "results.jsonl"

# Display name, Dragology source (or paper 1), and what the interface tests, in the paper's words.
META = {
    "toggle": ("toggle", "paper 1", "two states, one gesture"),
    "counter": ("counter", "paper 1", "a bounded count and a \\texttt{learn} hole"),
    "slider": ("slider", "paper 1", "a gridded value behind a guard"),
    "negotiate": ("negotiate", "paper 1", "Deal or No Deal as a finite game"),
    "reorder": ("reorder", "\\texttt{perm}", "five items, every order one move away"),
    "tracks": ("tracks", "\\texttt{lock-past}", "a block on rails: continuous position, discrete rail"),
    "slider_continuous": ("slider (cont.)", "paper 1", "a continuous value behind a guard"),
    "hanoi": ("hanoi", "\\texttt{hanoi}", "guards over other fields; a solver by example"),
    "kanban": ("kanban", "\\texttt{kanban}", "four cards in three columns"),
    "fifteen": ("fifteen", "\\texttt{fifteen}", "sliding tiles, $3\\times2$"),
    "rush_hour": ("rush hour", "\\texttt{rush-hour}", "six cars that block each other"),
    "sokoban": ("sokoban", "\\texttt{sokoban}", "pushing: three objects in one rule"),
    "insert_remove": ("insert/remove", "\\texttt{insert-and-remove}", "a list whose length changes"),
    "todo": ("todo", "\\texttt{todo}", "reorder plus a flag per card"),
    "scheduler": ("scheduler", "\\texttt{scheduler}", "three meetings, two rooms, ends that cannot cross"),
    "three_way_switch": ("3-way switch", "\\texttt{study-three-way-switch}", "\\texttt{between} over three places"),
    "linked_sliders": ("linked sliders", "\\texttt{linked-sliders}", "three continuous values that move together"),
}


def load() -> list[dict]:
    latest: dict[tuple, dict] = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[(r["spec"], r["tag"], r.get("seed"), r["mode"])] = r
    return [r for r in latest.values() if r["tag"] == "full" and r["mode"] in ("spec-assisted", "network")]


def cost(r: dict) -> tuple:
    if r.get("params") is not None:
        return r["params"], r.get("seconds")
    mj = ROOT / r["model"] / "model.json"
    if mj.exists():
        e = json.loads(mj.read_text(encoding="utf-8"))["eval"]
        return e.get("params"), e.get("seconds")
    return None, None


def fmt_int(x) -> str:
    return "--" if x is None else f"{int(x):,}"


def main() -> None:
    by_spec: dict[str, list[dict]] = defaultdict(list)
    for r in load():
        by_spec[r["spec"]].append(r)
    rows, points, params_seen = [], [], []
    n_exact = n_finite = n_cont = total_seconds = 0
    max_cells = (0, "")
    for spec in sorted(by_spec, key=lambda s: by_spec[s][0]["all"]["n"]):
        rs = by_spec[spec]
        name, src, tests = META.get(spec, (spec.replace("_", "\\_"), "--", ""))
        cont = bool(rs[0].get("continuous"))
        cells = rs[0]["all"]["n"]
        exact = min(r["all"]["exact"] for r in rs)
        viol = max(r["all"]["violation_rate"] for r in rs)
        params, secs = cost(rs[0])
        secs_all = [cost(r)[1] for r in rs if cost(r)[1] is not None]
        seeds = len({r.get("seed") for r in rs})
        if cont:
            n_cont += 1
            states, cells_txt = "cont.", f"{cells:,} random"
            exact_txt = f"{exact * 100:.1f}\\% within 1\\%"
        else:
            n_finite += 1
            n_exact += exact == 1.0 and viol == 0.0
            states, cells_txt = fmt_int(rs[0].get("candidate_count")), f"{cells:,}"
            exact_txt = f"{round(exact * cells):,}/{cells:,}" if exact < 1 else "all"
            if secs is not None:
                points.append((cells, secs, name))
            if cells > max_cells[0]:
                max_cells = (cells, name)
        if secs_all:
            total_seconds += sum(secs_all)
        if params is not None:
            params_seen.append(params)
        hole = ""
        he = rs[0].get("hole_examples")
        if he and he.get("n"):
            hits = min(round((r["hole_examples"]["exact"] or 0) * he["n"]) for r in rs)
            hole = f" ({hits}/{he['n']} examples)"
        rows.append(f"{name} & {src} & {tests} & {states} & {cells_txt} & {fmt_int(params)} & "
                    f"{'--' if secs is None else f'{secs:,.0f}'} & {seeds} & {exact_txt}{hole} & {viol * 100:.1f}\\% \\\\")
    # The rule is emitted here: a \bottomrule after an \input inside a tabular is a "Misplaced \noalign".
    (HERE / "table_round2_rows.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8", newline="\n")
    before = {"toggle", "counter", "slider", "slider_continuous", "negotiate", "tracks", "reorder"}
    numbers = {
        "nPlanned": str(len(META)),
        "nBefore": str(len(before)),
        "nNewDemos": str(len(set(META) - before)),
        "minParamsK": f"{min(params_seen) / 1000:.0f}" if params_seen else "--",
        "maxParamsK": f"{max(params_seen) / 1000:.0f}" if params_seen else "--",
        "nFullSpecs": str(n_finite + n_cont),
        "nFiniteSpecs": str(n_finite),
        "nExactSpecs": str(n_exact),
        "nContSpecs": str(n_cont),
        "maxCells": f"{max_cells[0]:,}",
        "maxCellsSpec": max_cells[1],
        "totalGpuMinutes": f"{total_seconds / 60:.0f}",
    }
    (HERE / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in numbers.items()), encoding="utf-8", newline="\n")
    coords = " ".join(f"({c},{s})" for c, s, _ in points)
    labels = "\n".join(
        f"\\node[anchor=west,font=\\tiny] at (axis cs:{c},{s}) {{{n}}};" for c, s, n in points)
    (HERE / "fig_cost.tex").write_text(
        "\\begin{tikzpicture}\n\\begin{axis}[width=0.62\\linewidth,height=0.42\\linewidth,xlabel={(state, action) cells},"
        "ylabel={wall-clock seconds, shared GPU},xmode=log,ymode=log,grid=major,tick label style={font=\\scriptsize},"
        "label style={font=\\small}]\n"
        f"\\addplot[only marks,mark=*,mark size=1.6pt,blue!60!black] coordinates {{{coords}}};\n{labels}\n"
        "\\end{axis}\n\\end{tikzpicture}\n", encoding="utf-8", newline="\n")
    print(f"{n_finite} finite ({n_exact} exact) + {n_cont} continuous interfaces; {len(points)} cost points")


if __name__ == "__main__":
    main()
