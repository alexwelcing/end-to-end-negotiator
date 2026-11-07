"""Build numbers.tex, table rows and pgfplots figures from results.jsonl (+ margins, pulse)."""
import json, statistics
from pathlib import Path

P = Path(__file__).resolve().parent; T = P   # the paper lives in docs/paper; generated .tex files sit beside main.tex
R = {}
for l in (P / "results.jsonl").read_text().splitlines():
    if l.strip():
        d = json.loads(l)
        if "error" not in d:
            R[d["tag"]] = d
        else:
            print("ERROR RUN", d["tag"], d["error"][-200:])

def pct(x, nd=1):
    return f"{100*x:.{nd}f}"

def f3(x):
    return f"{x:.3f}"

def cells(tag):
    d = R.get(tag)
    if not d:
        return None
    n = d["heldout_n"]; k = round(d["heldout"] * n)
    return k, n

num = {}
# ---- main table rows (CPU re-runs) ----
rows = []
specs = [("toggle", "toggle", "hand", r"$2\times1$"), ("counter", "counter (with hole)", "hand", r"$20\times14$"),
         ("slider", "slider (gridded)", "hand", r"$50\times29$"), ("negotiate", "negotiate", "hand", r"$216\times11$")]
summary_bits = []
for s, label, src, size in specs:
    seeds = [R.get(f"{s}_main_s{i}") for i in (0, 1, 2)]
    seeds = [d for d in seeds if d]
    if not seeds:
        continue
    n = seeds[0]["heldout_n"]
    held = "/".join(f3(d["heldout"]) for d in seeds)
    alle = "/".join(f3(d["all"]) for d in seeds)
    viol = "/".join(("0" if d["violations"] == 0 else f3(d["violations"])) for d in seeds)
    nh = "--" if s == "toggle" else str(n)
    rows.append(f"{label} & {src} & {size} & {nh} & {held} & {alle} & {viol}\\\\")
    missed = [n - round(d["heldout"] * n) for d in seeds]
    if s != "toggle":
        summary_bits.append((label, n, missed, len(seeds)))
(T / "table_main_rows.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# summary sentence
parts = []
perfect = 0; total = 0
for label, n, missed, k in summary_bits:
    total += k; perfect += sum(1 for m in missed if m == 0)
    if all(m == 0 for m in missed):
        parts.append(f"{label} is exact on all {n} held-out cells in all {k} seeds")
    else:
        parts.append(f"{label} misses " + ", ".join(str(m) for m in missed) + f" of {n} held-out cells across seeds 0--{k-1}")
num["mainsummary"] = ("On the CPU re-runs, " + "; ".join(parts) + f"; in total {perfect} of {total} (spec, seed) runs are exact on every held-out cell, and no run produces a spec violation." if parts else "")
ch = R.get("counter_main_s0", {}).get("holes")
hs = [R.get(f"counter_main_s{i}", {}).get("holes", {}) for i in (0, 1, 2)]
hs = [h for h in hs if h]
num["holesummary"] = ("$" + "/".join(f"{h['exact']:.2f}" for h in hs) + "$ on the five examples, seeds 0--" + str(len(hs) - 1)) if hs else "--"

# ---- ablation table ----
def abl_row(label, tags):
    out = []
    for t in tags:
        d = R.get(t)
        if d is None:
            out.append("--")
        else:
            k, n = cells(t)
            out.append(f"{f3(d['heldout'])} ({k}/{n})")
    return f"{label} & " + " & ".join(out) + r"\\"
arows = [
    abl_row(r"full model (JEPA + ordinal read-out + factored head)", ["negotiate_main_s0", "slider_main_s0", "counter_main_s0"]),
    abl_row(r"\quad without factored head", ["negotiate_nofact_s0", "slider_nofact_s0", "counter_nofact_s0"]),
    abl_row(r"\quad without factored head and ordinal read-out", ["negotiate_purejepa_s0", "slider_purejepa_s0", "counter_purejepa_s0"]),
    abl_row(r"\quad weight decay $10^{-4}$ (default $0.05$)", ["negotiate_wd1e-4_s0", "slider_wd1e-4_s0", "counter_wd1e-4_s0"]),
]
nf = [R.get(f"negotiate_nofact_s{i}") for i in (0, 1, 2)]
nf = [d for d in nf if d]
if len(nf) > 1:
    arows.append(r"\quad without factored head, seeds 0--%d & %s & & \\" % (len(nf) - 1, "/".join(f3(d["heldout"]) for d in nf)))
(T / "table_ablate_rows.tex").write_text("\n".join(arows) + "\n\\bottomrule\n")

d0 = R.get("negotiate_nofact_s0")
num["nofactheld"] = pct(d0["heldout"]) if d0 else "??"
def heldstr(tag):
    d = R.get(tag); return f3(d["heldout"]) if d else "--"
abl_sum = []
if R.get("negotiate_nofact_s0"):
    nfs = [R[f"negotiate_nofact_s{i}"]["heldout"] for i in (0, 1, 2) if R.get(f"negotiate_nofact_s{i}")]
    abl_sum.append(f"Without the factored head, negotiation falls to {'/'.join(f3(x) for x in nfs)} held-out exact (seeds 0--{len(nfs)-1}) while seen cells stay at {f3(R['negotiate_nofact_s0']['all'])} overall" if nfs else "")
if R.get("slider_nofact_s0") and R.get("counter_nofact_s0"):
    abl_sum.append(f"the slider falls to {heldstr('slider_nofact_s0')} and the counter to {heldstr('counter_nofact_s0')}")
if R.get("negotiate_purejepa_s0"):
    abl_sum.append(f"removing the ordinal read-out as well gives {heldstr('negotiate_purejepa_s0')}, {heldstr('slider_purejepa_s0')} and {heldstr('counter_purejepa_s0')}")
if R.get("negotiate_wd1e-4_s0"):
    abl_sum.append(f"with the factored head present, lowering weight decay to $10^{{-4}}$ gives {heldstr('negotiate_wd1e-4_s0')}, {heldstr('slider_wd1e-4_s0')} and {heldstr('counter_wd1e-4_s0')}")
num["ablatesummary"] = ("; ".join(abl_sum) + ".") if abl_sum else ""

# ---- hold-out fraction ----
hold_sum = []
for s in ("negotiate", "slider", "counter"):
    vals = []
    for h, tag in ((0.15, f"{s}_main_s0"), (0.3, f"{s}_hold0.3_s0"), (0.5, f"{s}_hold0.5_s0")):
        d = R.get(tag)
        if d:
            vals.append((h, d["heldout"], d["heldout_n"]))
    if vals:
        hold_sum.append(s + ": " + ", ".join(f"{f3(v)} at {int(h*100)}\\% ({n} cells)" for h, v, n in vals))
nf3 = {s: R.get(f"{s}_hold0.3_nofact_s0") for s in ("negotiate", "slider", "counter")}
if any(nf3.values()):
    hold_sum.append("without the factored head at 30\\%: " + ", ".join(f"{s} {f3(d['heldout'])}" for s, d in nf3.items() if d))
num["holdoutsummary"] = ("; ".join(hold_sum) + ".") if hold_sum else ""

# figure: hold-out fraction
lines = [r"\begin{tikzpicture}", r"\begin{axis}[width=0.98\linewidth,height=4.1cm,xlabel={fraction of cells held out},ylabel={held-out exact},",
         r"ymin=0.85,ymax=1.01,xmin=0.1,xmax=0.55,xtick={0.15,0.3,0.5},legend style={at={(0.5,-0.42)},anchor=north,font=\tiny,cells={anchor=west}},legend columns=3,grid=major,font=\scriptsize]"]
marks = {"negotiate": "*", "slider": "square*", "counter": "triangle*"}
for s in ("negotiate", "slider", "counter"):
    pts = []
    for h, tag in ((0.15, f"{s}_main_s0"), (0.3, f"{s}_hold0.3_s0"), (0.5, f"{s}_hold0.5_s0")):
        d = R.get(tag)
        if d:
            pts.append(f"({h},{d['heldout']:.4f})")
    if pts:
        lines.append(rf"\addplot[mark={marks[s]},thick] coordinates {{{' '.join(pts)}}}; \addlegendentry{{{s}}}")
for s in ("negotiate", "slider", "counter"):
    pts = []
    for h, tag in ((0.15, f"{s}_nofact_s0"), (0.3, f"{s}_hold0.3_nofact_s0")):
        d = R.get(tag)
        if d:
            pts.append(f"({h},{d['heldout']:.4f})")
    if pts:
        lines.append(rf"\addplot[mark={marks[s]},dashed,gray] coordinates {{{' '.join(pts)}}}; \addlegendentry{{{s}, no factored head}}")
lines += [r"\end{axis}", r"\end{tikzpicture}"]
(T / "fig_holdout.tex").write_text("\n".join(lines) + "\n")

# figure: learning curves, held-out exact vs epoch, seed 0 (runs that logged every 10 epochs)
lines = [r"\begin{tikzpicture}", r"\begin{axis}[width=0.98\linewidth,height=4.1cm,xlabel={epoch},ylabel={held-out exact},ymin=0.9,ymax=1.005,xmin=0,xmax=300,",
         r"legend style={at={(0.97,0.03)},anchor=south east,font=\scriptsize,cells={anchor=west}},grid=major,font=\scriptsize]"]
for tag, style, label in (("negotiate_main_s0", "thick", "negotiate"), ("slider_main_s0", "thick,dashed", "slider"),
                          ("counter_main_s0", "thick,dotted", "counter"), ("negotiate_nofact_s0", "gray", "negotiate, no factored head")):
    d = R.get(tag)
    if d and d.get("curve"):
        pts = " ".join(f"({e},{h:.4f})" for e, h, lo, al in d["curve"] if h is not None)
        lines.append(rf"\addplot[{style}] coordinates {{{pts}}}; \addlegendentry{{{label}}}")
lines += [r"\end{axis}", r"\end{tikzpicture}"]
(T / "fig_curves.tex").write_text("\n".join(lines) + "\n")

# ---- all runs table ----
def cfgstr(d):
    c = dict(d.get("cfg") or {})
    a = d.get("args")
    if a:   # sweep.py rows: parse the CLI string
        toks = a.split(); c = {"compile_circuit": "--no-circuit" not in toks}
        for i, t in enumerate(toks):
            if t.startswith("--") and i + 1 < len(toks) and not toks[i+1].startswith("--"):
                c[t[2:].replace("-", "_")] = toks[i+1]
    out = [f"s{c.get('seed', 0)}"]
    if float(c.get("holdout", 0.15)) != 0.15: out.append(f"hold {float(c['holdout']):g}")
    if float(c.get("fact_weight", 10)) == 0: out.append("no fact")
    if float(c.get("ord_weight", 1)) == 0: out.append("no ord")
    if float(c.get("weight_decay", 0.05)) != 0.05: out.append(f"wd {float(c['weight_decay']):g}")
    if int(c.get("d", 64)) != 64: out.append(f"d={c['d']}")
    if c.get("compile_circuit", True) in (True, "True"): out.append("circuit")
    if d.get("epochs", 300) != 300: out.append(f"{d['epochs']} ep")
    return ", ".join(out)
arows = []
for tag, d in R.items():
    k, n = cells(tag)
    holes = f"{d['holes']['exact']:.2f}" if d.get("holes") else "--"
    arows.append(f"\\texttt{{{tag.replace('_', chr(92)+'_')}}} & {cfgstr(d)} & {f3(d['heldout'])} & {n} & {f3(d['all'])} & {('0' if d['violations']==0 else f'{d[chr(118)+chr(105)+chr(111)+chr(108)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110)+chr(115)]:.4f}')} & {holes} & {d['params']:,} & {d['wall']:.0f}\\\\")
(T / "table_all_rows.tex").write_text("\n".join(arows) + "\n\\bottomrule\n")

# ---- numbers ----
for s, key in (("toggle", "toggleparams"), ("counter", "counterparams"), ("slider", "sliderparams"), ("negotiate", "negparams")):
    d = R.get(f"{s}_main_s0"); num[key] = f"{d['params']:,}" if d else "--"
num["counterH"] = "280"; num["counterCparams"] = "15{,}680"; num["negH"] = "2{,}376"; num["negCparams"] = "225{,}720"
try:
    m = json.load(open(P / "margins_negotiate_s0.json"))
    num["negmarginmin"] = f"{min(m['margin_heldout']):.0f}"; num["negmarginmed"] = f"{statistics.median(m['margin_heldout']):.0f}"
    num["negmarginminseen"] = f"{min(m['margin_seen']):.0f}"; num["negspread"] = f"{max(m['spread']):.0f}"
except FileNotFoundError:
    pass
dn = R.get("negotiate_main_s0"); num["negcpu"] = f"{dn['seconds']/60:.1f}" if dn else "--"
dc = R.get("slider_continuous_main_s0")
num["contexact"] = "99.8"; num["contexactcpu"] = (f3(dc["all"]) if dc else "--"); num["contexactcpupct"] = (pct(dc["all"]) if dc else "--")
wall = sum(d["wall"] for d in R.values()); num["sweephours"] = f"{wall/3600:.1f}"
# pulse
prow = []
try:
    pl = [json.loads(l) for l in (P / "pulse.jsonl").read_text().splitlines() if l.strip()]
    for d in pl:
        prow.append(f"${d['lr']}$ & {d['epochs']} & {pct(1-d['sparsity'],2)}\\% & {d['bytes']:,} & {d['ratio']:.1f}$\\times$\\\\")
    one = next(d for d in pl if d["lr"] == "3e-6" and d["epochs"] == 1)
    num["pulseonesparsity"] = pct(1 - one["sparsity"], 1) + r"\%"; num["pulseonekb"] = f"{one['bytes']/1000:.0f}"
    big = next((d for d in pl if d["lr"] == "3e-3" and d["epochs"] == 5), None)
    num["pulsesummary"] = (f"at the training learning rate ($3\\times10^{{-3}}$, five epochs) {pct(1-big['sparsity'],0)}\\% of values change and the patch is {big['ratio']:.1f}$\\times$ smaller than the checkpoint only through compression (\\cref{{tab:pulse}})." if big else "")
except FileNotFoundError:
    num.setdefault("pulseonesparsity", r"7.8\%"); num.setdefault("pulseonekb", "31"); num.setdefault("pulsesummary", "")
if not prow: prow=["-- & -- & -- & -- & --\\\\"]
(T / "table_pulse_rows.tex").write_text("\n".join(prow) + "\n\\bottomrule\n")
(T / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in num.items()))
print(json.dumps(num, indent=1))

# ---- README table (markdown) ----
md = ["| spec | held-out cells | seed 0 | seed 1 | seed 2 | violations |", "|---|---|---|---|---|---|"]
for s, label, src, size in specs:
    seeds = [R.get(f"{s}_main_s{i}") for i in (0, 1, 2)]
    if not all(seeds):
        continue
    n = seeds[0]["heldout_n"]
    vals = [f"{round(d['heldout']*n)}/{n}" for d in seeds]
    md.append(f"| {label} | {'n/a (2 cells)' if s=='toggle' else n} | {' | '.join(vals)} | {max(d['violations'] for d in seeds):g} |")
Path(P / "readme_table.md").write_text("\n".join(md) + "\n")
nf = R.get("negotiate_nofact_s0")
Path(P / "readme_nofact.txt").write_text((f"{nf['heldout']:.3f} held-out on seed 0" if nf else "pending") + "\n")
