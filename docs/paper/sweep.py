"""Seed sweep + ablations for the paper, on CPU. Writes one JSON line per run to results.jsonl."""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
OUT = Path(REPO) / "docs/paper/runs"; OUT.mkdir(parents=True, exist_ok=True)
RES = Path(REPO) / "docs/paper/results.jsonl"
done = set()
if RES.exists():
    for l in RES.read_text().splitlines():
        if l.strip():
            d = json.loads(l); done.add(d["tag"])


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=REPO, capture_output=True, text=True, **kw)


def ensure_log(spec, episodes=40):
    log = OUT / f"{spec}.episodes.v1.jsonl"
    if not log.exists():
        r = sh(f"python3 -m nerve rollout specs/{spec}.nerve --log {log} --episodes {episodes}")
        assert r.returncode == 0, r.stderr
    return log


def run(tag, spec, args, epochs=300):
    if tag in done:
        return
    log = ensure_log(spec)
    out = OUT / tag
    t0 = time.time()
    r = sh(f"python3 -m nerve train specs/{spec}.nerve --log {log} --out {out} --epochs {epochs} {args}")
    if r.returncode != 0:
        rec = {"tag": tag, "spec": spec, "args": args, "error": r.stderr[-800:]}
    else:
        ev = json.loads((out / "eval.json").read_text())
        rec = {"tag": tag, "spec": spec, "args": args, "epochs": epochs, "wall": round(time.time() - t0, 1),
               "score": ev["score"], "heldout": ev["heldout"]["exact"], "heldout_n": ev["heldout"]["n"],
               "all": ev["all"]["exact"], "all_n": ev["all"]["n"], "violations": ev["all"]["violation_rate"],
               "learned_only": ev["learned_only"]["exact"], "holes": ev.get("hole_examples"),
               "params": ev["params"], "seconds": ev["seconds"], "coverage": ev.get("coverage"),
               "mean_abs_err_frac": ev["all"].get("mean_abs_err_frac")}
        # learning curve
        hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines() if l.strip()]
        rec["curve"] = [(h["epoch"], h.get("heldout"), h.get("learned_only"), h.get("all")) for h in hist if "heldout" in h or "all" in h]
    with open(RES, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(tag, {k: rec.get(k) for k in ("heldout", "all", "learned_only", "violations", "wall", "error")}, flush=True)


MAIN = "--weight-decay 0.05 --no-circuit"
# 1. main config, three seeds, four hand-written finite specs
for seed in (0, 1, 2):
    for spec in ("toggle", "counter", "slider", "negotiate"):
        run(f"{spec}_main_s{seed}", spec, f"{MAIN} --seed {seed}")
# 2. ablations (seed 0): no factored head; no factored + no ordinal readout (pure JEPA); default weight decay
for spec in ("negotiate", "slider", "counter"):
    run(f"{spec}_nofact_s0", spec, f"{MAIN} --seed 0 --fact-weight 0")
    run(f"{spec}_purejepa_s0", spec, f"{MAIN} --seed 0 --fact-weight 0 --ord-weight 0")
    run(f"{spec}_wd1e-4_s0", spec, "--weight-decay 1e-4 --no-circuit --seed 0")
# 3. generalisation vs. hold-out fraction (seed 0)
for spec in ("negotiate", "slider", "counter"):
    for h in (0.3, 0.5):
        run(f"{spec}_hold{h}_s0", spec, f"{MAIN} --seed 0 --holdout {h}")
    run(f"{spec}_hold0.3_nofact_s0", spec, f"{MAIN} --seed 0 --holdout 0.3 --fact-weight 0")
# 4. more seeds for the ablation on negotiate
for seed in (1, 2):
    run(f"negotiate_nofact_s{seed}", "negotiate", f"{MAIN} --seed {seed} --fact-weight 0")
# 5. continuous slider (the GPU run used 200 epochs, d=128)
run("slider_continuous_main_s0", "slider_continuous", "--weight-decay 0.05 --seed 0 --d 128", epochs=200)
# 6. with the circuit on (full energy), one seed each, to show all=1.0 and the learned-only readout
for spec in ("counter", "negotiate"):
    run(f"{spec}_circuit_s0", spec, "--weight-decay 0.05 --seed 0")
print("SWEEP DONE", flush=True)
