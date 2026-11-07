"""Remaining sweep runs via the Python API (no periodic evals -> much faster on CPU)."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]          # repository root
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)
from nerve.spec import load_spec
from nerve.data import EpisodeLog, rollout_dataset
from nerve.train import train, TrainConfig

OUT = ROOT / "docs/paper/runs"; RES = ROOT / "docs/paper/results.jsonl"
done = {json.loads(l)["tag"] for l in RES.read_text().splitlines() if l.strip()} if RES.exists() else set()


def ensure_log(spec_name, episodes=40):
    log = OUT / f"{spec_name}.episodes.v1.jsonl"
    sp = load_spec(f"specs/{spec_name}.nerve")
    if not log.exists():
        lg = EpisodeLog(log)
        for pol in ("exhaustive", "coverage", "random", "adversary"):
            rollout_dataset(sp, lg, episodes=episodes, horizon=20, policy=pol, negatives=3, seed=0)
    return sp, EpisodeLog(log)


def run(tag, spec_name, epochs=300, curve=False, **kw):
    if tag in done:
        return
    sp, log = ensure_log(spec_name)
    cfg = TrainConfig(epochs=epochs, log_every=10 if curve else 10 ** 6, **kw)
    out = OUT / tag; t0 = time.time()
    try:
        ev = train(sp, log, out, cfg)
    except Exception as e:  # noqa
        rec = {"tag": tag, "spec": spec_name, "cfg": cfg.__dict__, "error": str(e)[-800:]}
    else:
        rec = {"tag": tag, "spec": spec_name, "cfg": cfg.__dict__, "epochs": epochs, "wall": round(time.time() - t0, 1),
               "score": ev["score"], "heldout": ev["heldout"]["exact"], "heldout_n": ev["heldout"]["n"],
               "all": ev["all"]["exact"], "all_n": ev["all"]["n"], "violations": ev["all"]["violation_rate"],
               "learned_only": ev["learned_only"]["exact"], "holes": ev.get("hole_examples"),
               "params": ev["params"], "seconds": ev["seconds"], "coverage": ev.get("coverage"),
               "mean_abs_err_frac": ev["all"].get("mean_abs_err_frac")}
        hist = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines() if l.strip()]
        rec["curve"] = [(h["epoch"], h.get("heldout"), h.get("learned_only"), h.get("all")) for h in hist if "heldout" in h or "all" in h]
    with open(RES, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(tag, {k: rec.get(k) for k in ("heldout", "all", "learned_only", "violations", "wall", "error")}, flush=True)


M = dict(weight_decay=0.05, compile_circuit=False)
# 1. remaining main seeds
for seed in (1, 2):
    for s in ("toggle", "counter", "slider", "negotiate"):
        run(f"{s}_main_s{seed}", s, seed=seed, **M)
# 2. ablations (negotiate with curves for the no-fact run)
run("negotiate_nofact_s0", "negotiate", seed=0, fact_weight=0.0, curve=True, **M)
for s in ("slider", "counter"):
    run(f"{s}_nofact_s0", s, seed=0, fact_weight=0.0, **M)
for s in ("negotiate", "slider", "counter"):
    run(f"{s}_purejepa_s0", s, seed=0, fact_weight=0.0, ord_weight=0.0, **M)
    run(f"{s}_wd1e-4_s0", s, seed=0, weight_decay=1e-4, compile_circuit=False)
# 3. hold-out fraction
for s in ("negotiate", "slider", "counter"):
    for h in (0.3, 0.5):
        run(f"{s}_hold{h}_s0", s, seed=0, holdout=h, **M)
    run(f"{s}_hold0.3_nofact_s0", s, seed=0, holdout=0.3, fact_weight=0.0, **M)
# 4. more seeds for the no-fact ablation on negotiate
for seed in (1, 2):
    run(f"negotiate_nofact_s{seed}", "negotiate", seed=seed, fact_weight=0.0, **M)
# 5. circuit on (full energy)
for s in ("counter", "negotiate"):
    run(f"{s}_circuit_s0", s, seed=0, weight_decay=0.05, compile_circuit=True)
# 6. continuous slider (GPU run used 200 epochs, d=128)
run("slider_continuous_main_s0", "slider_continuous", epochs=200, seed=0, d=128, weight_decay=0.05)
print("SWEEP2 DONE", flush=True)
