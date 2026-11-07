"""PULSE patch size vs learning rate on the negotiate seed-0 model (continued training on the same log)."""
import json, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import torch
from nerve.spec import load_spec
from nerve.data import EpisodeLog
from nerve.train import train, TrainConfig
from nerve.sync import encode_patch, apply_patch

sp = load_spec("specs/negotiate.nerve")
log = EpisodeLog(str(ROOT / "docs/paper/runs/negotiate.episodes.v1.jsonl"))
base = str(ROOT / "docs/paper/runs/negotiate_main_s0/model.pt")
done = {(json.loads(l)["lr"], json.loads(l)["epochs"]) for l in open(str(ROOT / "docs/paper/pulse.jsonl")) if l.strip()} if os.path.exists(str(ROOT / "docs/paper/pulse.jsonl")) else set()
out = open(str(ROOT / "docs/paper/pulse.jsonl"), "a")
prev = torch.load(base, map_location="cpu")["jepa"]
full = sum(v.numel() * 2 for v in prev.values() if v.is_floating_point())
for lr in ("3e-6", "3e-5", "3e-4", "3e-3"):
    for ep in (1, 5):
        if (lr, ep) in done:
            continue
        d = str(ROOT / f"docs/paper/runs/pulse_lr{lr}_ep{ep}")
        train(sp, log, d, TrainConfig(epochs=ep, lr=float(lr), weight_decay=0.05, compile_circuit=False, init=base, log_every=10**6))
        new = torch.load(d + "/model.pt", map_location="cpu")["jepa"]
        p = encode_patch(prev, new)
        rec = apply_patch(prev, p)
        assert all(torch.equal(rec[k].view(torch.int16), new[k].view(torch.int16)) for k in new)
        r = {"lr": lr, "epochs": ep, "bytes": len(p.data), "full_bf16_bytes": full, "ratio": round(full / len(p.data), 2),
             "sparsity": round(p.sparsity, 4), "n_changed": p.n_changed, "n_total": p.n_total}
        out.write(json.dumps(r) + "\n"); out.flush(); print(r, flush=True)
