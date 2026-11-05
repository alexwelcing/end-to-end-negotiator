"""Parity through the real Workers runtime: Python energy == workerd energy on every transition."""
import json
import pytest
import torch

from nerve.spec import load_spec
from nerve.spec.encode import Encoder
from nerve.data import EpisodeLog, rollout_dataset
from nerve.train import train, TrainConfig
from nerve.model import NerveEnergy
from nerve.serve.export import export_edge
from nerve.serve import workerd as wd

pytestmark = pytest.mark.skipif(not wd.available(), reason="workerd not installed (cd edge && npm install)")


ROOT = ".test-export"      # isolated from edge/export so a real export is never overwritten


def test_workerd_parity_toggle_and_counter():
    specs = {}
    for name in ("toggle", "counter"):
        sp = load_spec(f"specs/{name}.nerve")
        log = EpisodeLog(wd.EDGE / ROOT / f"{name}.jsonl")
        rollout_dataset(sp, log, episodes=20, horizon=10, seed=1)
        run = wd.EDGE / ROOT / f"run-{name}"
        train(sp, log, run, TrainConfig(epochs=10, log_every=100))
        m = NerveEnergy.load(sp, str(run / "model.pt"))
        out = wd.EDGE / ROOT / name; out.mkdir(parents=True, exist_ok=True)
        export_edge(sp, m, out / "edge.json")
        specs[name] = (sp, m)
    with wd.Workerd(port=8799, export_root=ROOT) as w:
        for name, (sp, m) in specs.items():
            enc = Encoder(sp)
            for s in enc.all_states():
                for a in enc.all_actions():
                    r = w.step(name, s, a)
                    assert r["state"] == m.step(s, a), (name, s, a)
                    assert abs(r["energy"] - float(m.energies(s, a).min())) < 0.05
        # sessions persist in the Durable Object
        r1 = w.post("/counter/session/t1", {"action": "inc"}); r2 = w.post("/counter/session/t1", {"action": "inc"})
        assert r1["state"]["count"] == 1 and r2["state"]["count"] == 2
