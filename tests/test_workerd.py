"""Parity and PULSE through the real Workers runtime.

1. Python energy == workerd energy on every transition (toggle, counter), over HTTP.
2. A PULSE patch dropped into the export dir changes the Worker's weights to the trainer's,
   bit-identical by digest, with no redeploy. A corrupted patch is rejected and leaves weights untouched.
"""
import json
import pytest
import torch

from nerve.spec import load_spec
from nerve.spec.encode import Encoder
from nerve.data import EpisodeLog, rollout_dataset
from nerve.train import train, TrainConfig
from nerve.model import NerveEnergy
from nerve.serve.export import export_edge, write_patch_index
from nerve.serve import workerd as wd
from nerve.sync import encode_patch
from nerve.sync.pulse import checkpoint_digest

pytestmark = pytest.mark.skipif(not wd.available(), reason="workerd not installed (cd edge && npm install)")
ROOT = ".test-export"      # isolated from edge/export so a real export is never overwritten


def _train(name, epochs, init=None, lr=3e-3, seed=1):
    sp = load_spec(f"specs/{name}.nerve")
    log = EpisodeLog(wd.EDGE / ROOT / f"{name}.jsonl")
    if not list(log):
        rollout_dataset(sp, log, episodes=20, horizon=10, seed=1)
    run = wd.EDGE / ROOT / f"run-{name}-{epochs}-{'warm' if init else 'cold'}"
    train(sp, log, run, TrainConfig(epochs=epochs, log_every=10**6, init=init, lr=lr, seed=seed))
    return sp, run / "model.pt"


def test_workerd_parity_and_pulse():
    # toggle: plain parity. counter: v1 exported as base, v2 (continued training) shipped as a patch.
    sp_t, ck_t = _train("toggle", 10)
    sp_c, ck_c1 = _train("counter", 3)
    _, ck_c2 = _train("counter", 15, init=str(ck_c1))
    m_t = NerveEnergy.load(sp_t, str(ck_t)); m_c1 = NerveEnergy.load(sp_c, str(ck_c1)); m_c2 = NerveEnergy.load(sp_c, str(ck_c2))
    for sp, m, name in ((sp_t, m_t, "toggle"), (sp_c, m_c1, "counter")):
        out = wd.EDGE / ROOT / name; out.mkdir(parents=True, exist_ok=True)
        export_edge(sp, m, out / "edge.json")
    pdir = wd.EDGE / ROOT / "counter" / "patches"; pdir.mkdir(exist_ok=True)
    sd1 = torch.load(ck_c1)["jepa"]; sd2 = torch.load(ck_c2)["jepa"]
    patch = encode_patch(sd1, sd2); patch.save(str(pdir / "0001.nrvp")); write_patch_index(pdir)
    assert patch.n_changed > 0

    with wd.Workerd(port=8799, export_root=ROOT) as w:
        # (1) parity on toggle
        enc = Encoder(sp_t)
        for s in enc.all_states():
            for a in enc.all_actions():
                r = w.step("toggle", s, a)
                assert r["state"] == m_t.step(s, a)
                assert abs(r["energy"] - float(m_t.energies(s, a).min())) < 0.05
        # (2) counter loaded base v1 and applied the patch on load -> weights are v2, bit-identical
        h = json.loads(w.get("/counter/health"))
        assert h["patches"] == ["0001.nrvp"]
        assert h["base_digest"] == checkpoint_digest(sd1)
        assert h["digest"] == checkpoint_digest(sd2)
        enc = Encoder(sp_c)
        diff = 0
        for s in enc.all_states():
            for a in enc.all_actions():
                r = w.step("counter", s, a)
                e2 = float(m_c2.energies(s, a).min()); e1 = float(m_c1.energies(s, a).min())
                assert abs(r["energy"] - e2) < 0.05, (s, a)
                diff += abs(e1 - e2) > 0.05
        assert diff > 0   # the patch actually changed the energies
        # (3) a corrupted patch is rejected; weights stay at v2
        bad = bytearray(patch.data); bad[-1] ^= 0xFF
        (pdir / "0002.nrvp").write_bytes(bytes(bad)); write_patch_index(pdir)
        r = json.loads(w.get("/counter/sync"))
        # rejected either at decode (corrupt zstd frame) or at the digest check; either way weights are untouched
        assert "0002.nrvp" in r.get("rejected", "") and ("digest" in r["rejected"] or "corrupt" in r["rejected"])
        assert r["digest"] == checkpoint_digest(sd2) and r["patches"] == ["0001.nrvp"]
        # sessions persist in the Durable Object
        r1 = w.post("/counter/session/t1", {"action": "inc"}); r2 = w.post("/counter/session/t1", {"action": "inc"})
        assert r1["state"]["count"] == 1 and r2["state"]["count"] == 2
