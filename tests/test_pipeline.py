import torch
from nerve.spec import load_spec
from nerve.data import EpisodeLog, rollout_dataset
from nerve.train import train, TrainConfig
from nerve.sync import encode_patch, apply_patch
from nerve.serve import NerveRuntime
from nerve.model import NerveEnergy


def test_end_to_end_counter(tmp_path):
    sp = load_spec("specs/counter.nerve")
    log = EpisodeLog(tmp_path / "episodes.v1.jsonl")
    rollout_dataset(sp, log, episodes=30, horizon=15, policy="coverage", seed=0)
    r = train(sp, log, tmp_path / "v1", TrainConfig(epochs=25, log_every=100))
    assert r["all"]["exact"] == 1.0                 # circuit carries explicit transitions
    assert r["all"]["violation_rate"] == 0.0
    assert r["learned_only"]["exact"] > 0.8         # the JEPA actually learned the dynamics
    # continued training at RL-regime lr -> sparse, bit-identical patch
    r2 = train(sp, log, tmp_path / "v2", TrainConfig(epochs=1, lr=3e-6, log_every=100, init=str(tmp_path / "v1" / "model.pt")))
    prev = torch.load(tmp_path / "v1" / "model.pt")["jepa"]; new = torch.load(tmp_path / "v2" / "model.pt")["jepa"]
    p = encode_patch(prev, new)
    rec = apply_patch(prev, p)
    for k in new:
        assert torch.equal(rec[k].view(torch.int16), new[k].view(torch.int16))
    assert p.sparsity > 0.3
    (tmp_path / "patches").mkdir(); p.save(str(tmp_path / "patches" / "0001.nrvp"))
    rt = NerveRuntime(sp, tmp_path / "v1" / "model.pt", tmp_path / "patches")
    v2 = NerveEnergy.load(sp, str(tmp_path / "v2" / "model.pt"))
    for a, b in zip(rt.model.jepa.state_dict().values(), v2.jepa.state_dict().values()):
        assert torch.equal(a.to(torch.bfloat16).view(torch.int16), b.to(torch.bfloat16).view(torch.int16))
    out = rt.step({"count": 3, "armed": True}, ("inc", {}))
    assert out["state"] == {"count": 4, "armed": True} and out["violations"] == []


def test_patch_rejects_tampering(tmp_path):
    import pytest
    a = {"w": torch.randn(64)}; b = {"w": a["w"] + 0.1}
    p = encode_patch(a, b)
    bad = bytearray(p.data); bad[-1] ^= 0xFF
    with pytest.raises(Exception):
        apply_patch(a, bytes(bad))
