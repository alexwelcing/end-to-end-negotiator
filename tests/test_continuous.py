"""Continuous fields: inference by gradient descent on the energy, evaluated within tolerance."""
from nerve.spec import load_spec, Oracle
from nerve.spec.encode import Encoder
from nerve.data import EpisodeLog, rollout_dataset
from nerve.train import train, TrainConfig
from nerve.train.eval import close


def test_continuous_slider_small(tmp_path):
    sp = load_spec("specs/slider_continuous.nerve")
    assert not sp.finite and sp.continuous_fields() == ["value"]
    enc = Encoder(sp)
    assert enc.action_cont_dim == 1 and enc.disc_dim == 2          # grabbed one-hot; `to` is a continuous param
    log = EpisodeLog(tmp_path / "log.jsonl")
    rollout_dataset(sp, log, episodes=60, horizon=10, negatives=4, seed=0)
    r = train(sp, log, tmp_path / "run", TrainConfig(epochs=25, log_every=100, d=64, hidden=128, lr=3e-3))
    assert r["continuous"] and r["all"]["n"] == 1000
    assert r["all"]["exact"] > 0.4            # a short CPU run; the GPU run (200 epochs, d=128) reaches 0.998
    assert r["all"]["mean_abs_err_frac"] < 0.1


def test_close_tolerance():
    sp = load_spec("specs/slider_continuous.nerve")
    assert close(sp, {"value": 137.19, "grabbed": True}, {"value": 137.2, "grabbed": True})
    assert not close(sp, {"value": 137.19, "grabbed": False}, {"value": 137.2, "grabbed": True})
    assert not close(sp, {"value": 150.0, "grabbed": True}, {"value": 137.2, "grabbed": True})


def test_tolerant_ensures():
    sp = load_spec("specs/slider_continuous.nerve"); o = Oracle(sp)
    s = {"value": 10.0, "grabbed": True}; a = ("move", {"to": 137.2})
    assert o.violations(s, a, {"value": 137.19, "grabbed": True}) == []
    assert o.violations(s, a, {"value": 150.0, "grabbed": True})
