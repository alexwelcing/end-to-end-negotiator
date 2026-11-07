"""The compiled circuit decides every explicit transition no matter what the learned term says.

Theorem (paper, Thm. 1): the circuit's energy is the Hamming distance to the true successor over the number of
fields n, so 0 on the true successor and >= 1/n elsewhere. Corollary: if the circuit's weight exceeds
n * (spread of the learned energies) the argmin is the oracle's answer. `NerveEnergy.circuit_weight` raises the
weight per query to guarantee that; this test checks it with untrained (random) and with deliberately hostile
learned weights, and that hole actions are still left to the learned term.
"""
import pytest
import torch

from nerve.model import NerveEnergy
from nerve.model.jepa import JEPAConfig
from nerve.spec import Oracle, load_spec
from nerve.spec.encode import Encoder


@pytest.mark.parametrize("name", ["counter", "negotiate", "slider"])
def test_random_weights_are_still_exact_where_the_spec_is_explicit(name):
    sp = load_spec(f"specs/{name}.nerve"); enc = Encoder(sp); o = Oracle(sp)
    torch.manual_seed(123)
    m = NerveEnergy.build(sp, compile=True)                      # fresh, untrained JEPA + factored head
    with torch.no_grad():                                        # make the learned term loud and wrong
        for p in m.jepa.parameters():
            p.mul_(50.0)
    holes = {h.action for h in sp.holes.values()}
    for s in enc.all_states():
        for a in enc.all_actions():
            if a[0] in holes:
                continue
            assert m.step(s, a) == o.step(s, a), (name, s, a)


def test_hole_actions_are_left_to_the_learned_term():
    sp = load_spec("specs/counter.nerve"); enc = Encoder(sp)
    m = NerveEnergy.build(sp, compile=True)
    s = {"count": 0, "armed": True}
    E_full = m.energies(s, ("nudge", {}))
    with torch.no_grad():
        E_learned = m.jepa.energy_all(torch.from_numpy(enc.state(s))[None], torch.tensor([enc.action_id(("nudge", {}))]),
                                      m._cand, None, m._cur_idx(torch.from_numpy(enc.state(s))[None]), m._cand_idx)[0]
    # the circuit is silent on `nudge`: it adds the same constant to every candidate
    diff = (E_full - E_learned)
    assert float(diff.max() - diff.min()) < 1e-3
    assert int(E_full.argmin()) == int(E_learned.argmin())
