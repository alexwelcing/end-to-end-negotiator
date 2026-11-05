import pytest
from nerve.spec import load_spec, Oracle
from nerve.spec.expr import compile_expr, eval_expr, ExprError
from nerve.spec.circuit import compile_circuit
from nerve.spec.encode import Encoder
import torch

SPECS = ["toggle", "counter", "slider", "negotiate"]


def test_expr_safe():
    assert eval_expr(compile_expr("min(a + 1, 9)"), {"a": 9}) == 9
    assert eval_expr(compile_expr("0 <= x <= 2 and y in ['a']"), {"x": 1, "y": "a"})
    assert eval_expr(compile_expr("old.c + 1"), {"c": 5}, old={"c": 1}) == 2
    with pytest.raises(ExprError):
        compile_expr("__import__('os')")
    with pytest.raises(ExprError):
        compile_expr("x.__class__")


@pytest.mark.parametrize("name", SPECS)
def test_specs_self_check(name):
    sp = load_spec(f"specs/{name}.nerve")
    assert Oracle(sp).self_check() == []


@pytest.mark.parametrize("name", SPECS)
def test_compiled_circuit_is_exact(name):
    sp = load_spec(f"specs/{name}.nerve"); enc = Encoder(sp); o = Oracle(sp)
    c = compile_circuit(sp, exclude_actions={h.action for h in sp.holes.values()})
    S = torch.from_numpy(enc.all_state_vectors())
    for s in enc.all_states():
        for a in enc.all_actions():
            if a[0] in {h.action for h in sp.holes.values()}:
                continue
            E = c.energy_all(torch.from_numpy(enc.state(s))[None], torch.from_numpy(enc.action(a))[None], S)[0]
            assert enc.all_states()[int(E.argmin())] == o.step(s, a)
            assert abs(float(E.min())) < 1e-6


def test_yaml_on_off_are_strings():
    sp = load_spec("specs/toggle.nerve")
    assert sp.fields["status"].values == ("off", "on")
