import json
from nerve.evolve import Supervisor


def test_offline_supervisor_loop(tmp_path):
    sup = Supervisor("specs/toggle.nerve", tmp_path, budget_usd=1.0, iterations=2, mode="offline")
    out = sup.run()
    hist = [json.loads(l) for l in open(out["history"])]
    assert len(hist) == 2 and all("eval" in h for h in hist)
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "iter_001" / "team_config.json").exists()
