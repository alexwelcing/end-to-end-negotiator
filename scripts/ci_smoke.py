"""Smoke-test a running lab: the served networks load and answer, and the machine room ticks the chain.

usage: python scripts/ci_smoke.py http://127.0.0.1:8787

Standard library only, so CI and a fresh clone can run it against `nerve serve` with nothing installed.
Exits non-zero on the first failed check.
"""
import json
import sys
import urllib.request

TICKS = 20


def get(base: str, path: str) -> object:
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.load(r)


def post(base: str, path: str, body: object) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    if isinstance(out, dict) and out.get("error"):
        raise SystemExit(f"{path}: {out['error']}")
    return out


def check_specs(base: str) -> None:
    specs = get(base, "/api/specs")
    served = [s["name"] for s in specs if s["model"]]
    print(f"{len(specs)} interfaces, {len(served)} with a network: {', '.join(served)}")
    assert served, "no model loaded"
    r = post(base, "/api/hanoi/step", {"state": {"d1": 0, "d2": 0, "d3": 0, "d4": 0}, "action": "move", "params": {"src": 0, "dst": 2}})
    print("hanoi move(0, 2):", r["verdict"], r["network"])
    assert r["verdict"] == "agrees", r


def check_machines(base: str) -> None:
    machines = {m["name"]: m for m in get(base, "/api/machines")}
    assert "chain" in machines, f"no chain machine among {sorted(machines)}"
    chain = machines["chain"]
    print(", ".join(f"{b['instance']}:{b['type']}" + ("" if b["network"] else " (spec only)") for b in chain["blocks"]))
    assert all(b["network"] for b in chain["blocks"]), "a chain block has no network"
    state = chain["initial"]
    advance = [{"block": "clock", "action": "advance"}]
    fired: set[str] = set()
    for i in range(TICKS):
        t = post(base, "/api/machine/chain/tick", {"state": state, "external": advance})
        assert t["agrees"], f"tick {i + 1}: the networks disagree with the composed oracle: {t}"
        assert not t["violations"], f"tick {i + 1}: {t['violations']}"
        fired.update(f"{s['via']['block']}.{s['via']['event']}" for s in t["steps"] if s["via"])
        if i == 0:
            print("tick 1:", " -> ".join(f"{s['block']}.{s['action']}" for s in t["steps"]), f"agrees, {t['ms']:.2f} ms")
        state = t["state"]
    print(f"{TICKS} ticks agree with the composed oracle; wires fired: {', '.join(sorted(fired))}")
    assert fired, "no wire fired"


def main() -> None:
    base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8787"
    check_specs(base)
    check_machines(base)
    print("smoke ok")


if __name__ == "__main__":
    main()
