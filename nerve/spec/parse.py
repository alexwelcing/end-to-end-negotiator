"""Parse ``.nerve`` behavior specs (YAML surface syntax)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import Action, Ensures, FieldType, Future, Hole, Spec, Transition


class SpecError(ValueError):
    pass


class _Loader(yaml.SafeLoader):
    """SafeLoader that treats only true/false as booleans (not on/off/yes/no)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    __import__("re").compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)
# drop the stock bool resolver (which also eats on/off/yes/no)
for ch in list("yYnNoO"):
    _Loader.yaml_implicit_resolvers.pop(ch, None)


def _field(name: str, d: Any) -> FieldType:
    if isinstance(d, str):
        d = {"type": d}
    kind = d.get("type")
    if kind == "bool":
        return FieldType(name, "bool")
    if kind == "enum":
        vals = d.get("values")
        if not vals:
            raise SpecError(f"enum field {name} needs values")
        return FieldType(name, "enum", values=tuple(str(v) for v in vals))
    if kind == "int":
        rng = d.get("range")
        lo, hi = (rng if rng else (None, None))
        return FieldType(name, "int", lo=lo, hi=hi)
    if kind == "float":
        rng = d.get("range") or (None, None)
        return FieldType(name, "float", lo=rng[0], hi=rng[1], step=d.get("step"))
    raise SpecError(f"field {name}: unknown type {kind!r}")


def parse_spec(doc: dict[str, Any]) -> Spec:
    if doc.get("nerve") != 1:
        raise SpecError("spec must declare `nerve: 1`")
    name = doc.get("name") or "unnamed"
    fields = {k: _field(k, v) for k, v in (doc.get("state") or {}).items()}
    if not fields:
        raise SpecError("spec has no state")

    actions: dict[str, Action] = {}
    for aname, ad in (doc.get("actions") or {}).items():
        ad = ad or {}
        params = {pk: _field(pk, pv) for pk, pv in (ad.get("params") or {}).items()}
        actions[aname] = Action(aname, params)

    transitions: dict[str, Transition] = {}
    for aname, td in (doc.get("transitions") or {}).items():
        if aname not in actions:
            raise SpecError(f"transition for unknown action {aname!r}")
        td = dict(td or {})
        when = td.pop("when", None)
        for k in td:
            if k not in fields:
                raise SpecError(f"transition {aname} updates unknown field {k!r}")
        transitions[aname] = Transition(aname, when, {k: str(v) for k, v in td.items()})

    holes: dict[str, Hole] = {}
    for hd in doc.get("learn") or []:
        h = Hole(hd["name"], hd["action"], {k: str(v) for k, v in (hd.get("fallback") or {}).items()},
                 list(hd.get("examples") or []))
        if h.action not in actions:
            raise SpecError(f"learn hole {h.name} refers to unknown action {h.action!r}")
        act = actions[h.action]
        for i, ex in enumerate(h.examples):
            if "state" not in ex or "next" not in ex:
                raise SpecError(f"learn hole {h.name} example {i} needs `state` and `next`")
            if act.params:
                missing = [pk for pk in act.params if pk not in (ex.get("params") or {})]
                if missing:
                    raise SpecError(f"learn hole {h.name} example {i}: action {h.action!r} takes params {list(act.params)}; "
                                    f"give them as `params: {{...}}` (missing {missing})")
                for pk, pv in ex["params"].items():
                    if pk not in act.params or not act.params[pk].contains(pv):
                        raise SpecError(f"learn hole {h.name} example {i}: param {pk}={pv!r} is not in the action's type")
            for k, f in fields.items():
                for part in ("state", "next"):
                    if k not in ex[part] or not f.contains(ex[part][k]):
                        raise SpecError(f"learn hole {h.name} example {i}: {part}.{k}={ex[part].get(k)!r} is not in type")
        holes[h.name] = h
        if h.action not in transitions:
            # the fallback is the spec-visible transition; the hole is learned on top
            transitions[h.action] = Transition(h.action, None, h.fallback)

    for aname in actions:
        if aname not in transitions:
            transitions[aname] = Transition(aname, None, {})   # no-op

    ensures = [Ensures(e.get("after"), str(e["holds"])) for e in doc.get("ensures") or []]
    futures: dict[str, Future] = {}
    for fname, fd in (doc.get("futures") or {}).items():
        futures[fname] = Future(fname, fd.get("kind", "closest"), fd["over"], fd.get("values"))

    initial = dict(doc.get("initial") or {})
    for k, f in fields.items():
        if k not in initial:
            initial[k] = f.domain()[0] if f.finite or f.step is not None else (f.lo or 0)
        if not f.contains(initial[k]):
            raise SpecError(f"initial value for {k} is out of type: {initial[k]!r}")

    return Spec(name=name, fields=fields, actions=actions, transitions=transitions,
                invariants=[str(i) for i in doc.get("invariants") or []], ensures=ensures,
                futures=futures, holes=holes, initial=initial, render=dict(doc.get("render") or {}))


def load_spec(path: str | Path) -> Spec:
    with open(path) as fh:
        return parse_spec(yaml.load(fh, Loader=_Loader))
