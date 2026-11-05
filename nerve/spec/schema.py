"""Behavior spec schema.

The spec is the *only* source of truth for an interface. It is deliberately
close to three things we borrow from:

* ink: typed state, pure transitions, ``ensures`` guards, ``learn`` holes
  (behavior the spec cannot write down and must be learned from examples).
* dragology: ``futures`` — for a gesture, the set of states it may lead to.
  Energy-based inference scores exactly those candidates.
* Lean-style properties: ``invariants`` and ``ensures`` are the checks a
  trained network must satisfy on every held-out transition.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator

from .expr import compile_expr


@dataclass(frozen=True)
class FieldType:
    """A finite or scalar field type."""
    name: str
    kind: str                       # "int" | "enum" | "bool" | "float"
    lo: int | float | None = None
    hi: int | float | None = None
    values: tuple[str, ...] = ()
    step: int | float | None = None

    @property
    def finite(self) -> bool:
        return self.kind in ("int", "enum", "bool") and (self.kind != "int" or (self.lo is not None and self.hi is not None))

    def domain(self) -> list[Any]:
        if self.kind == "bool":
            return [False, True]
        if self.kind == "enum":
            return list(self.values)
        if self.kind == "int":
            if not self.finite:
                raise ValueError(f"int field {self.name} has no range")
            return list(range(int(self.lo), int(self.hi) + 1))
        if self.kind == "float":
            if self.step is None or self.lo is None or self.hi is None:
                raise ValueError(f"float field {self.name} needs lo/hi/step to enumerate")
            n = int(round((self.hi - self.lo) / self.step))
            return [self.lo + i * self.step for i in range(n + 1)]
        raise ValueError(self.kind)

    def contains(self, v: Any) -> bool:
        if self.kind == "bool":
            return isinstance(v, bool)
        if self.kind == "enum":
            return v in self.values
        if self.kind == "int":
            return isinstance(v, int) and not isinstance(v, bool) and (self.lo is None or v >= self.lo) and (self.hi is None or v <= self.hi)
        if self.kind == "float":
            return isinstance(v, (int, float)) and (self.lo is None or v >= self.lo) and (self.hi is None or v <= self.hi)
        return False

    @property
    def width(self) -> int:
        """Feature width when encoded for the network."""
        if self.kind == "float":
            return 1
        return len(self.domain())


@dataclass(frozen=True)
class Action:
    name: str
    params: dict[str, FieldType] = field(default_factory=dict)

    def instances(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Enumerate concrete (name, params) instances."""
        if not self.params:
            yield (self.name, {})
            return
        keys = list(self.params)
        for combo in itertools.product(*(self.params[k].domain() for k in keys)):
            yield (self.name, dict(zip(keys, combo)))


@dataclass
class Transition:
    action: str
    when: str | None                # guard; if false the state is unchanged
    updates: dict[str, str]         # field -> expression over state + params + old
    _when: Any = field(default=None, repr=False)
    _updates: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._when = compile_expr(self.when) if self.when else None
        self._updates = {k: compile_expr(v) for k, v in self.updates.items()}


@dataclass
class Future:
    """dragology-style declaration: a gesture leads to one of these states."""
    name: str
    kind: str                       # "closest" | "between" | "vary"
    field_name: str
    values: list[Any] | None = None


@dataclass
class Ensures:
    after: str | None               # action name, or None for all
    holds: str
    _holds: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._holds = compile_expr(self.holds)


@dataclass
class Hole:
    """ink ``learn`` hole: behavior given only by examples / a fallback."""
    name: str
    action: str
    fallback: dict[str, str]        # the guaranteed-safe default transition
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Spec:
    name: str
    fields: dict[str, FieldType]
    actions: dict[str, Action]
    transitions: dict[str, Transition]
    invariants: list[str]
    ensures: list[Ensures]
    futures: dict[str, Future]
    holes: dict[str, Hole]
    initial: dict[str, Any]
    render: dict[str, Any] = field(default_factory=dict)
    _invariants: list[Any] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._invariants = [compile_expr(s) for s in self.invariants]

    @property
    def finite(self) -> bool:
        return all(f.finite or (f.kind == "float" and f.step is not None) for f in self.fields.values())

    def state_space(self) -> list[dict[str, Any]]:
        keys = list(self.fields)
        return [dict(zip(keys, c)) for c in itertools.product(*(self.fields[k].domain() for k in keys))]

    def state_space_size(self) -> int:
        n = 1
        for f in self.fields.values():
            n *= len(f.domain())
        return n

    def action_instances(self) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        for a in self.actions.values():
            out.extend(a.instances())
        return out

    def state_width(self) -> int:
        return sum(f.width for f in self.fields.values())

    def action_width(self) -> int:
        return len(self.action_instances())

    def state_key(self, s: dict[str, Any]) -> tuple:
        return tuple(s[k] for k in self.fields)

    def action_key(self, a: tuple[str, dict[str, Any]]) -> tuple:
        name, params = a
        return (name, tuple(sorted(params.items())))
