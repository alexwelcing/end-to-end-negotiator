"""Write specs/rush_hour.nerve (usage: python scripts/gen_rush_hour_spec.py specs/rush_hour.nerve).

Dragology's `rush-hour` demo (src/demos/rush-hour.tsx): cars slide along their own row or column, any
number of cells, until the next cell is off the board or taken by another car. The red car alone may drive
up to two cells past the right edge, which is how it leaves. Scaled down from a 6 x 6 board with eight cars
to 5 x 5 with six: a car only ever moves along its line, so its state is one int (the column of a
horizontal car's left end, the row of a vertical car's top end), and the six cars make 4,608 tuples.

`slide_<car>(to)` moves that car to `to` when nothing is in the way: no other car may touch any cell the car
sweeps through between where it is and `to`, which is exactly the run of positions Dragology's `tryMove`
walks before it hits something. Each car has its own action, so dragging a car can only move that car.
`ensures` say the same of any transition, old state to new, so a guard that let a car jump another car,
or a prediction that did, is caught by the spec's own check.
"""
import sys
from collections import deque
from itertools import product
from pathlib import Path
from typing import NamedTuple

W, H = 5, 5
EXIT = 2                           # Dragology: the red car's right boundary is the board width + 2
SIZE, ORIGIN = 50, 30              # render: px per cell, top-left corner of the board


class Car(NamedTuple):
    name: str
    dir: str                       # "h" slides along a row, "v" along a column
    line: int                      # its row (h) or column (v)
    length: int
    start: int

    @property
    def last(self) -> int:
        """Largest position: the board edge, or past it for the red car."""
        span = (W + EXIT if self.name == "red" else W) if self.dir == "h" else H
        return span - self.length


CARS = (
    Car("red", "h", 2, 2, 0),      # the car to get out, on the middle row
    Car("a", "h", 0, 3, 1),        # a truck across the top: b can never go up past it
    Car("b", "v", 2, 2, 1),        # blocks red; the way out is down, once d moves
    Car("c", "v", 4, 2, 2),        # blocks red; the way out is up
    Car("d", "h", 4, 2, 1),        # d and e share the bottom row, so each blocks the other
    Car("e", "h", 4, 2, 3),
)


def covers(car: Car, cell: int) -> tuple[int, int] | None:
    """Positions of `car` at which it covers `cell` of its line, or None if it never does."""
    lo, hi = max(0, cell - car.length + 1), min(car.last, cell)
    return (lo, hi) if lo <= hi else None


def between(car: Car, span: tuple[int, int], pos: str | None = None) -> str | None:
    """Expression that `car` (at `pos`, by default its field) is in `span`; None when that is every position it has."""
    lo, hi = span
    pos = pos or car.name
    if (lo, hi) == (0, car.last):
        return None
    return f"{pos} == {lo}" if lo == hi else f"{lo} <= {pos} <= {hi}"


def both(*terms: str | None) -> str:
    return " and ".join(t for t in terms if t is not None) or "True"


def overlap(k: Car, o: Car) -> str | None:
    """Expression that cars `k` and `o` share a cell, or None if they never can."""
    if k.dir == o.dir:
        if k.line != o.line:
            return None
        return f"{k.name} < {o.name} + {o.length} and {o.name} < {k.name} + {k.length}"
    on_k, on_o = covers(k, o.line), covers(o, k.line)     # k on o's line, and o on k's line
    if on_k is None or on_o is None:
        return None
    return both(between(k, on_k), between(o, on_o))


def blocks(k: Car, o: Car, src: str | None = None, dst: str = "to", other: str | None = None) -> str | None:
    """Expression that `o` (at `other`) touches a cell `k` sweeps through going from `src` to `dst`.

    By default that is `o` in the way of `slide_<k>(to)`; `ensures` asks it of `old.<k>` to `<k>` instead.
    """
    src, other = src or k.name, other or o.name
    lo, hi = f"min({src}, {dst})", f"max({src}, {dst}) + {k.length - 1}"
    if k.dir == o.dir:
        if k.line != o.line:
            return None
        return f"{other} <= {hi} and {lo} <= {other} + {o.length - 1}"
    on_o = covers(o, k.line)
    if on_o is None:
        return None
    return both(f"{lo} <= {o.line} <= {hi}", between(o, on_o, other))


def never_passes(k: Car) -> str:
    """`ensures`: whatever moved `k`, no other car was on a cell between where it was and where it is."""
    terms = [f"not ({b})" for o in CARS if o != k and (b := blocks(k, o, f"old.{k.name}", k.name, f"old.{o.name}"))]
    return " and ".join(terms)


def transitions() -> list[str]:
    out = []
    for k in CARS:
        clear = [f"not ({b})" for o in CARS if o != k and (b := blocks(k, o))]
        out += [f"  slide_{k.name}:", f"    when: \"{' and '.join(clear)}\"", f"    {k.name}: \"to\""]
    return out


def render(car: Car) -> str:
    pos = f"{SIZE} * {car.name}"
    along, across = f"{ORIGIN + SIZE * car.length // 2} + {pos}", f"{ORIGIN + SIZE * car.line + SIZE // 2}"
    x, y = (along, across) if car.dir == "h" else (across, along)
    return f"  {car.name}: {{ x: \"{x}\", y: \"{y}\" }}"


def describe(car: Car) -> str:
    where = f"row {car.line}, left end" if car.dir == "h" else f"column {car.line}, top end"
    off = f"; past {W - car.length} it drives off the board, and at {W} it is out" if car.last > W - car.length else ""
    return f"{'horizontal' if car.dir == 'h' else 'vertical'}, length {car.length}: {where}{off}"


# ---- the same rules, independently, from cells, for the header and the sanity checks ---------------

def cells(car: Car, p: int) -> set[tuple[int, int]]:
    return {(p + i, car.line) if car.dir == "h" else (car.line, p + i) for i in range(car.length)}


def is_board(s: tuple[int, ...]) -> bool:
    taken = [cells(c, p) for c, p in zip(CARS, s)]
    return sum(map(len, taken)) == len(set().union(*taken))


def slides(s: tuple[int, ...]):
    """Dragology's tryMove for every car and both directions: every position up to the first block."""
    for i, car in enumerate(CARS):
        others = set().union(*(cells(c, p) for j, (c, p) in enumerate(zip(CARS, s)) if j != i))
        for step in (-1, 1):
            p = s[i] + step
            while 0 <= p <= car.last and not cells(car, p) & others:
                yield s[:i] + (p,) + s[i + 1:]
                p += step


def stats() -> tuple[int, int, int, int]:
    """(tuples, boards, legal moves over all boards, fewest drags to get the red car off the board)."""
    tuples = list(product(*(range(c.last + 1) for c in CARS)))
    boards = [s for s in tuples if is_board(s)]
    moves = sum(1 for s in boards for _ in slides(s))
    start = tuple(c.start for c in CARS)
    assert is_board(start)
    dist, queue = {start: 0}, deque([start])
    while queue:
        s = queue.popleft()
        if s[0] == W:
            return len(tuples), len(boards), moves, dist[s]
        for n in slides(s):
            if n not in dist:
                dist[n] = dist[s] + 1
                queue.append(n)
    raise AssertionError("the red car cannot get out")


def spec() -> str:
    n_tuples, n_boards, n_moves, solution = stats()
    pairs = [(k, o) for i, k in enumerate(CARS) for o in CARS[i + 1:]]
    apart = [f"  - \"not ({e})\"   # {k.name} and {o.name} never share a cell" for k, o in pairs if (e := overlap(k, o))]
    one_car = " + ".join(f"int({c.name} != old.{c.name})" for c in CARS)
    lines = [
        "# Dragology's `rush-hour` (src/demos/rush-hour.tsx): a car slides along its row or column, any number",
        "# of cells, until the next is off the board or taken; the red car alone may drive two cells past the",
        f"# right edge, which is how it gets out. Scaled down from 6 x 6 with eight cars to {W} x {H} with six: each",
        f"# car is one int, its position along its line, so there are {n_tuples:,} tuples, {n_boards} of them boards.",
        f"# From the start the red car is out in {solution} drags. Dragology slides a car smoothly between its",
        "# stops (`between`); here the drop snaps to the closest one.",
        "# Generated by scripts/gen_rush_hour_spec.py; edit the script, not this file.",
        "nerve: 1",
        "name: rush_hour",
        "state:",
        *[f"  {c.name}: {{type: int, range: [0, {c.last}]}}   # {describe(c)}" for c in CARS],
        "actions:",
        *[f"  {'slide_' + c.name}: {{params: {{to: {{type: int, range: [0, {c.last}]}}}}}}" for c in CARS],
        "transitions:",
        "  # a car goes to `to` when no other car touches a cell it would sweep through on the way",
        *transitions(),
        "invariants:",
        *apart,
        "ensures:",
        f"  - {{ holds: \"{one_car} <= 1\" }}   # one car moves at a time",
        *[f"  - {{ holds: \"{never_passes(c)}\" }}   # {c.name} never passes through another car" for c in CARS],
        *[f"  - {{ after: slide_{c.name}, holds: \"{c.name} == to or {c.name} == old.{c.name}\" }}   # it stops where it was dropped, or stays"
          for c in CARS],
        "render:",
        *[render(c) for c in CARS],
        "futures:",
        "  # one gesture per car: every stop it can slide to along its line, in either direction",
        *[f"  drag_{c.name}: {{ kind: closest, over: {c.name}, grab: {c.name}, action: slide_{c.name}, stickiness: 6, floating: true }}"
          for c in CARS],
        "initial: {" + ", ".join(f"{c.name}: {c.start}" for c in CARS) + "}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(spec(), encoding="utf-8")
    n_tuples, n_boards, n_moves, solution = stats()
    print(f"wrote {sys.argv[1]}: {n_tuples} tuples, {n_boards} boards, {n_moves} legal moves, red out in {solution} drags")
