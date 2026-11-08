"""Write specs/fifteen.nerve (usage: python scripts/gen_fifteen_spec.py specs/fifteen.nerve).

Dragology's `fifteen` demo (src/demos/fifteen.tsx), the sliding-tile puzzle, on a 3 x 2 board instead of
4 x 4: five tiles and the blank. A drag swaps the dragged piece with an orthogonal neighbour when one of the
two is the blank; as in Dragology the blank itself is draggable, onto any tile next to it.

Every piece needs a field a gesture can be indexed by, but six cells in [0, 5] are 6^6 = 46,656 tuples. So
tile 5 has no field: it is on the one cell no other piece is on, `15 - t1 - t2 - t3 - t4 - blank` (the cells
sum to 15). Its gesture is indexed by `blank` instead, which a slide of tile 5 always changes. That leaves
6^5 = 7,776 tuples, of which the 720 with five distinct cells are boards.
"""
import sys
from collections import deque
from itertools import permutations, product
from pathlib import Path

W, H = 3, 2
CELLS = W * H
TILES = range(1, CELLS)                       # tile CELLS - 1 is the derived one
LAST = CELLS - 1
FIELD = {k: f"t{k}" for k in TILES if k != LAST}
FIELDS = [*FIELD.values(), "blank"]
CELL_SUM = sum(range(CELLS))
SIZE, ORIGIN = 60, 50                         # render: px per cell, centre of cell 0
# Dragology's start has the blank bottom left; this board is four slides from solved (1 2 3 / 4 5 _).
INITIAL = {"t1": 0, "t2": 2, "t3": 5, "t4": 4, "blank": 3}


def cell(k: int, old: bool = False) -> str:
    """Expression for the cell of tile `k` (0 = the blank)."""
    pre = "old." if old else ""
    if k == 0:
        return f"{pre}blank"
    if k in FIELD:
        return f"{pre}{FIELD[k]}"
    return f"({CELL_SUM} - " + " - ".join(f"{pre}{f}" for f in FIELDS) + ")"


def adjacent(a: str, b: str) -> str:
    """Cells `a` and `b` share an edge."""
    return f"abs({a} // {W} - {b} // {W}) + abs({a} % {W} - {b} % {W}) == 1"


def transitions() -> list[str]:
    out = []
    for k in TILES:
        out += [f"  slide_{k}:", f"    when: \"{adjacent(cell(k), 'blank')}\""]
        if k in FIELD:
            out.append(f"    {FIELD[k]}: \"blank\"")
        out.append(f"    blank: \"{cell(k)}\"")
    out += ["  slide_blank:", f"    when: \"{adjacent('blank', 'to')}\"", "    blank: \"to\""]
    out += [f"    {f}: \"blank if {f} == to else {f}\"" for f in FIELD.values()]
    return out


def ensures() -> list[str]:
    one_tile = " + ".join(f"int({cell(k)} != {cell(k, old=True)})" for k in TILES)
    out = [
        f"  - {{ holds: \"abs(blank // {W} - old.blank // {W}) + abs(blank % {W} - old.blank % {W}) <= 1\" }}"
        "   # one slide at a time: the blank moves to a neighbouring cell or stays",
        f"  - {{ holds: \"{one_tile} <= 1\" }}   # and at most one tile moves, onto the blank's old cell",
    ]
    for k in TILES:
        t, ot = cell(k), cell(k, old=True)
        out.append(f"  - {{ after: slide_{k}, holds: \"({t} == old.blank and blank == {ot}) or ({t} == {ot} and blank == old.blank)\" }}"
                   "   # the tile trades places with the blank, or nothing moves")
    out.append("  - { after: slide_blank, holds: \"blank == to or blank == old.blank\" }   # the blank lands where it was dropped, or stays")
    return out


def render(name: str, c: str) -> str:
    return f"  {name}: {{ x: \"{ORIGIN} + {SIZE} * ({c} % {W})\", y: \"{ORIGIN} + {SIZE} * ({c} // {W})\" }}"


def futures() -> list[str]:
    out = ["  # one gesture per piece, each moving only that piece: a tile goes to the blank if it is next to it"]
    for k in TILES:
        over = FIELD.get(k, "blank")
        out.append(f"  drag_{k}: {{ kind: closest, over: {over}, grab: tile{k}, action: slide_{k}, stickiness: 6, floating: true }}"
                   + ("   # tile 5 has no field; sliding it always moves the blank" if k not in FIELD else ""))
    out += ["  # Dragology's \"weird experiment\": the blank is draggable too, onto any tile next to it",
            "  drag_blank: { kind: closest, over: blank, grab: blank, action: slide_blank, stickiness: 6, floating: true }"]
    return out


# ---- the same rules, independently, for the header and the sanity checks ----------------------------

def neighbours(c: int):
    x, y = c % W, c // W
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        if 0 <= x + dx < W and 0 <= y + dy < H:
            yield (y + dy) * W + x + dx


def board_stats() -> tuple[int, int, int]:
    """(tuples, boards, boards reachable from INITIAL)."""
    tuples = CELLS ** len(FIELDS)
    boards = sum(1 for t in product(range(CELLS), repeat=len(FIELDS)) if len(set(t)) == len(FIELDS))
    assert boards == len(list(permutations(range(CELLS), len(FIELDS))))
    start = tuple(INITIAL[f] for f in FIELDS)              # cells of t1..t4, blank
    seen, queue = {start}, deque([start])
    while queue:
        s = queue.popleft()
        blank = s[-1]
        for n in neighbours(blank):
            nxt = tuple(blank if c == n else c for c in s[:-1]) + (n,)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    solved = tuple(range(LAST - 1)) + (CELLS - 1,)
    assert solved in seen, "the initial board must be solvable"
    return tuples, boards, len(seen)


def spec() -> str:
    tuples, boards, reachable = board_stats()
    distinct = " and ".join(f"{a} != {b}" for i, a in enumerate(FIELDS) for b in FIELDS[i + 1:])
    lines = [
        "# Dragology's `fifteen` (src/demos/fifteen.tsx), the sliding-tile puzzle: drag a piece and it trades",
        "# places with an orthogonal neighbour when one of the two is the blank, and the blank is draggable too.",
        f"# Scaled down from 4 x 4 to {W} x {H}: tiles 1-{LAST} and the blank, cells numbered row by row from 0.",
        f"# Tile {LAST} has no field (it is on the cell no other piece is on), so the {tuples:,} tuples hold {boards}",
        f"# boards; {reachable} are reachable from any one of them, as in the real puzzle. Dragology chains snaps so",
        "# one gesture can make several slides; here a gesture makes one slide.",
        "# Generated by scripts/gen_fifteen_spec.py; edit the script, not this file.",
        "nerve: 1",
        "name: fifteen",
        "state:",
        *[f"  {f}: {{type: int, range: [0, {CELLS - 1}]}}   # cell of tile {k}" for k, f in FIELD.items()],
        f"  blank: {{type: int, range: [0, {CELLS - 1}]}}   # cell of the blank",
        "actions:",
        *[f"  slide_{k}: {{}}   # tile {k} slides into the blank, if it is next to it" for k in TILES],
        f"  slide_blank: {{params: {{to: {{type: int, range: [0, {CELLS - 1}]}}}}}}   # the blank trades places with the tile at `to`",
        "transitions:",
        *transitions(),
        "invariants:",
        f"  - \"{distinct}\"   # no two pieces share a cell (tile {LAST} gets the one left over)",
        "ensures:",
        *ensures(),
        "render:",
        *[render(f"tile{k}", cell(k)) for k in TILES],
        render("blank", "blank"),
        "futures:",
        *futures(),
        "initial: {" + ", ".join(f"{f}: {INITIAL[f]}" for f in FIELDS) + "}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(spec(), encoding="utf-8")
    tuples, boards, reachable = board_stats()
    print(f"wrote {sys.argv[1]}: {tuples} tuples, {boards} boards, {reachable} reachable from the initial board")
