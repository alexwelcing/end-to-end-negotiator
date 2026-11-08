"""Write specs/sokoban.nerve (usage: python scripts/gen_sokoban_spec.py specs/sokoban.nerve).

Dragology's `sokoban` demo: drag the player one cell left, right, up or down onto floor; a box in the way is
pushed one cell further, but only onto floor that has no box on it. Only the player is draggable (the demo's
"make level editable" switch, which lets walls, boxes and goals be dragged too, is left out).

Scaled down: Dragology's level is 8 x 9 with seven boxes, far too many states to enumerate. This spec keeps the
top three rows of that level (its first two boxes and two goals, player where Dragology puts it) and walls them
off below. The level is the `LEVEL` string, in Dragology's own notation (`#` wall, `p` player, `b` box, `g` goal,
`B` box on a goal); everything else (the cell grid, the guards, the render) is derived from it here.

Each object is one int field: its cell `y * W + x` on the grid of the level's interior (the walls' bounding box
minus its rim). Walls are fixed, so they are not state: an invariant keeps every object off them. `move(dir)`
is the drag's only action, and the guard is Dragology's rule. There is no win condition in the demo, so goals
are drawn nowhere and appear only in the header comment.
"""
import sys
from pathlib import Path

LEVEL = """\
#######
###   #
#gpb  #
### bg#
#######"""

# Dragology's moves, as (name, dx, dy), in the order its drag lists them.
DIRS = [("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1)]


def parse(level: str):
    rows = level.split("\n")
    cells = {(x, y): ch for y, row in enumerate(rows) for x, ch in enumerate(row)}
    assert all(ch in "#pbgB " for ch in cells.values()), "unknown level character"
    inner = [xy for xy, ch in cells.items() if ch != "#"]
    x0, y0 = min(x for x, _ in inner), min(y for _, y in inner)
    w, h = max(x for x, _ in inner) - x0 + 1, max(y for _, y in inner) - y0 + 1
    # The level must be closed: every interior cell's neighbours exist, so a step off the grid is a step into a wall.
    for (x, y) in inner:
        for _, dx, dy in DIRS:
            assert (x + dx, y + dy) in cells, f"level is open at {(x + dx, y + dy)}"

    def idx(x: int, y: int) -> int:
        return (y - y0) * w + (x - x0)

    walls = sorted(idx(x, y) for y in range(y0, y0 + h) for x in range(x0, x0 + w) if cells.get((x, y), "#") == "#")
    player = [idx(*xy) for xy, ch in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])) if ch == "p"]
    boxes = [idx(*xy) for xy, ch in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])) if ch in "bB"]
    goals = [idx(*xy) for xy, ch in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])) if ch in "gB"]
    assert len(player) == 1 and len(boxes) == len(goals) >= 1
    return w, h, walls, player[0], boxes, goals


W, H, WALLS, PLAYER, BOXES, GOALS = parse(LEVEL)
N = W * H
FLOOR = [c for c in range(N) if c not in WALLS]
BOX = [f"b{i}" for i in range(1, len(BOXES) + 1)]
OBJECTS = ["p", *BOX]
TILE = 60                           # px between cell centres, so drawn objects never touch
ORIGIN = 50


def offset(dx: int, dy: int) -> int:
    return dx + W * dy


def stays_on_grid(dx: int, dy: int, k: int) -> str:
    """`p` can take `k` steps in direction (dx, dy) without leaving the grid."""
    if dx < 0:
        return f"p % {W} >= {k}"
    if dx > 0:
        return f"p % {W} <= {W - 1 - k}"
    if dy < 0:
        return f"p >= {W * k}"
    return f"p <= {N - 1 - W * k}"


def shifted(k: int) -> str:
    return "p" if k == 0 else f"p + {k}" if k > 0 else f"p - {-k}"


def floor_at(k: int) -> str:
    """Cell `p + k` is floor (not a wall); written over `p` so the guard reads without nested arithmetic.
    Walls that `p + k` could only hit from outside the grid are left out: the grid check rules those out."""
    return " and ".join(f"p != {w - k}" for w in WALLS if 0 <= w - k < N) or "True"


def box_at(cell: str) -> str:
    return "(" + " or ".join(f"{b} == {cell}" for b in BOX) + ")"


def no_box_at(cell: str) -> str:
    return " and ".join(f"{b} != {cell}" for b in BOX)


def guard(dx: int, dy: int) -> str:
    """Dragology: the next cell is floor, and either empty or holding a box whose next cell is empty floor."""
    s = offset(dx, dy)
    nxt, beyond = shifted(s), shifted(2 * s)
    push = f"{stays_on_grid(dx, dy, 2)} and {floor_at(2 * s)} and {no_box_at(beyond)}"
    return f"({stays_on_grid(dx, dy, 1)} and {floor_at(s)} and (not {box_at(nxt)} or ({push})))"


def by_dir(per_dir) -> str:
    """`dir`-indexed conditional over DIRS: `a if dir == 'left' else b if ... else d`."""
    parts = [f"{per_dir(dx, dy)} if dir == '{name}' else" for name, dx, dy in DIRS[:-1]]
    return " ".join(parts + [per_dir(*DIRS[-1][1:])])


def box_update(b: str) -> str:
    return by_dir(lambda dx, dy: f"({shifted(2 * offset(dx, dy))} if {b} == {shifted(offset(dx, dy))} else {b})")


def on_floor(f: str) -> str:
    return " and ".join(f"{f} != {w}" for w in WALLS)


def render(f: str) -> str:
    return f"{{ x: \"{ORIGIN} + {TILE} * ({f} % {W})\", y: \"{ORIGIN} + {TILE} * ({f} // {W})\" }}"


def picture() -> list[str]:
    """The interior grid as cell numbers, for the header comment."""
    out = []
    for y in range(H):
        row = []
        for x in range(W):
            c = y * W + x
            mark = "##" if c in WALLS else f"{c:2d}"
            row.append(mark)
        out.append("#   " + " ".join(row))
    return out


def counts() -> tuple[int, int]:
    from itertools import permutations
    valid = sum(1 for _ in permutations(FLOOR, len(OBJECTS)))
    return N ** len(OBJECTS), valid


def spec() -> str:
    tuples, valid = counts()
    step = by_dir(lambda dx, dy: f"{offset(dx, dy)}")
    distinct = " and ".join(f"{x} != {y}" for i, x in enumerate(OBJECTS) for y in OBJECTS[i + 1:])
    lines = [
        "# Dragology's sokoban: drag the player one cell onto floor; a box in the way is pushed one cell",
        "# further, but only onto floor with no box on it. Only the player is draggable.",
        "# Scaled down: Dragology's 8 x 9 level with seven boxes is too big to enumerate, so this is its top",
        "# three rows (first two boxes and goals, player where Dragology puts it), walled off below:",
        "#",
        *[f"#   {row}" for row in LEVEL.split("\n")],
        "#",
        f"# Each object is one cell number, y * {W} + x over the interior ({W} x {H}, `##` = wall):",
        "#",
        *picture(),
        "#",
        f"# Goals: cells {' and '.join(map(str, GOALS))} (the demo draws them and has no win condition, so they are",
        "# not state). Boxes b1, b2 are Dragology's box objects in reading order. One drag is one step:",
        "# Dragology's `chain: true` (keep walking within one drag) has no Nerve counterpart yet.",
        f"# {tuples:,} cell tuples, {valid:,} of them boards (every object on floor, none sharing a cell).",
        "# Generated by scripts/gen_sokoban_spec.py; edit the script, not this file.",
        "nerve: 1",
        "name: sokoban",
        "state:",
        f"  p: {{type: int, range: [0, {N - 1}]}}    # player's cell",
        *[f"  {b}: {{type: int, range: [0, {N - 1}]}}   # cell of box {b[1:]}" for b in BOX],
        "actions:",
        f"  move: {{params: {{dir: {{type: enum, values: [{', '.join(n for n, _, _ in DIRS)}]}}}}}}",
        "transitions:",
        "  move:",
        # Dragology offers only the directions this guard allows; any other `dir` is a no-op.
        f"    when: \"{by_dir(guard)}\"",
        f"    p: \"{by_dir(lambda dx, dy: shifted(offset(dx, dy)))}\"",
        *[f"    {b}: \"{box_update(b)}\"" for b in BOX],
        "invariants:",
        f"  - \"{' and '.join(on_floor(f) for f in OBJECTS)}\"   # nothing stands on a wall",
        f"  - \"{distinct}\"   # one object per cell",
        "ensures:",
        f"  - {{ after: move, holds: \"p == old.p or (p - old.p == ({step}) and (p // {W} == old.p // {W} or p % {W} == old.p % {W}))\" }}   # one step the dragged way, never wrapping a row",
        *[f"  - {{ after: move, holds: \"{b} == old.{b} or ({b} - old.{b} == p - old.p and p == old.{b})\" }}   # {b} moves only when pushed, the way the player went"
          for b in BOX],
        "render:",
        f"  player: {render('p')}",
        *[f"  box{b[1:]}: {render(b)}" for b in BOX],
        "futures:",
        "  # Dragology: closest over the four neighbouring moves (and staying); the guard decides which exist",
        "  drag_player: { kind: closest, over: p, grab: player, action: move, stickiness: 6 }",
        "initial: {" + ", ".join([f"p: {PLAYER}", *[f"{b}: {c}" for b, c in zip(BOX, BOXES)]]) + "}",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(spec(), encoding="utf-8")
    t, v = counts()
    print(f"wrote {sys.argv[1]}: {W} x {H} grid, {len(FLOOR)} floor cells, {t} tuples, {v} boards")
