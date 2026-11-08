//! Dragology's `fifteen` demo (src/demos/fifteen.tsx), the sliding-tile puzzle, on a 3 x 2 board
//! instead of 4 x 4: five tiles and the blank. A drag swaps the dragged piece with an orthogonal
//! neighbour when one of the two is the blank; as in Dragology the blank itself is draggable, onto
//! any tile next to it. Every piece needs a field a gesture can be indexed by, but six cells in [0, 5]
//! are 6^6 = 46,656 tuples. So tile 5 has no field: it is on the one cell no other piece is on,
//! `15 - t1 - t2 - t3 - t4 - blank` (the cells sum to 15). Its gesture is indexed by `blank` instead,
//! which a slide of tile 5 always changes. That leaves 6^5 = 7,776 tuples, of which the 720 with five
//! distinct cells are boards.

use super::{commas, generated_by, l, text};
use std::collections::{HashSet, VecDeque};

const W: usize = 3;
const H: usize = 2;
const CELLS: usize = W * H;
const LAST: usize = CELLS - 1;
const CELL_SUM: usize = 15;
const SIZE: usize = 60;
const ORIGIN: usize = 50;
/// Cells of t1..t4 and the blank: four slides from solved (1 2 3 / 4 5 _).
const INITIAL: [usize; 5] = [0, 2, 5, 4, 3];

fn tiles() -> std::ops::Range<usize> {
    1..CELLS
}

fn has_field(k: usize) -> bool {
    k != LAST
}

fn fields() -> Vec<String> {
    let mut v: Vec<String> = tiles().filter(|&k| has_field(k)).map(|k| format!("t{k}")).collect();
    v.push(l("blank"));
    v
}

/// Expression for the cell of tile `k` (0 = the blank).
fn cell(k: usize, old: bool) -> String {
    let pre = if old { "old." } else { "" };
    if k == 0 {
        return format!("{pre}blank");
    }
    if has_field(k) {
        return format!("{pre}t{k}");
    }
    format!("({CELL_SUM} - {})", fields().iter().map(|f| format!("{pre}{f}")).collect::<Vec<_>>().join(" - "))
}

/// Cells `a` and `b` share an edge.
fn adjacent(a: &str, b: &str) -> String {
    format!("abs({a} // {W} - {b} // {W}) + abs({a} % {W} - {b} % {W}) == 1")
}

fn transitions() -> Vec<String> {
    let mut out = Vec::new();
    for k in tiles() {
        out.push(format!("  slide_{k}:"));
        out.push(format!("    when: \"{}\"", adjacent(&cell(k, false), "blank")));
        if has_field(k) {
            out.push(format!("    t{k}: \"blank\""));
        }
        out.push(format!("    blank: \"{}\"", cell(k, false)));
    }
    out.push(l("  slide_blank:"));
    out.push(format!("    when: \"{}\"", adjacent("blank", "to")));
    out.push(l("    blank: \"to\""));
    out.extend(tiles().filter(|&k| has_field(k)).map(|k| format!("    t{k}: \"blank if t{k} == to else t{k}\"")));
    out
}

fn ensures() -> Vec<String> {
    let one_tile: Vec<String> = tiles().map(|k| format!("int({} != {})", cell(k, false), cell(k, true))).collect();
    let mut out = vec![
        format!("  - {{ holds: \"abs(blank // {W} - old.blank // {W}) + abs(blank % {W} - old.blank % {W}) <= 1\" }}   # one slide at a time: the blank moves to a neighbouring cell or stays"),
        format!("  - {{ holds: \"{} <= 1\" }}   # and at most one tile moves, onto the blank's old cell", one_tile.join(" + ")),
    ];
    for k in tiles() {
        let (t, ot) = (cell(k, false), cell(k, true));
        out.push(format!("  - {{ after: slide_{k}, holds: \"({t} == old.blank and blank == {ot}) or ({t} == {ot} and blank == old.blank)\" }}   # the tile trades places with the blank, or nothing moves"));
    }
    out.push(l("  - { after: slide_blank, holds: \"blank == to or blank == old.blank\" }   # the blank lands where it was dropped, or stays"));
    out
}

fn position(c: &str) -> String {
    format!("x: \"{ORIGIN} + {SIZE} * ({c} % {W})\", y: \"{ORIGIN} + {SIZE} * ({c} // {W})\"")
}

fn futures() -> Vec<String> {
    let mut out = vec![l("  # one gesture per piece, each moving only that piece: a tile goes to the blank if it is next to it")];
    for k in tiles() {
        let over = if has_field(k) { format!("t{k}") } else { l("blank") };
        let note = if has_field(k) { "" } else { "   # tile 5 has no field; sliding it always moves the blank" };
        out.push(format!("  drag_{k}: {{ kind: closest, over: {over}, grab: tile{k}, action: slide_{k}, stickiness: 6, floating: true }}{note}"));
    }
    out.push(l("  # Dragology's \"weird experiment\": the blank is draggable too, onto any tile next to it"));
    out.push(l("  drag_blank: { kind: closest, over: blank, grab: blank, action: slide_blank, stickiness: 6, floating: true }"));
    out
}

fn neighbours(c: usize) -> Vec<usize> {
    let (x, y) = ((c % W) as i64, (c / W) as i64);
    let mut out = Vec::new();
    for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
        let (nx, ny) = (x + dx, y + dy);
        if nx >= 0 && nx < W as i64 && ny >= 0 && ny < H as i64 {
            out.push((ny as usize) * W + nx as usize);
        }
    }
    out
}

/// (tuples, boards, boards reachable from INITIAL): the same rules, independently, for the header.
fn board_stats() -> (usize, usize, usize) {
    let nf = fields().len();
    let tuples = CELLS.pow(nf as u32);
    // Boards: injective assignments of nf pieces to CELLS cells.
    let boards: usize = (0..nf).map(|i| CELLS - i).product();
    let start: Vec<usize> = INITIAL.to_vec();
    let mut seen: HashSet<Vec<usize>> = HashSet::from([start.clone()]);
    let mut queue = VecDeque::from([start]);
    while let Some(s) = queue.pop_front() {
        let blank = s[nf - 1];
        for n in neighbours(blank) {
            let mut nxt: Vec<usize> = s[..nf - 1].iter().map(|&c| if c == n { blank } else { c }).collect();
            nxt.push(n);
            if seen.insert(nxt.clone()) {
                queue.push_back(nxt);
            }
        }
    }
    let mut solved: Vec<usize> = (0..LAST - 1).collect();
    solved.push(CELLS - 1);
    assert!(seen.contains(&solved), "the initial board must be solvable");
    (tuples, boards, seen.len())
}

pub fn spec() -> String {
    let (tuples, boards, reachable) = board_stats();
    let fs = fields();
    let mut distinct = Vec::new();
    for (i, a) in fs.iter().enumerate() {
        for b in &fs[i + 1..] {
            distinct.push(format!("{a} != {b}"));
        }
    }
    let mut lines = vec![
        l("# Dragology's `fifteen` (src/demos/fifteen.tsx), the sliding-tile puzzle: drag a piece and it trades"),
        l("# places with an orthogonal neighbour when one of the two is the blank, and the blank is draggable too."),
        format!("# Scaled down from 4 x 4 to {W} x {H}: tiles 1-{LAST} and the blank, cells numbered row by row from 0."),
        format!("# Tile {LAST} has no field (it is on the cell no other piece is on), so the {} tuples hold {boards}", commas(tuples)),
        format!("# boards; {reachable} are reachable from any one of them, as in the real puzzle. Dragology chains snaps so"),
        l("# one gesture can make several slides; here a gesture makes one slide."),
        generated_by("fifteen"),
        l("nerve: 1"),
        l("name: fifteen"),
        l("state:"),
    ];
    lines.extend(tiles().filter(|&k| has_field(k)).map(|k| format!("  t{k}: {{type: int, range: [0, {}]}}   # cell of tile {k}", CELLS - 1)));
    lines.push(format!("  blank: {{type: int, range: [0, {}]}}   # cell of the blank", CELLS - 1));
    lines.push(l("actions:"));
    lines.extend(tiles().map(|k| format!("  slide_{k}: {{}}   # tile {k} slides into the blank, if it is next to it")));
    lines.push(format!("  slide_blank: {{params: {{to: {{type: int, range: [0, {}]}}}}}}   # the blank trades places with the tile at `to`", CELLS - 1));
    lines.push(l("transitions:"));
    lines.extend(transitions());
    lines.push(l("invariants:"));
    lines.push(format!("  - \"{}\"   # no two pieces share a cell (tile {LAST} gets the one left over)", distinct.join(" and ")));
    lines.push(l("ensures:"));
    lines.extend(ensures());
    lines.push(l("scene:"));
    lines.push(format!("  box: [{}, {}, {}, {}]", ORIGIN as i64 - 36, ORIGIN as i64 - 36, ORIGIN + SIZE * (W - 1) + 36, ORIGIN + SIZE * (H - 1) + 36));
    lines.push(l("  shapes:"));
    let (bx, by) = (ORIGIN as f64 - 30.0 + SIZE as f64 * W as f64 / 2.0, ORIGIN as f64 - 30.0 + SIZE as f64 * H as f64 / 2.0);
    lines.push(format!("    - {{rect: [{bx}, {by}, {}, {}], r: 6, color: panel, stroke: line, width: 1.5}}", SIZE * W, SIZE * H));
    for i in 1..W {
        let x = ORIGIN - 30 + SIZE * i;
        lines.push(format!("    - {{line: [{x}, {}, {x}, {}], color: line, width: 1.5}}", ORIGIN - 30, ORIGIN - 30 + SIZE * H));
    }
    for j in 1..H {
        let y = ORIGIN - 30 + SIZE * j;
        lines.push(format!("    - {{line: [{}, {y}, {}, {y}], color: line, width: 1.5}}", ORIGIN - 30, ORIGIN - 30 + SIZE * W));
    }
    lines.push(l("render:"));
    lines.extend(tiles().map(|k| {
        format!("  tile{k}: {{ {}, shapes: [{{rect: [0, 0, 52, 52], r: 6, color: f{k}, stroke: ink, width: 1.5}}, {{label: '{k}', at: [0, 0], color: white, size: 3}}] }}", position(&cell(k, false)))
    }));
    lines.push(format!("  blank: {{ {}, shape: {{rect: [0, 0, 52, 52], r: 6, color: bg, stroke: mut, width: 1.5}} }}", position("blank")));
    lines.push(l("futures:"));
    lines.extend(futures());
    lines.push(format!("initial: {{{}}}", fs.iter().zip(INITIAL).map(|(f, c)| format!("{f}: {c}")).collect::<Vec<_>>().join(", ")));
    text(lines)
}
