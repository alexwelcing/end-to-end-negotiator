//! Dragology's `fifteen` demo (src/demos/fifteen.tsx), the sliding-tile puzzle, on a 3 x 2 board
//! instead of 4 x 4: five tiles and the blank. A drag swaps the dragged piece with an orthogonal
//! neighbour when one of the two is the blank; as in Dragology the blank itself is draggable, onto
//! any tile next to it. Every piece needs a field a gesture can be indexed by, but six cells in [0, 5]
//! are 6^6 = 46,656 tuples. So tile 5 has no field: it is on the one cell no other piece is on,
//! `15 - t1 - t2 - t3 - t4 - blank` (the cells sum to 15). Its gesture is indexed by `blank` instead,
//! which a slide of tile 5 always changes. That leaves 6^5 = 7,776 tuples, of which the 720 with five
//! distinct cells are boards.
//!
//! As in `hanoi`, a `learn` hole on the action `hint` gives the next slide of a shortest solution for
//! every reachable board, found here by breadth-first search from the solved board over the same
//! slide rule the transitions write; the spec itself only bounds a hint to a legal slide. The solved
//! board is an example too, of staying put: a hole's action gets no training pairs from the oracle,
//! so without it a machine pulsing `hint` every tick would be free to slide the solved board apart.

use super::{commas, generated_by, l, text};
use std::collections::{HashMap, VecDeque};

const W: usize = 3;
const H: usize = 2;
const CELLS: usize = W * H;
const LAST: usize = CELLS - 1;
const CELL_SUM: usize = 15;
const SIZE: usize = 60;
const ORIGIN: usize = 50;
/// Cells of t1..t4 and the blank: four slides from solved (1 2 3 / 4 5 _).
const INITIAL: [usize; 5] = [0, 2, 5, 4, 3];

/// The cells of t1..t4 and the blank, in field order.
type Board = Vec<usize>;

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

/// The solved board, 1 2 3 / 4 5 _: tiles 1-4 on cells 0-3 (so tile 5 on cell 4) and the blank on the last cell.
fn solved_board() -> Board {
    let mut b: Board = (0..LAST - 1).collect();
    b.push(CELLS - 1);
    b
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

/// The board is solved (tile 5's cell follows from the others').
fn solved_expr(old: bool) -> String {
    let pre = if old { "old." } else { "" };
    fields().iter().zip(solved_board()).map(|(f, c)| format!("{pre}{f} == {c}")).collect::<Vec<_>>().join(" and ")
}

/// `solved` fires on the step that completes the board: a slide, or a hint the network learned (the
/// spec's own hint changes nothing, so the event is derived from whatever successor is checked).
fn solved_emit() -> String {
    format!("    emits: {{ solved: \"{} and not ({})\" }}", solved_expr(false), solved_expr(true))
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
        out.push(solved_emit());
    }
    out.push(l("  slide_blank:"));
    out.push(format!("    when: \"{}\"", adjacent("blank", "to")));
    out.push(l("    blank: \"to\""));
    out.extend(tiles().filter(|&k| has_field(k)).map(|k| format!("    t{k}: \"blank if t{k} == to else t{k}\"")));
    out.push(solved_emit());
    out.push(l("  # the spec's own hint changes nothing: the network's is learned from the examples under `learn`"));
    out.push(l("  hint:"));
    out.extend(fields().iter().map(|f| format!("    {f}: \"{f}\"")));
    out.push(solved_emit());
    out
}

/// How many tiles are on a different cell than before.
fn moved_tiles() -> String {
    tiles().map(|k| format!("int({} != {})", cell(k, false), cell(k, true))).collect::<Vec<_>>().join(" + ")
}

/// Tile `k` and the blank traded places.
fn trade(k: usize) -> String {
    format!("({} == old.blank and blank == {})", cell(k, false), cell(k, true))
}

/// Neither tile `k` nor the blank moved.
fn stays(k: usize) -> String {
    format!("({} == {} and blank == old.blank)", cell(k, false), cell(k, true))
}

/// What a hint may do: nothing, or one slide (the blank trades places with a tile on a neighbouring
/// cell and no other tile moves), as the slide transitions do.
fn legal_single_slide() -> String {
    let still = fields().iter().map(|f| format!("{f} == old.{f}")).collect::<Vec<_>>().join(" and ");
    let trades = tiles().map(trade).collect::<Vec<_>>().join(" or ");
    format!("{} <= 1 and (({still}) or ({} and ({trades})))", moved_tiles(), adjacent("blank", "old.blank"))
}

fn ensures() -> Vec<String> {
    let mut out = vec![
        format!("  - {{ holds: \"abs(blank // {W} - old.blank // {W}) + abs(blank % {W} - old.blank % {W}) <= 1\" }}   # one slide at a time: the blank moves to a neighbouring cell or stays"),
        format!("  - {{ holds: \"{} <= 1\" }}   # and at most one tile moves, onto the blank's old cell", moved_tiles()),
    ];
    for k in tiles() {
        out.push(format!("  - {{ after: slide_{k}, holds: \"{} or {}\" }}   # the tile trades places with the blank, or nothing moves", trade(k), stays(k)));
    }
    out.push(l("  - { after: slide_blank, holds: \"blank == to or blank == old.blank\" }   # the blank lands where it was dropped, or stays"));
    out.push(format!("  - {{ after: hint, holds: \"{}\" }}   # a hint may only make a legal slide: the blank stays, or trades places with a neighbouring tile", legal_single_slide()));
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

/// The tile on cell `c` of `s`: tile 5 has no field, so it is on the cell none of the others is on.
fn tile_on(s: &Board, c: usize) -> usize {
    s[..s.len() - 1].iter().position(|&x| x == c).map_or(LAST, |i| i + 1)
}

/// Every slide from `s`, as (the tile that slid, the board after): the blank trades places with a
/// neighbour. The same rule the transitions write, independently.
fn slides(s: &Board) -> Vec<(usize, Board)> {
    let blank = s[s.len() - 1];
    neighbours(blank)
        .into_iter()
        .map(|n| {
            let mut next: Board = s[..s.len() - 1].iter().map(|&c| if c == n { blank } else { c }).collect();
            next.push(n);
            (tile_on(s, n), next)
        })
        .collect()
}

/// Every board reachable from `start`, with how many slides it is away: breadth-first search.
fn distances(start: &Board) -> HashMap<Board, usize> {
    let mut dist: HashMap<Board, usize> = HashMap::from([(start.clone(), 0)]);
    let mut queue = VecDeque::from([start.clone()]);
    while let Some(s) = queue.pop_front() {
        let d = dist[&s];
        for (_, n) in slides(&s) {
            if !dist.contains_key(&n) {
                dist.insert(n.clone(), d + 1);
                queue.push_back(n);
            }
        }
    }
    dist
}

/// The next slide of a shortest solution for every reachable board, as (board, board after), sorted;
/// and the length of the longest solution. Slides are reversible, so the search runs outward from the
/// solved board. Where two slides are equally short the lowest-numbered tile slides, so the table is
/// deterministic. The solved board's next is itself.
fn solution() -> (Vec<(Board, Board)>, usize) {
    let dist = distances(&solved_board());
    let mut best: Vec<(Board, Board)> = dist
        .iter()
        .map(|(s, &d)| {
            if d == 0 {
                return (s.clone(), s.clone());
            }
            let closer = slides(s).into_iter().filter(|(_, n)| dist[n] == d - 1).min_by_key(|(k, _)| *k);
            (s.clone(), closer.expect("a board d slides from solved has a slide to one d - 1 away").1)
        })
        .collect();
    best.sort();
    let farthest = dist.values().copied().max().unwrap_or(0);
    (best, farthest)
}

/// (tuples, boards, boards reachable from INITIAL): the same rules, independently, for the header.
fn board_stats() -> (usize, usize, usize) {
    let nf = fields().len();
    let tuples = CELLS.pow(nf as u32);
    // Boards: injective assignments of nf pieces to CELLS cells.
    let boards: usize = (0..nf).map(|i| CELLS - i).product();
    let reachable = distances(&INITIAL.to_vec());
    assert!(reachable.contains_key(&solved_board()), "the initial board must be solvable");
    (tuples, boards, reachable.len())
}

pub fn spec() -> String {
    let (tuples, boards, reachable) = board_stats();
    let (best, farthest) = solution();
    assert_eq!(best.len(), reachable, "one hint example per reachable board");
    let fs = fields();
    let mut distinct = Vec::new();
    for (i, a) in fs.iter().enumerate() {
        for b in &fs[i + 1..] {
            distinct.push(format!("{a} != {b}"));
        }
    }
    let board = |b: &Board| fs.iter().zip(b).map(|(f, c)| format!("{f}: {c}")).collect::<Vec<_>>().join(", ");
    let examples: Vec<String> = best.iter().map(|(s, n)| format!("      - {{ state: {{{}}}, next: {{{}}} }}", board(s), board(n))).collect();
    let mut lines = vec![
        l("# Dragology's `fifteen` (src/demos/fifteen.tsx), the sliding-tile puzzle: drag a piece and it trades"),
        l("# places with an orthogonal neighbour when one of the two is the blank, and the blank is draggable too."),
        format!("# Scaled down from 4 x 4 to {W} x {H}: tiles 1-{LAST} and the blank, cells numbered row by row from 0."),
        format!("# Tile {LAST} has no field (it is on the cell no other piece is on), so the {} tuples hold {boards}", commas(tuples)),
        format!("# boards; {reachable} are reachable from any one of them, as in the real puzzle. Dragology chains snaps so"),
        l("# one gesture can make several slides; here a gesture makes one slide."),
        l("# `hint` is the next slide of a shortest solution (1 2 3 / 4 5 _ is solved), given only by example: one per"),
        format!("# reachable board, from a breadth-first search in the generator (the farthest board is {farthest} slides out)."),
        l("# The spec only bounds it, to a legal slide. `solved` fires on the step that completes the board, so a"),
        l("# machine can pulse `hint` every tick and the puzzle solves itself."),
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
    lines.push(l("  hint: {}   # the next slide of a shortest solution: shown by examples, never written down"));
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
    lines.push(l("learn:"));
    lines.push(l("  # the `hint` transition above is the fallback; the examples are the whole breadth-first solution"));
    lines.push(l("  - name: solve"));
    lines.push(l("    action: hint"));
    lines.push(l("    examples:"));
    lines.extend(examples);
    lines.push(format!("initial: {{{}}}", fs.iter().zip(INITIAL).map(|(f, c)| format!("{f}: {c}")).collect::<Vec<_>>().join(", ")));
    text(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::Oracle;
    use crate::spec::{ActionInst, Spec, State};
    use crate::value::Value;

    fn state(b: &Board) -> State {
        b.iter().map(|&c| Value::Int(c as i64)).collect()
    }

    fn action(name: &str) -> ActionInst {
        ActionInst { name: name.to_string(), params: Vec::new() }
    }

    #[test]
    fn the_hint_table_is_a_shortest_solution_from_every_reachable_board() {
        let (best, farthest) = solution();
        let (_, _, reachable) = board_stats();
        assert_eq!((best.len(), reachable), (360, 360));
        let dist = distances(&solved_board());
        assert_eq!(dist[&INITIAL.to_vec()], 4, "the initial board is four slides from solved");
        assert!(farthest > 4, "some board is farther out than the initial one");
        let table: HashMap<&Board, &Board> = best.iter().map(|(s, n)| (s, n)).collect();
        for (s, n) in &best {
            if *s == solved_board() {
                assert_eq!(s, n, "the solved board stays put");
                continue;
            }
            assert!(slides(s).iter().any(|(_, after)| after == n), "{s:?} -> {n:?} is one slide");
            assert_eq!(dist[n] + 1, dist[s], "{s:?} -> {n:?} is one slide closer");
        }
        // Following the hints from any board reaches the solved board in exactly its distance.
        for (s, &d) in &dist {
            let mut cur = s;
            for _ in 0..d {
                cur = table[cur];
            }
            assert_eq!(*cur, solved_board(), "{s:?} is solved in {d} hints");
        }
    }

    #[test]
    fn every_hint_example_is_a_slide_the_spec_allows_and_solved_fires_once() {
        let sp = Spec::parse(&spec()).unwrap();
        let o = Oracle::new(&sp);
        let hole = sp.holes.iter().find(|h| h.name == "solve").expect("the solve hole");
        assert_eq!(hole.action, "hint");
        assert_eq!(hole.examples.len(), 360);
        let hint = action("hint");
        let slide_actions: Vec<ActionInst> = tiles().map(|k| action(&format!("slide_{k}"))).collect();
        let solved = state(&solved_board());
        for ex in &hole.examples {
            let key = sp.state_key(&ex.state);
            let v = o.violations(&ex.state, Some(&hint), &ex.next, 0.01);
            assert!(v.is_empty(), "{key}: {v:?}");
            let is_slide = slide_actions.iter().any(|a| o.step(&ex.state, a).unwrap() == ex.next && ex.next != ex.state);
            assert!(is_slide || (ex.state == solved && ex.next == solved), "{key}: the example is one of the spec's own slides");
            let events = o.events(&ex.state, &hint, &ex.next).unwrap();
            assert_eq!(events == ["solved"], ex.next == solved && ex.state != solved, "{key}: `solved` fires exactly on the completing hint");
        }
        // The spec's own hint changes nothing and emits nothing, from the solved board too.
        assert_eq!(o.step(&solved, &hint).unwrap(), solved);
        assert!(o.events(&solved, &hint, &solved).unwrap().is_empty());
    }
}
