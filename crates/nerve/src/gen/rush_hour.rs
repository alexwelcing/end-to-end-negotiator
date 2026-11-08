//! Dragology's `rush-hour` demo (src/demos/rush-hour.tsx): cars slide along their own row or column,
//! any number of cells, until the next cell is off the board or taken by another car. The red car
//! alone may drive up to two cells past the right edge, which is how it leaves. Scaled down from a
//! 6 x 6 board with eight cars to 5 x 5 with six: a car only ever moves along its line, so its state is
//! one int, and the six cars make 4,608 tuples. `slide_<car>(to)` moves that car to `to` when nothing
//! is in the way: no other car may touch any cell the car sweeps through, exactly the run of positions
//! Dragology's `tryMove` walks before it hits something. `ensures` say the same of any transition.
//! `hint` is the way out, learned from examples alone as hanoi's is: one per board the red car can
//! still leave, the board after the first slide of a shortest escape, found here by breadth-first
//! search backwards from every board with the red car out. A machine pulsing `hint` every tick
//! drives the red car off the board, and the slide that takes it out emits `escaped`.

use super::{commas, generated_by, l, text};
use std::collections::{HashMap, HashSet, VecDeque};

const W: usize = 5;
const H: usize = 5;
const EXIT: usize = 2;
const SIZE: usize = 50;
const ORIGIN: usize = 30;

#[derive(Clone, Copy, PartialEq)]
struct Car {
    name: &'static str,
    horizontal: bool,
    line: usize,
    length: usize,
    start: usize,
}

impl Car {
    /// Largest position: the board edge, or past it for the red car.
    fn last(&self) -> usize {
        let span = if self.horizontal { if self.name == "red" { W + EXIT } else { W } } else { H };
        span - self.length
    }

    fn cells(&self, p: usize) -> HashSet<(usize, usize)> {
        (0..self.length).map(|i| if self.horizontal { (p + i, self.line) } else { (self.line, p + i) }).collect()
    }
}

const CARS: [Car; 6] = [
    Car { name: "red", horizontal: true, line: 2, length: 2, start: 0 },
    Car { name: "a", horizontal: true, line: 0, length: 3, start: 1 },
    Car { name: "b", horizontal: false, line: 2, length: 2, start: 1 },
    Car { name: "c", horizontal: false, line: 4, length: 2, start: 2 },
    Car { name: "d", horizontal: true, line: 4, length: 2, start: 1 },
    Car { name: "e", horizontal: true, line: 4, length: 2, start: 3 },
];

/// Positions of `car` at which it covers `cell` of its line, or None if it never does.
fn covers(car: &Car, cell: usize) -> Option<(usize, usize)> {
    let lo = (cell as i64 - car.length as i64 + 1).max(0) as usize;
    let hi = car.last().min(cell);
    (lo <= hi).then_some((lo, hi))
}

/// `car` (at `pos`, by default its field) is in `span`; None when that is every position it has.
fn between(car: &Car, span: (usize, usize), pos: Option<&str>) -> Option<String> {
    let (lo, hi) = span;
    let pos = pos.unwrap_or(car.name);
    if (lo, hi) == (0, car.last()) {
        return None;
    }
    Some(if lo == hi { format!("{pos} == {lo}") } else { format!("{lo} <= {pos} <= {hi}") })
}

fn both(terms: &[Option<String>]) -> String {
    let t: Vec<&str> = terms.iter().flatten().map(String::as_str).collect();
    if t.is_empty() { l("True") } else { t.join(" and ") }
}

/// Cars `k` and `o` share a cell, or None if they never can.
fn overlap(k: &Car, o: &Car) -> Option<String> {
    if k.horizontal == o.horizontal {
        if k.line != o.line {
            return None;
        }
        return Some(format!("{} < {} + {} and {} < {} + {}", k.name, o.name, o.length, o.name, k.name, k.length));
    }
    let (on_k, on_o) = (covers(k, o.line)?, covers(o, k.line)?);
    Some(both(&[between(k, on_k, None), between(o, on_o, None)]))
}

/// `o` (at `other`) touches a cell `k` sweeps through going from `src` to `dst`.
fn blocks(k: &Car, o: &Car, src: Option<&str>, dst: &str, other: Option<&str>) -> Option<String> {
    let src = src.unwrap_or(k.name);
    let other = other.unwrap_or(o.name);
    let lo = format!("min({src}, {dst})");
    let hi = format!("max({src}, {dst}) + {}", k.length - 1);
    if k.horizontal == o.horizontal {
        if k.line != o.line {
            return None;
        }
        return Some(format!("{other} <= {hi} and {lo} <= {other} + {}", o.length - 1));
    }
    let on_o = covers(o, k.line)?;
    Some(both(&[Some(format!("{lo} <= {} <= {hi}", o.line)), between(o, on_o, Some(other))]))
}

/// `ensures`: whatever moved `k`, no other car was on a cell between where it was and where it is.
fn never_passes(k: &Car) -> String {
    CARS.iter()
        .filter(|o| *o != k)
        .filter_map(|o| blocks(k, o, Some(&format!("old.{}", k.name)), k.name, Some(&format!("old.{}", o.name))))
        .map(|b| format!("not ({b})"))
        .collect::<Vec<_>>()
        .join(" and ")
}

/// `emits`: the red car has just got out, reaching the first position with every cell past the edge.
fn escaped() -> String {
    let red = CARS[0].name;
    format!("emits: {{ escaped: \"{red} == {W} and old.{red} < {W}\" }}")
}

fn transitions() -> Vec<String> {
    let mut out = Vec::new();
    for k in &CARS {
        let clear: Vec<String> = CARS.iter().filter(|o| *o != k).filter_map(|o| blocks(k, o, None, "to", None)).map(|b| format!("not ({b})")).collect();
        out.push(format!("  slide_{}:", k.name));
        out.push(format!("    when: \"{}\"", clear.join(" and ")));
        out.push(format!("    {}: \"to\"", k.name));
        if k.name == "red" {
            out.push(format!("    {}", escaped()));
        }
    }
    // The hole's fallback: a board no example covers stays as it is. The network learns the examples on top.
    out.push(l("  hint:"));
    out.extend(CARS.iter().map(|k| format!("    {}: \"{}\"", k.name, k.name)));
    out.push(format!("    {}", escaped()));
    out
}

fn render(car: &Car) -> String {
    let pos = format!("{SIZE} * {}", car.name);
    let along = format!("{} + {pos}", ORIGIN + SIZE * car.length / 2);
    let across = format!("{}", ORIGIN + SIZE * car.line + SIZE / 2);
    let (x, y) = if car.horizontal { (along, across) } else { (across, along) };
    let (w, h) = if car.horizontal { (SIZE * car.length - 8, 40) } else { (40, SIZE * car.length - 8) };
    let color = match car.name {
        "red" => "f4",
        "a" => "f0",
        "b" => "f1",
        "c" => "f5",
        "d" => "f2",
        _ => "f3",
    };
    format!(
        "  {}: {{ x: \"{x}\", y: \"{y}\", shapes: [{{rect: [0, 0, {w}, {h}], r: 9, color: {color}, stroke: ink, width: 1.5}}, {{label: {}, at: [0, 0], color: white, size: 2}}] }}",
        car.name, car.name
    )
}

fn describe(car: &Car) -> String {
    let where_ = if car.horizontal { format!("row {}, left end", car.line) } else { format!("column {}, top end", car.line) };
    let off = if car.last() > W - car.length { format!("; past {} it drives off the board, and at {W} it is out", W - car.length) } else { String::new() };
    format!("{}, length {}: {where_}{off}", if car.horizontal { "horizontal" } else { "vertical" }, car.length)
}

type Board = [usize; 6];

fn is_board(s: &Board) -> bool {
    let taken: Vec<HashSet<(usize, usize)>> = CARS.iter().zip(s).map(|(c, &p)| c.cells(p)).collect();
    let total: usize = taken.iter().map(HashSet::len).sum();
    let union: HashSet<(usize, usize)> = taken.iter().flatten().copied().collect();
    total == union.len()
}

/// Dragology's tryMove for every car and both directions: every position up to the first block.
fn slides(s: &Board) -> Vec<Board> {
    let mut out = Vec::new();
    for (i, car) in CARS.iter().enumerate() {
        let others: HashSet<(usize, usize)> = CARS.iter().zip(s).enumerate().filter(|(j, _)| *j != i).flat_map(|(_, (c, &p))| c.cells(p)).collect();
        for step in [-1i64, 1] {
            let mut p = s[i] as i64 + step;
            while p >= 0 && p <= car.last() as i64 && car.cells(p as usize).is_disjoint(&others) {
                let mut n = *s;
                n[i] = p as usize;
                out.push(n);
                p += step;
            }
        }
    }
    out
}

/// Every tuple of positions, one per value of the state, in field order (the red car varies slowest).
fn tuples() -> Vec<Board> {
    let mut out: Vec<Board> = vec![[0; 6]];
    for (i, car) in CARS.iter().enumerate() {
        out = out.iter().flat_map(|t| (0..=car.last()).map(move |p| { let mut n = *t; n[i] = p; n })).collect();
    }
    out
}

/// Fewest slides from each board to one with the red car out, by breadth-first search backwards from
/// every such board at once: a slide runs back along the same clear cells, so the graph is undirected.
/// A board the red car can never leave is absent.
fn distances(boards: &[Board]) -> HashMap<Board, usize> {
    let out: Vec<Board> = boards.iter().copied().filter(|b| b[0] == W).collect();
    let mut dist: HashMap<Board, usize> = out.iter().map(|b| (*b, 0)).collect();
    let mut queue: VecDeque<Board> = out.into_iter().collect();
    while let Some(s) = queue.pop_front() {
        for n in slides(&s) {
            if !dist.contains_key(&n) {
                dist.insert(n, dist[&s] + 1);
                queue.push_back(n);
            }
        }
    }
    dist
}

/// The `hint` examples: for every board the red car can still get out of, the board after the first slide
/// of a shortest way out. Ties break by car, in state order (red, a, b, c, d, e), then by the nearer stop.
fn hints(boards: &[Board], dist: &HashMap<Board, usize>) -> Vec<(Board, Board)> {
    let mut out = Vec::new();
    for s in boards {
        let Some(&d) = dist.get(s) else { continue };
        if d == 0 {
            continue;
        }
        let moved = |n: &Board| (0..CARS.len()).find(|&i| n[i] != s[i]).expect("a slide moves one car");
        let mut closer: Vec<(usize, usize, Board)> = slides(s)
            .into_iter()
            .filter(|n| dist.get(n) == Some(&(d - 1)))
            .map(|n| {
                let i = moved(&n);
                (i, n[i], n)
            })
            .collect();
        closer.sort_unstable();
        out.push((*s, closer[0].2));
    }
    out
}

pub fn spec() -> String {
    let tuples = tuples();
    let n_tuples = tuples.len();
    let boards: Vec<Board> = tuples.into_iter().filter(is_board).collect();
    let n_boards = boards.len();
    let dist = distances(&boards);
    let hints = hints(&boards, &dist);
    let start: Board = CARS.map(|c| c.start);
    assert!(is_board(&start));
    let solution = dist.get(&start).copied().expect("the red car cannot get out");
    let n_out = boards.iter().filter(|b| b[0] == W).count();
    let n_stuck = n_boards - dist.len();
    let longest = dist.values().copied().max().unwrap_or(0);
    let rest = if n_stuck == 0 {
        format!("{n_out} boards all have it out already; they")
    } else {
        format!("{} boards ({n_out} have it out already, {n_stuck} are stuck for good)", n_out + n_stuck)
    };
    let board = |b: &Board| CARS.iter().zip(b).map(|(c, p)| format!("{}: {p}", c.name)).collect::<Vec<_>>().join(", ");
    let examples: Vec<String> = hints.iter().map(|(s, n)| format!("      - {{ state: {{{}}}, next: {{{}}} }}", board(s), board(n))).collect();
    let mut apart = Vec::new();
    for (i, k) in CARS.iter().enumerate() {
        for o in &CARS[i + 1..] {
            if let Some(e) = overlap(k, o) {
                apart.push(format!("  - \"not ({e})\"   # {} and {} never share a cell", k.name, o.name));
            }
        }
    }
    let one_car: Vec<String> = CARS.iter().map(|c| format!("int({} != old.{})", c.name, c.name)).collect();
    let mut lines = vec![
        l("# Dragology's `rush-hour` (src/demos/rush-hour.tsx): a car slides along its row or column, any number"),
        l("# of cells, until the next is off the board or taken; the red car alone may drive two cells past the"),
        format!("# right edge, which is how it gets out. Scaled down from 6 x 6 with eight cars to {W} x {H} with six: each"),
        format!("# car is one int, its position along its line, so there are {} tuples, {n_boards} of them boards.", commas(n_tuples)),
        format!("# From the start the red car is out in {solution} drags. Dragology slides a car smoothly between its"),
        l("# stops (`between`); here the drop snaps to the closest one."),
        format!("# `hint` is the way out, given only by example as hanoi's is: for each of the {} boards the red car can", hints.len()),
        format!("# still leave, the board after the first slide of a shortest escape ({longest} slides at most). The other"),
        format!("# {rest} get no example, and a hint leaves them as they are."),
        l("# A machine pulsing `hint` every tick drives the red car off the board; `escaped` fires as it leaves."),
        generated_by("rush_hour"),
        l("nerve: 1"),
        l("name: rush_hour"),
        l("state:"),
    ];
    lines.extend(CARS.iter().map(|c| format!("  {}: {{type: int, range: [0, {}]}}   # {}", c.name, c.last(), describe(c))));
    lines.push(l("actions:"));
    lines.extend(CARS.iter().map(|c| format!("  slide_{}: {{params: {{to: {{type: int, range: [0, {}]}}}}}}", c.name, c.last())));
    lines.push(l("  hint: {}   # the first slide of a shortest way out: shown by examples, never written down"));
    lines.push(l("transitions:"));
    lines.push(l("  # a car goes to `to` when no other car touches a cell it would sweep through on the way"));
    lines.extend(transitions());
    lines.push(l("invariants:"));
    lines.extend(apart);
    lines.push(l("ensures:"));
    lines.push(format!("  - {{ holds: \"{} <= 1\" }}   # one car moves at a time", one_car.join(" + ")));
    lines.extend(CARS.iter().map(|c| format!("  - {{ holds: \"{}\" }}   # {} never passes through another car", never_passes(c), c.name)));
    lines.extend(CARS.iter().map(|c| format!("  - {{ after: slide_{}, holds: \"{} == to or {} == old.{}\" }}   # it stops where it was dropped, or stays", c.name, c.name, c.name, c.name)));
    let any_slide: Vec<String> = CARS.iter().map(never_passes).collect();
    lines.push(format!("  - {{ after: hint, holds: \"{} <= 1 and {}\" }}   # a hint is one legal slide of one car", one_car.join(" + "), any_slide.join(" and ")));
    lines.push(l("scene:"));
    lines.push(format!("  box: [{}, {}, {}, {}]", ORIGIN - 10, ORIGIN - 10, ORIGIN + SIZE * W + 65, ORIGIN + SIZE * H + 10));
    lines.push(l("  shapes:"));
    let (cx, cy) = (ORIGIN as f64 + SIZE as f64 * W as f64 / 2.0, ORIGIN as f64 + SIZE as f64 * H as f64 / 2.0);
    lines.push(format!("    - {{rect: [{cx}, {cy}, {}, {}], r: 6, color: panel, stroke: line, width: 1.5}}", SIZE * W, SIZE * H));
    for i in 1..W {
        let x = ORIGIN + SIZE * i;
        lines.push(format!("    - {{line: [{x}, {ORIGIN}, {x}, {}], color: line, width: 1.5}}", ORIGIN + SIZE * H));
    }
    for j in 1..H {
        let y = ORIGIN + SIZE * j;
        lines.push(format!("    - {{line: [{ORIGIN}, {y}, {}, {y}], color: line, width: 1.5}}", ORIGIN + SIZE * W));
    }
    // The exit: the red car's row opens to the right, where only it may drive out.
    let exit_y = ORIGIN + SIZE * CARS[0].line + SIZE / 2;
    let right = ORIGIN + SIZE * W;
    lines.push(format!("    - {{rect: [{}, {exit_y}, 4, 46], color: bg}}", right + 1));
    lines.push(format!("    - {{rect: [{}, {exit_y}, 24, 6], color: f4, opacity: 0.6}}", right + 25));
    lines.push(format!("    - {{label: '>', at: [{}, {exit_y}], color: f4, size: 3}}", right + 50));
    lines.push(format!("    - {{label: exit, at: [{}, {}], color: mut, size: 1.8}}", right + 33, exit_y + 25));
    lines.push(l("render:"));
    lines.extend(CARS.iter().map(render));
    lines.push(l("futures:"));
    lines.push(l("  # one gesture per car: every stop it can slide to along its line, in either direction"));
    lines.extend(CARS.iter().map(|c| format!("  drag_{}: {{ kind: closest, over: {}, grab: {}, action: slide_{}, stickiness: 6, floating: true }}", c.name, c.name, c.name, c.name)));
    lines.push(l("learn:"));
    lines.push(l("  - name: solve"));
    lines.push(l("    action: hint   # its transition above is the fallback for the boards no example covers"));
    lines.push(l("    examples:"));
    lines.extend(examples);
    lines.push(format!("initial: {{{}}}", CARS.iter().map(|c| format!("{}: {}", c.name, c.start)).collect::<Vec<_>>().join(", ")));
    text(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::Oracle;
    use crate::spec::{ActionInst, Spec, State};
    use crate::value::Value;

    fn search() -> (Vec<Board>, HashMap<Board, usize>) {
        let boards: Vec<Board> = tuples().into_iter().filter(is_board).collect();
        let dist = distances(&boards);
        (boards, dist)
    }

    #[test]
    fn every_hint_is_one_legal_slide_toward_the_exit() {
        let sp = Spec::parse(&spec()).unwrap();
        let o = Oracle::new(&sp);
        let hole = sp.holes.iter().find(|h| h.name == "solve").expect("the solve hole");
        assert_eq!(hole.action, "hint");
        let (boards, dist) = search();
        let out = boards.iter().filter(|b| b[0] == W).count();
        assert_eq!(hole.examples.len(), dist.len() - out, "one example per board the red car can still leave");
        let hint = ActionInst::new("hint");
        for ex in &hole.examples {
            assert_ne!(ex.state, ex.next, "a hint moves a car");
            let v = o.violations(&ex.state, Some(&hint), &ex.next, 0.01);
            assert!(v.is_empty(), "{} -> {}: {v:?}", sp.state_key(&ex.state), sp.state_key(&ex.next));
        }
    }

    #[test]
    fn hints_from_the_start_take_the_red_car_out_in_the_fewest_slides() {
        let sp = Spec::parse(&spec()).unwrap();
        let hole = sp.holes.iter().find(|h| h.name == "solve").unwrap();
        let next_of: HashMap<String, &State> = hole.examples.iter().map(|e| (sp.state_key(&e.state), &e.next)).collect();
        let (_, dist) = search();
        let start: Board = CARS.map(|c| c.start);
        let mut s = sp.initial.clone();
        let mut n = 0;
        while s[0] != Value::Int(W as i64) {
            s = next_of[&sp.state_key(&s)].clone();
            n += 1;
        }
        assert_eq!(n, dist[&start]);
        assert_eq!(n, 4, "the header's count of drags from the start");
    }
}
