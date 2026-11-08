//! Dragology's `rush-hour` (src/demos/rush-hour.tsx) as specs/rush_hour.nerve, on a 5 x 5 board
//! with six cars.
//!
//! Dragology writes each car's drag by hand (`tryMove`): step along the car's axis, one cell at a
//! time in each direction, until the car would leave the board (the red car's right boundary is
//! the width + 2) or overlap another car's rectangle. The spec writes only `slide_<car>(to)`;
//! these tests run Dragology's loop independently, on car rectangles, and check that the drag
//! futures Nerve derives from the spec are exactly the states it collects.

use nerve::drag::{drag, DragSession};
use nerve::oracle::Oracle;
use nerve::spec::{Spec, State};
use nerve::value::Value;
use rand::SeedableRng;
use std::collections::HashSet;

const W: i64 = 5;
const H: i64 = 5;

/// A car as Dragology draws it, minus the coordinate it never changes: `line` is its row when
/// horizontal, its column when vertical; the state holds the other coordinate.
struct Car {
    name: &'static str,
    horizontal: bool,
    line: i64,
    len: i64,
}

const CARS: [Car; 6] = [
    Car { name: "red", horizontal: true, line: 2, len: 2 },
    Car { name: "a", horizontal: true, line: 0, len: 3 },
    Car { name: "b", horizontal: false, line: 2, len: 2 },
    Car { name: "c", horizontal: false, line: 4, len: 2 },
    Car { name: "d", horizontal: true, line: 4, len: 2 },
    Car { name: "e", horizontal: true, line: 4, len: 2 },
];

/// Positions of the six cars, in spec field order.
type Board = [i64; 6];

fn spec() -> Spec {
    Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/rush_hour.nerve")).unwrap()
}

fn state(b: &Board) -> State {
    b.iter().map(|&p| Value::Int(p)).collect()
}

/// Dragology's `{x, y, w, h}` for car `i` at position `p`.
fn rect(i: usize, p: i64) -> (i64, i64, i64, i64) {
    let c = &CARS[i];
    if c.horizontal {
        (p, c.line, c.len, 1)
    } else {
        (c.line, p, 1, c.len)
    }
}

fn right_boundary(i: usize) -> i64 {
    if CARS[i].name == "red" {
        W + 2
    } else {
        W
    }
}

/// Dragology's out-of-bounds and overlap tests for car `i` moved to `p`, against the others on `b`.
fn blocked(b: &Board, i: usize, p: i64) -> bool {
    let (x, y, w, h) = rect(i, p);
    if x < 0 || x + w > right_boundary(i) || y < 0 || y + h > H {
        return true;
    }
    (0..CARS.len()).filter(|&j| j != i).any(|j| {
        let (tx, ty, tw, th) = rect(j, b[j]);
        !(x + w <= tx || x >= tx + tw || y + h <= ty || y >= ty + th)
    })
}

/// Dragology's `nextStates` for dragging car `i`: the board itself, then every stop of
/// `tryMove` in each direction along the car's axis.
fn dragology_futures(b: &Board, i: usize) -> Vec<Board> {
    let mut out = vec![*b];
    for step in [-1, 1] {
        let mut p = b[i] + step;
        while !blocked(b, i, p) {
            let mut n = *b;
            n[i] = p;
            out.push(n);
            p += step;
        }
    }
    out
}

/// Every placement of the six cars on their lines, inside the bounds Dragology allows.
fn all_placements() -> Vec<Board> {
    let last = |i: usize| (if CARS[i].horizontal { right_boundary(i) } else { H }) - CARS[i].len;
    let mut out: Vec<Board> = vec![[0; 6]];
    for i in 0..CARS.len() {
        out = out
            .into_iter()
            .flat_map(|b| {
                (0..=last(i)).map(move |p| {
                    let mut n = b;
                    n[i] = p;
                    n
                })
            })
            .collect();
    }
    out
}

/// A board is a placement where no two cars overlap (Dragology's overlap test, car against car).
fn all_boards() -> Vec<Board> {
    all_placements()
        .into_iter()
        .filter(|b| {
            (0..CARS.len()).all(|i| {
                let (x, y, w, h) = rect(i, b[i]);
                (i + 1..CARS.len()).all(|j| {
                    let (tx, ty, tw, th) = rect(j, b[j]);
                    x + w <= tx || x >= tx + tw || y + h <= ty || y >= ty + th
                })
            })
        })
        .collect()
}

fn sorted(sp: &Spec, mut v: Vec<State>) -> Vec<State> {
    v.sort_by_key(|s| sp.state_key(s));
    v
}

fn assert_futures_match(sp: &Spec, b: &Board, i: usize) {
    let name = CARS[i].name;
    let f = sp.future(&format!("drag_{name}")).unwrap();
    assert!(f.floating);
    let sess = DragSession::new(sp, f, state(b), None).unwrap();
    let got = sorted(sp, sess.candidates().iter().map(|c| c.state.clone()).collect());
    let want = sorted(sp, dragology_futures(b, i).iter().map(state).collect());
    assert_eq!(got, want, "dragging {name} on {b:?}");
    for c in sess.candidates() {
        if let Some(a) = &c.action {
            assert_eq!(a.name, format!("slide_{name}"), "dragging {name} may only move {name}");
        }
    }
}

/// Where the spec's `render` draws car `i` at position `p`: the middle of its rectangle.
fn px(i: usize, p: i64) -> (f64, f64) {
    let (x, y, w, h) = rect(i, p);
    (30.0 + 50.0 * x as f64 + 25.0 * w as f64, 30.0 + 50.0 * y as f64 + 25.0 * h as f64)
}

const START: Board = [0, 1, 1, 2, 1, 3];

#[test]
fn rush_hour_futures_match_dragology_on_named_boards() {
    let sp = spec();
    assert_eq!(sp.initial, state(&START));
    // The start; the start with d moved left; and the board one drag from out.
    for b in [START, [0, 1, 1, 2, 0, 3], [0, 1, 3, 0, 0, 3]] {
        for i in 0..CARS.len() {
            assert_futures_match(&sp, &b, i);
        }
    }
    let stops = |b: &Board, i: usize| {
        let sess = DragSession::new(&sp, sp.future(&format!("drag_{}", CARS[i].name)).unwrap(), state(b), None).unwrap();
        let mut v: Vec<i64> = sess.candidates().iter().map(|c| c.value.as_f64().unwrap() as i64).collect();
        v.sort();
        v
    };
    assert_eq!(stops(&START, 0), vec![0], "red is boxed in by b");
    assert_eq!(stops(&START, 2), vec![1, 2], "b cannot pass the truck above or d below");
    assert_eq!(stops(&START, 4), vec![0, 1], "d stops at e, on its own row");
    assert_eq!(stops(&START, 5), vec![3], "e is stuck between d and the edge");
    assert_eq!(stops(&[0, 1, 3, 0, 0, 3], 0), vec![0, 1, 2, 3, 4, 5], "red can drive all the way out");
}

#[test]
fn rush_hour_futures_match_dragology_on_every_board() {
    let sp = spec();
    let boards = all_boards();
    assert_eq!(boards.len(), 360);
    for b in &boards {
        for i in 0..CARS.len() {
            assert_futures_match(&sp, b, i);
        }
    }
}

#[test]
fn rush_hour_boards_are_exactly_the_valid_states() {
    let sp = spec();
    let o = Oracle::new(&sp);
    assert_eq!(sp.state_space_size().unwrap(), all_placements().len());
    assert_eq!(sp.state_space_size().unwrap(), 4608);
    let valid: HashSet<String> =
        sp.state_space().unwrap().into_iter().filter(|s| !o.breaks_invariant(s)).map(|s| sp.state_key(&s)).collect();
    let boards: HashSet<String> = all_boards().iter().map(|b| sp.state_key(&state(b))).collect();
    assert_eq!(valid, boards);
}

#[test]
fn rush_hour_legal_moves_are_dragologys_moves() {
    // Every (board, slide) cell that changes the board is one of Dragology's stops, and back.
    let sp = spec();
    let o = Oracle::new(&sp);
    let actions = sp.action_instances().unwrap();
    let mut spec_moves = 0;
    let mut dragology_moves = 0;
    for b in all_boards() {
        let s = state(&b);
        for a in &actions {
            if o.step(&s, a).unwrap() != s {
                spec_moves += 1;
            }
        }
        dragology_moves += (0..CARS.len()).map(|i| dragology_futures(&b, i).len() - 1).sum::<usize>();
    }
    assert_eq!(spec_moves, dragology_moves);
    assert_eq!(spec_moves, 2352);
}

#[test]
fn rush_hour_four_drags_get_the_red_car_out() {
    let sp = spec();
    // d left, b down, c up, then red all the way off the right edge, each by dragging the car.
    let mut b = START;
    for (i, to) in [(4, 0), (2, 3), (3, 0), (0, 5)] {
        let (from, dest) = (px(i, b[i]), px(i, to));
        let mid = ((from.0 + dest.0) / 2.0, (from.1 + dest.1) / 2.0);
        let (_, fin) = drag(&sp, &format!("drag_{}", CARS[i].name), state(&b), &[from, mid, dest]).unwrap();
        b[i] = to;
        assert_eq!(fin, state(&b), "drag {} to {to}", CARS[i].name);
    }
    assert_eq!(b, [5, 1, 3, 0, 0, 3]);
}

#[test]
fn rush_hour_a_car_stops_at_the_first_car_in_its_way() {
    let sp = spec();
    // Red pulled to the exit at the start: b is in the way, so it stays.
    let (_, fin) = drag(&sp, "drag_red", state(&START), &[px(0, 0), px(0, 5)]).unwrap();
    assert_eq!(fin, state(&START));
    // b pulled to the bottom row stops above d.
    let (_, fin) = drag(&sp, "drag_b", state(&START), &[px(2, 1), px(2, 3)]).unwrap();
    assert_eq!(fin, state(&[0, 1, 2, 2, 1, 3]));
}

#[test]
fn rush_hour_passes_its_own_self_check() {
    let sp = spec();
    let problems = Oracle::new(&sp).self_check(&mut rand::rngs::StdRng::seed_from_u64(0)).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}
