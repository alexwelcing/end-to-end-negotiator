//! Dragology's `fifteen` (src/demos/fifteen.tsx) as specs/fifteen.nerve, on a 3 x 2 board.
//!
//! Dragology writes each piece's drag by hand: look at the four neighbours, and swap with one
//! when the dragged piece or the neighbour is the blank. The spec writes only `slide_<piece>`;
//! these tests compute Dragology's swaps independently, from a board of cells, and check that
//! the drag futures Nerve derives from the spec are exactly those.

use nerve::drag::{drag, DragSession};
use nerve::oracle::Oracle;
use nerve::spec::{Spec, State};
use nerve::value::Value;
use rand::SeedableRng;
use std::collections::{HashSet, VecDeque};

const W: usize = 3;
const H: usize = 2;
const CELLS: usize = W * H;
const BLANK: u8 = 0;
/// Every piece, tiles then the blank, with its gesture and the action that gesture may use.
const PIECES: [(u8, &str, &str); 6] = [
    (1, "drag_1", "slide_1"),
    (2, "drag_2", "slide_2"),
    (3, "drag_3", "slide_3"),
    (4, "drag_4", "slide_4"),
    (5, "drag_5", "slide_5"),
    (BLANK, "drag_blank", "slide_blank"),
];

/// The piece on each cell, row by row; 0 is the blank.
type Board = [u8; CELLS];

fn spec() -> Spec {
    Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/fifteen.nerve")).unwrap()
}

/// `"152_43"` is `1 5 2` over `_ 4 3`.
fn board(s: &str) -> Board {
    let mut b = [BLANK; CELLS];
    for (i, ch) in s.chars().enumerate() {
        b[i] = if ch == '_' { BLANK } else { ch.to_digit(10).unwrap() as u8 };
    }
    b
}

fn cell_of(b: &Board, piece: u8) -> usize {
    b.iter().position(|&p| p == piece).unwrap()
}

/// The spec's state `[t1, t2, t3, t4, blank]`: tile 5 has no field.
fn state(b: &Board) -> State {
    [1, 2, 3, 4, BLANK].iter().map(|&p| Value::Int(cell_of(b, p) as i64)).collect()
}

/// Where the spec's `render` draws a cell.
fn px(cell: usize) -> (f64, f64) {
    (50.0 + 60.0 * (cell % W) as f64, 50.0 + 60.0 * (cell / W) as f64)
}

/// Dragology's `dragologyOnDrag` for `piece`: each neighbour (left, right, up, down) on the
/// board, swapped in when the piece or the neighbour is the blank; plus staying put.
fn dragology_futures(b: &Board, piece: u8) -> Vec<Board> {
    let i = cell_of(b, piece);
    let (x, y) = ((i % W) as i64, (i / W) as i64);
    let mut out = vec![*b];
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let (ax, ay) = (x + dx, y + dy);
        if ax < 0 || ax >= W as i64 || ay < 0 || ay >= H as i64 {
            continue;
        }
        let j = ay as usize * W + ax as usize;
        if piece != BLANK && b[j] != BLANK {
            continue;
        }
        let mut n = *b;
        n.swap(i, j);
        out.push(n);
    }
    out
}

/// Every board: the 720 arrangements of five tiles and the blank.
fn all_boards() -> Vec<Board> {
    fn go(prefix: &mut Vec<u8>, left: &mut Vec<u8>, out: &mut Vec<Board>) {
        if left.is_empty() {
            out.push(prefix.as_slice().try_into().unwrap());
            return;
        }
        for k in 0..left.len() {
            let p = left.remove(k);
            prefix.push(p);
            go(prefix, left, out);
            prefix.pop();
            left.insert(k, p);
        }
    }
    let mut out = Vec::new();
    go(&mut Vec::new(), &mut (0..CELLS as u8).collect(), &mut out);
    out
}

fn sorted(sp: &Spec, mut v: Vec<State>) -> Vec<State> {
    v.sort_by_key(|s| sp.state_key(s));
    v
}

/// The drag's candidate states for `piece` on `b` equal Dragology's, and only that piece's action realises them.
fn assert_futures_match(sp: &Spec, b: &Board, piece: u8, gesture: &str, action: &str) {
    let f = sp.future(gesture).unwrap();
    assert!(f.floating, "{gesture} floats like Dragology's");
    let sess = DragSession::new(sp, f, state(b), None).unwrap();
    let got = sorted(sp, sess.candidates().iter().map(|c| c.state.clone()).collect());
    let want = sorted(sp, dragology_futures(b, piece).iter().map(state).collect());
    assert_eq!(got, want, "dragging {piece} on {b:?}");
    for c in sess.candidates() {
        if let Some(a) = &c.action {
            assert_eq!(a.name, action, "dragging {piece} on {b:?} may only use its own action");
        }
    }
}

#[test]
fn fifteen_futures_match_dragology_on_named_boards() {
    let sp = spec();
    // Blank in a corner, blank in the middle of an edge, and the solved board.
    for (s, blank_moves) in [("152_43", 2), ("1_2453", 3), ("12345_", 2)] {
        let b = board(s);
        for (piece, gesture, action) in PIECES {
            assert_futures_match(&sp, &b, piece, gesture, action);
        }
        let sess = DragSession::new(&sp, sp.future("drag_blank").unwrap(), state(&b), None).unwrap();
        assert_eq!(sess.candidates().len(), 1 + blank_moves, "the blank on {s} has {blank_moves} neighbours");
    }
}

#[test]
fn fifteen_futures_match_dragology_on_every_board() {
    let sp = spec();
    let boards = all_boards();
    assert_eq!(boards.len(), 720);
    for b in &boards {
        for (piece, gesture, action) in PIECES {
            assert_futures_match(&sp, b, piece, gesture, action);
        }
    }
}

#[test]
fn fifteen_boards_are_exactly_the_valid_states() {
    let sp = spec();
    let o = Oracle::new(&sp);
    assert_eq!(sp.state_space_size().unwrap(), 7776);
    let valid: HashSet<String> =
        sp.state_space().unwrap().into_iter().filter(|s| !o.breaks_invariant(s)).map(|s| sp.state_key(&s)).collect();
    let boards: HashSet<String> = all_boards().iter().map(|b| sp.state_key(&state(b))).collect();
    assert_eq!(valid, boards);
}

#[test]
fn fifteen_reaches_half_the_boards_like_the_real_puzzle() {
    // On a board 3 wide every slide keeps the tiles' inversion parity, so exactly half are reachable.
    let sp = spec();
    let o = Oracle::new(&sp);
    let actions = sp.action_instances().unwrap();
    let mut seen: HashSet<String> = HashSet::from([sp.state_key(&sp.initial)]);
    let mut queue = VecDeque::from([sp.initial.clone()]);
    while let Some(s) = queue.pop_front() {
        for a in &actions {
            let n = o.step(&s, a).unwrap();
            assert!(o.violations(&s, Some(a), &n, 0.01).is_empty());
            if seen.insert(sp.state_key(&n)) {
                queue.push_back(n);
            }
        }
    }
    assert_eq!(seen.len(), 360);
    assert!(seen.contains(&sp.state_key(&state(&board("12345_")))));
    assert!(!seen.contains(&sp.state_key(&state(&board("21345_")))), "two tiles swapped is the unsolvable half");
}

#[test]
fn fifteen_four_drags_solve_the_initial_board() {
    let sp = spec();
    let start = board("152_43");
    assert_eq!(sp.initial, state(&start));
    // Drag each tile from its cell onto the blank's: 4, then 5 (no field of its own), then 2, then 3.
    let mut b = start;
    for (piece, gesture) in [(4, "drag_4"), (5, "drag_5"), (2, "drag_2"), (3, "drag_3")] {
        let (from, to) = (px(cell_of(&b, piece)), px(cell_of(&b, BLANK)));
        let mid = ((from.0 + to.0) / 2.0, (from.1 + to.1) / 2.0);
        let (frames, fin) = drag(&sp, gesture, state(&b), &[from, mid, to]).unwrap();
        let (i, j) = (cell_of(&b, piece), cell_of(&b, BLANK));
        b.swap(i, j);
        assert_eq!(fin, state(&b), "{gesture}");
        assert_eq!(frames.len(), 3);
    }
    assert_eq!(b, board("12345_"));
}

#[test]
fn fifteen_drags_without_the_blank_nearby_stay_put() {
    let sp = spec();
    let b = board("152_43");
    let blank = px(cell_of(&b, BLANK));
    // Tile 2 is two cells from the blank: pulled right onto it, it stays.
    let (_, fin) = drag(&sp, "drag_2", state(&b), &[px(cell_of(&b, 2)), blank]).unwrap();
    assert_eq!(fin, state(&b));
    // The blank dragged onto tile 1 above it trades places; dragged diagonally onto 5 it does not.
    let (_, fin) = drag(&sp, "drag_blank", state(&b), &[blank, px(cell_of(&b, 1))]).unwrap();
    assert_eq!(fin, state(&board("_52143")));
    let (_, fin) = drag(&sp, "drag_blank", state(&b), &[blank, px(cell_of(&b, 5))]).unwrap();
    assert_ne!(fin, state(&board("1_2543")), "5 is diagonal to the blank");
    assert!(dragology_futures(&b, BLANK).iter().any(|f| state(f) == fin), "it lands on a neighbour instead");
}

#[test]
fn fifteen_passes_its_own_self_check() {
    let sp = spec();
    let problems = Oracle::new(&sp).self_check(&mut rand::rngs::StdRng::seed_from_u64(0)).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}
