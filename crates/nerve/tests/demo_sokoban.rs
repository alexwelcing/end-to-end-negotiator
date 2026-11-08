//! Dragology's `sokoban` as `specs/sokoban.nerve`: drag the player one cell onto floor, pushing a box in the
//! way one cell further if that cell is empty floor. The rule is re-implemented here on (x, y) positions, as
//! `sokoban.tsx` writes it, and every drag the spec offers is compared with it.

use nerve::drag::{drag, element_position, DragSession};
use nerve::oracle::Oracle;
use nerve::spec::{Spec, State};
use nerve::value::Value;
use rand::SeedableRng;

/// The level the spec encodes (the top three rows of Dragology's), in Dragology's notation.
const LEVEL: &str = "#######\n###   #\n#gpb  #\n### bg#\n#######";

/// Dragology's four drag directions, in the order `sokoban.tsx` lists them.
const DIRS: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

type Pos = (i64, i64);

fn spec() -> Spec {
    Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/sokoban.nerve")).unwrap()
}

/// Dragology's state: the player and the boxes, by id (box 0 is the first in reading order).
#[derive(Debug, Clone, PartialEq)]
struct Board {
    player: Pos,
    boxes: Vec<Pos>,
}

struct Level {
    cells: Vec<(Pos, char)>,
    /// The interior's top-left cell and width: the spec numbers cells `(y - y0) * w + (x - x0)`.
    x0: i64,
    y0: i64,
    w: i64,
}

impl Level {
    fn parse() -> Self {
        let cells: Vec<(Pos, char)> = LEVEL
            .lines()
            .enumerate()
            .flat_map(|(y, row)| row.chars().enumerate().map(move |(x, ch)| ((x as i64, y as i64), ch)))
            .collect();
        let inner: Vec<Pos> = cells.iter().filter(|(_, ch)| *ch != '#').map(|(p, _)| *p).collect();
        let x0 = inner.iter().map(|p| p.0).min().unwrap();
        let y0 = inner.iter().map(|p| p.1).min().unwrap();
        let w = inner.iter().map(|p| p.0).max().unwrap() - x0 + 1;
        Self { cells, x0, y0, w }
    }

    fn marked(&self, marks: &str) -> Vec<Pos> {
        self.cells.iter().filter(|(_, ch)| marks.contains(*ch)).map(|(p, _)| *p).collect()
    }

    /// Dragology's `isFloor`: in bounds and not a wall.
    fn is_floor(&self, p: Pos) -> bool {
        self.cells.iter().any(|&(q, ch)| q == p && ch != '#')
    }

    fn start(&self) -> Board {
        Board { player: self.marked("p")[0], boxes: self.marked("bB") }
    }

    fn cell(&self, p: Pos) -> Value {
        Value::Int((p.1 - self.y0) * self.w + (p.0 - self.x0))
    }

    fn pos(&self, v: &Value) -> Pos {
        let c = v.as_f64().unwrap() as i64;
        (self.x0 + c % self.w, self.y0 + c / self.w)
    }

    /// The spec's `[p, b1, b2]` for a board.
    fn state(&self, b: &Board) -> State {
        std::iter::once(b.player).chain(b.boxes.iter().copied()).map(|p| self.cell(p)).collect()
    }

    fn board(&self, s: &State) -> Board {
        Board { player: self.pos(&s[0]), boxes: s[1..].iter().map(|v| self.pos(v)).collect() }
    }

    /// `sokoban.tsx`'s player drag, one `d.between([state, next])` per direction that passes its checks.
    fn dragology_moves(&self, b: &Board) -> Vec<Board> {
        let box_at = |p: Pos| b.boxes.iter().position(|&q| q == p);
        let mut out = Vec::new();
        for (dx, dy) in DIRS {
            let next = (b.player.0 + dx, b.player.1 + dy);
            if !self.is_floor(next) {
                continue;
            }
            let Some(i) = box_at(next) else {
                out.push(Board { player: next, ..b.clone() });
                continue;
            };
            let beyond = (next.0 + dx, next.1 + dy);
            if !self.is_floor(beyond) || box_at(beyond).is_some() {
                continue;
            }
            let mut pushed = Board { player: next, ..b.clone() };
            pushed.boxes[i] = beyond;
            out.push(pushed);
        }
        out
    }

    /// Every state the drag can end in: each `between`'s far end, or its near end (the start).
    fn dragology_futures(&self, b: &Board) -> Vec<State> {
        let mut want: Vec<State> = std::iter::once(b.clone()).chain(self.dragology_moves(b)).map(|x| self.state(&x)).collect();
        sort(&mut want);
        want
    }
}

fn sort(states: &mut Vec<State>) {
    states.sort_by_key(|s| s.iter().map(|v| v.as_f64().unwrap() as i64).collect::<Vec<_>>());
    states.dedup();
}

fn candidate_states(sp: &Spec, start: &State) -> Vec<State> {
    let sess = DragSession::new(sp, sp.future("drag_player").unwrap(), start.clone(), None).unwrap();
    for c in sess.candidates() {
        assert!(c.action.as_ref().is_none_or(|a| a.name == "move"), "{:?}", c.action);
    }
    let mut got: Vec<State> = sess.candidates().iter().map(|c| c.state.clone()).collect();
    sort(&mut got);
    got
}

#[test]
fn the_spec_encodes_the_level_it_says() {
    let (sp, lv) = (spec(), Level::parse());
    assert_eq!(sp.initial, lv.state(&lv.start()), "player and boxes start where the level puts them");
    let oracle = Oracle::new(&sp);
    let space = sp.state_space().unwrap();
    assert_eq!(space.len(), 15 * 15 * 15);
    let boards: Vec<&State> = space.iter().filter(|s| !oracle.breaks_invariant(s)).collect();
    assert_eq!(boards.len(), 11 * 10 * 9, "three distinct objects on the eleven floor cells");
    for s in boards {
        let b = lv.board(s);
        assert!(std::iter::once(b.player).chain(b.boxes).all(|p| lv.is_floor(p)), "{s:?} stands on a wall");
    }
}

#[test]
fn player_futures_are_dragologys_moves_from_named_boards() {
    let (sp, lv) = (spec(), Level::parse());
    let named = [
        // the start: left onto the goal, or right pushing box 1; up and down are walls
        (lv.start(), 3),
        // box 1 against the right wall, box 2 against the bottom wall: neither push is allowed
        (Board { player: (4, 2), boxes: vec![(5, 2), (4, 3)] }, 3),
        // box 1 would be pushed into box 2: only up is open
        (Board { player: (3, 3), boxes: vec![(4, 3), (5, 3)] }, 2),
        // pushing box 1 up into the top row is allowed; box 2 is against the right wall
        (Board { player: (4, 3), boxes: vec![(4, 2), (5, 3)] }, 3),
        // top-right corner: only left and down
        (Board { player: (5, 1), boxes: vec![(3, 2), (4, 3)] }, 3),
    ];
    for (b, n) in named {
        let start = lv.state(&b);
        let want = lv.dragology_futures(&b);
        assert_eq!(want.len(), n, "the test's own count for {b:?}");
        assert_eq!(candidate_states(&sp, &start), want, "dragging the player on {b:?}");
    }
}

#[test]
fn player_futures_are_dragologys_moves_on_every_board() {
    let (sp, lv) = (spec(), Level::parse());
    let oracle = Oracle::new(&sp);
    let mut checked = 0;
    for s in sp.state_space().unwrap() {
        if oracle.breaks_invariant(&s) {
            continue;
        }
        assert_eq!(candidate_states(&sp, &s), lv.dragology_futures(&lv.board(&s)), "from {s:?}");
        checked += 1;
    }
    assert_eq!(checked, 990);
}

/// Pointer path from the player's drawn position to the neighbouring cell `(dx, dy)`, 60 px away.
fn path_towards(sp: &Spec, s: &State, (dx, dy): (i64, i64)) -> Vec<(f64, f64)> {
    let (x, y) = element_position(sp, "player", s).unwrap();
    let (tx, ty) = (x + 60.0 * dx as f64, y + 60.0 * dy as f64);
    vec![(x, y), ((x + tx) / 2.0 + 3.0, (y + ty) / 2.0 + 3.0), (tx + 2.0, ty - 2.0)]
}

#[test]
fn dragging_the_player_into_a_box_pushes_it() {
    let (sp, lv) = (spec(), Level::parse());
    let (frames, fin) = drag(&sp, "drag_player", sp.initial.clone(), &path_towards(&sp, &sp.initial, (1, 0))).unwrap();
    let want = Board { player: (3, 2), boxes: vec![(4, 2), (4, 3)] };
    assert_eq!(fin, lv.state(&want));
    assert_eq!(frames.last().unwrap().branch, Some(Value::Int(7)), "the branch is the player's new cell");
    // Dragging up from the start goes nowhere: that cell is a wall.
    let (_, fin) = drag(&sp, "drag_player", sp.initial.clone(), &path_towards(&sp, &sp.initial, (0, -1))).unwrap();
    assert_eq!(fin, sp.initial);
}

#[test]
fn the_level_is_solved_by_drags_alone() {
    let (sp, lv) = (spec(), Level::parse());
    let (l, r, u, d) = (DIRS[0], DIRS[1], DIRS[2], DIRS[3]);
    let mut s = sp.initial.clone();
    for dir in [r, d, r, l, u, u, r, r, d, l, l, l] {
        let b = lv.board(&s);
        let next = lv.dragology_moves(&b).into_iter().find(|n| n.player == (b.player.0 + dir.0, b.player.1 + dir.1));
        let next = next.unwrap_or_else(|| panic!("Dragology has no move {dir:?} from {b:?}"));
        let (_, fin) = drag(&sp, "drag_player", s.clone(), &path_towards(&sp, &s, dir)).unwrap();
        assert_eq!(fin, lv.state(&next), "drag {dir:?} from {b:?}");
        s = fin;
    }
    let mut boxes = lv.board(&s).boxes;
    boxes.sort();
    let mut goals = lv.marked("gB");
    goals.sort();
    assert_eq!(boxes, goals, "both boxes end on goals");
}

#[test]
fn spec_passes_its_own_check() {
    let sp = spec();
    let problems = Oracle::new(&sp).self_check(&mut rand::rngs::StdRng::seed_from_u64(0)).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}
