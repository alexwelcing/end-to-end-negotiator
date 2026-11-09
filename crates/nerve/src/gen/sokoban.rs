//! Dragology's `sokoban` demo: drag the player one cell left, right, up or down onto floor; a box in
//! the way is pushed one cell further, but only onto floor that has no box on it. Only the player is
//! draggable. Scaled down: this spec keeps the top three rows of Dragology's level (its first two
//! boxes and two goals, player where Dragology puts it) and walls them off below. The level is the
//! `LEVEL` string in Dragology's own notation (`#` wall, `p` player, `b` box, `g` goal, `B` box on a
//! goal); everything else (the cell grid, the guards, the render) is derived from it here.
//!
//! As in `hanoi`, a `learn` hole on the action `hint` gives the next move of a shortest solution only
//! by example: one per board that can still be solved, found here by a breadth-first search over the
//! same rule the `move` transition writes. A machine pulses `hint` every tick and wires `solved`, the
//! event `move` and `hint` emit when both boxes come to stand on goals.

use super::{commas, generated_by, l, text};
use std::collections::{BTreeMap, HashMap, VecDeque};

const LEVEL: &str = "#######\n###   #\n#gpb  #\n### bg#\n#######";
/// Dragology's moves, as (name, dx, dy), in the order its drag lists them.
const DIRS: [(&str, i64, i64); 4] = [("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1)];
const TILE: usize = 60;
const ORIGIN: usize = 50;

struct Level {
    w: usize,
    h: usize,
    walls: Vec<usize>,
    player: usize,
    boxes: Vec<usize>,
    goals: Vec<usize>,
}

fn parse(level: &str) -> Level {
    let rows: Vec<&str> = level.split('\n').collect();
    let mut cells: HashMap<(i64, i64), char> = HashMap::new();
    for (y, row) in rows.iter().enumerate() {
        for (x, ch) in row.chars().enumerate() {
            assert!("#pbgB ".contains(ch), "unknown level character");
            cells.insert((x as i64, y as i64), ch);
        }
    }
    let inner: Vec<(i64, i64)> = cells.iter().filter(|(_, &ch)| ch != '#').map(|(&xy, _)| xy).collect();
    let x0 = inner.iter().map(|p| p.0).min().unwrap();
    let y0 = inner.iter().map(|p| p.1).min().unwrap();
    let w = (inner.iter().map(|p| p.0).max().unwrap() - x0 + 1) as usize;
    let h = (inner.iter().map(|p| p.1).max().unwrap() - y0 + 1) as usize;
    // The level must be closed: every interior cell's neighbours exist, so a step off the grid is a step into a wall.
    for &(x, y) in &inner {
        for (_, dx, dy) in DIRS {
            assert!(cells.contains_key(&(x + dx, y + dy)), "level is open at {:?}", (x + dx, y + dy));
        }
    }
    let idx = |x: i64, y: i64| ((y - y0) as usize) * w + (x - x0) as usize;
    let mut walls: Vec<usize> = (y0..y0 + h as i64)
        .flat_map(|y| (x0..x0 + w as i64).map(move |x| (x, y)))
        .filter(|&(x, y)| *cells.get(&(x, y)).unwrap_or(&'#') == '#')
        .map(|(x, y)| idx(x, y))
        .collect();
    walls.sort_unstable();
    let mut reading: Vec<((i64, i64), char)> = cells.iter().map(|(&xy, &ch)| (xy, ch)).collect();
    reading.sort_by_key(|((x, y), _)| (*y, *x));
    let pick = |f: &dyn Fn(char) -> bool| -> Vec<usize> { reading.iter().filter(|(_, ch)| f(*ch)).map(|((x, y), _)| idx(*x, *y)).collect() };
    let player = pick(&|c| c == 'p');
    let boxes = pick(&|c| c == 'b' || c == 'B');
    let goals = pick(&|c| c == 'g' || c == 'B');
    assert!(player.len() == 1 && boxes.len() == goals.len() && !goals.is_empty());
    Level { w, h, walls, player: player[0], boxes, goals }
}

/// One board: the player's cell and each box's cell, in the spec's field order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Board {
    p: usize,
    boxes: Vec<usize>,
}

impl Level {
    fn is_floor(&self, c: usize) -> bool {
        self.walls.binary_search(&c).is_err()
    }

    /// The cell one step from `c` in (dx, dy); None off the grid, which (the level being closed) is a wall.
    fn neighbour(&self, c: usize, dx: i64, dy: i64) -> Option<usize> {
        let (x, y) = ((c % self.w) as i64 + dx, (c / self.w) as i64 + dy);
        let on_grid = (0..self.w as i64).contains(&x) && (0..self.h as i64).contains(&y);
        on_grid.then(|| y as usize * self.w + x as usize)
    }

    /// Dragology's move, the rule the `move` transition writes: the next cell is floor, and either empty
    /// or holding a box whose next cell is empty floor. None when the guard fails and the move is a no-op.
    fn step(&self, b: &Board, dx: i64, dy: i64) -> Option<Board> {
        let next = self.neighbour(b.p, dx, dy).filter(|&c| self.is_floor(c))?;
        let Some(pushed) = b.boxes.iter().position(|&c| c == next) else {
            return Some(Board { p: next, boxes: b.boxes.clone() });
        };
        let beyond = self.neighbour(next, dx, dy).filter(|&c| self.is_floor(c) && !b.boxes.contains(&c))?;
        let boxes = b.boxes.iter().enumerate().map(|(i, &c)| if i == pushed { beyond } else { c }).collect();
        Some(Board { p: next, boxes })
    }

    fn solved(&self, b: &Board) -> bool {
        b.boxes.iter().all(|c| self.goals.contains(c))
    }

    fn initial(&self) -> Board {
        Board { p: self.player, boxes: self.boxes.clone() }
    }

    /// Every board: each object on floor, no two sharing a cell, in state order (p, b1, b2, ...).
    fn boards(&self) -> Vec<Board> {
        let floor: Vec<usize> = (0..self.w * self.h).filter(|&c| self.is_floor(c)).collect();
        let tuples = (0..=self.boxes.len()).fold(vec![Vec::new()], |partial: Vec<Vec<usize>>, _| {
            partial
                .iter()
                .flat_map(|t| {
                    floor.iter().copied().filter(move |c| !t.contains(c)).map(move |c| {
                        let mut longer = t.clone();
                        longer.push(c);
                        longer
                    })
                })
                .collect()
        });
        tuples.into_iter().map(|t| Board { p: t[0], boxes: t[1..].to_vec() }).collect()
    }
}

/// The shortest solutions: for every board that can still be solved, the board after the first move of
/// one, ties broken by DIRS order. Pushes cannot be undone, so the search runs backwards from every
/// solved board over the predecessors of every board.
struct Solution {
    next: BTreeMap<Board, Board>,
    /// Boards from which no solution exists (a box stuck against the walls, or the two blocking each other).
    unsolvable: usize,
    /// Moves in the longest of the shortest solutions, and in the initial board's.
    longest: usize,
    from_initial: usize,
}

fn solve(lv: &Level) -> Solution {
    let boards = lv.boards();
    let mut preds: HashMap<Board, Vec<Board>> = HashMap::new();
    for b in &boards {
        for (_, dx, dy) in DIRS {
            if let Some(n) = lv.step(b, dx, dy) {
                preds.entry(n).or_default().push(b.clone());
            }
        }
    }
    let mut dist: HashMap<Board, usize> = boards.iter().filter(|b| lv.solved(b)).map(|b| (b.clone(), 0)).collect();
    let mut queue: VecDeque<Board> = dist.keys().cloned().collect();
    while let Some(s) = queue.pop_front() {
        let d = dist[&s] + 1;
        for p in preds.get(&s).map(Vec::as_slice).unwrap_or_default() {
            if !dist.contains_key(p) {
                dist.insert(p.clone(), d);
                queue.push_back(p.clone());
            }
        }
    }
    let from_initial = *dist.get(&lv.initial()).expect("the initial board must be solvable");
    let next = boards
        .iter()
        .filter_map(|b| {
            let d = *dist.get(b)?;
            if d == 0 {
                // A solved board's hint is to stay put: a hole action gets no other training signal there.
                return Some((b.clone(), b.clone()));
            }
            let closer = DIRS.iter().filter_map(|&(_, dx, dy)| lv.step(b, dx, dy)).find(|n| dist.get(n) == Some(&(d - 1)));
            Some((b.clone(), closer.expect("a board d moves from solved has a move to one d - 1 away")))
        })
        .collect();
    Solution { next, unsolvable: boards.len() - dist.len(), longest: dist.values().copied().max().unwrap_or(0), from_initial }
}

struct Gen {
    lv: Level,
    n: usize,
    boxes: Vec<String>,
    /// Prefix on every field this generator reads: "" for the state an expression is evaluated in,
    /// "old." for the state before a step (the `ensures` and `emits` relate the two).
    pre: &'static str,
}

impl Gen {
    fn new() -> Self {
        Self::with_prefix("")
    }

    fn with_prefix(pre: &'static str) -> Self {
        let lv = parse(LEVEL);
        let n = lv.w * lv.h;
        let boxes = (1..=lv.boxes.len()).map(|i| format!("b{i}")).collect();
        Self { lv, n, boxes, pre }
    }

    /// The same generator reading the state before a step: every field as `old.<field>`.
    fn old(&self) -> Self {
        Self::with_prefix("old.")
    }

    /// Field `f` as this generator reads it.
    fn var(&self, f: &str) -> String {
        format!("{}{f}", self.pre)
    }

    fn p(&self) -> String {
        self.var("p")
    }

    fn objects(&self) -> Vec<String> {
        let mut v = vec![l("p")];
        v.extend(self.boxes.iter().cloned());
        v
    }

    fn offset(&self, dx: i64, dy: i64) -> i64 {
        dx + self.lv.w as i64 * dy
    }

    /// `p` can take `k` steps in direction (dx, dy) without leaving the grid.
    fn stays_on_grid(&self, dx: i64, dy: i64, k: usize) -> String {
        let (w, p) = (self.lv.w, self.p());
        if dx < 0 {
            format!("{p} % {w} >= {k}")
        } else if dx > 0 {
            format!("{p} % {w} <= {}", w - 1 - k)
        } else if dy < 0 {
            format!("{p} >= {}", w * k)
        } else {
            format!("{p} <= {}", self.n - 1 - w * k)
        }
    }

    fn shifted(&self, k: i64) -> String {
        match k {
            0 => self.p(),
            k if k > 0 => format!("{} + {k}", self.p()),
            k => format!("{} - {}", self.p(), -k),
        }
    }

    /// Cell `p + k` is floor (not a wall); walls that `p + k` could only hit from outside the grid are left out.
    fn floor_at(&self, k: i64) -> String {
        let terms: Vec<String> = self.lv.walls.iter().map(|&w| w as i64 - k).filter(|&c| c >= 0 && c < self.n as i64).map(|c| format!("{} != {c}", self.p())).collect();
        if terms.is_empty() { l("True") } else { terms.join(" and ") }
    }

    fn box_at(&self, cell: &str) -> String {
        format!("({})", self.boxes.iter().map(|b| format!("{} == {cell}", self.var(b))).collect::<Vec<_>>().join(" or "))
    }

    fn no_box_at(&self, cell: &str) -> String {
        self.boxes.iter().map(|b| format!("{} != {cell}", self.var(b))).collect::<Vec<_>>().join(" and ")
    }

    /// Dragology: the next cell is floor, and either empty or holding a box whose next cell is empty floor.
    fn guard(&self, dx: i64, dy: i64) -> String {
        let s = self.offset(dx, dy);
        let (nxt, beyond) = (self.shifted(s), self.shifted(2 * s));
        let push = format!("{} and {} and {}", self.stays_on_grid(dx, dy, 2), self.floor_at(2 * s), self.no_box_at(&beyond));
        format!("({} and {} and (not {} or ({push})))", self.stays_on_grid(dx, dy, 1), self.floor_at(s), self.box_at(&nxt))
    }

    /// `dir`-indexed conditional over DIRS: `a if dir == 'left' else b if ... else d`.
    fn by_dir(&self, per_dir: &dyn Fn(i64, i64) -> String) -> String {
        let mut parts: Vec<String> = DIRS[..DIRS.len() - 1].iter().map(|(name, dx, dy)| format!("{} if dir == '{name}' else", per_dir(*dx, *dy))).collect();
        let (_, dx, dy) = DIRS[DIRS.len() - 1];
        parts.push(per_dir(dx, dy));
        parts.join(" ")
    }

    /// Where box `b` is after a step in (dx, dy): one cell further when the player stepped into it.
    fn box_push(&self, b: &str, dx: i64, dy: i64) -> String {
        let (s, bv) = (self.offset(dx, dy), self.var(b));
        format!("({} if {bv} == {} else {bv})", self.shifted(2 * s), self.shifted(s))
    }

    fn box_update(&self, b: &str) -> String {
        self.by_dir(&|dx, dy| self.box_push(b, dx, dy))
    }

    fn on_floor(&self, f: &str) -> String {
        self.lv.walls.iter().map(|w| format!("{f} != {w}")).collect::<Vec<_>>().join(" and ")
    }

    fn position(&self, f: &str) -> String {
        format!("x: \"{ORIGIN} + {TILE} * ({f} % {})\", y: \"{ORIGIN} + {TILE} * ({f} // {})\"", self.lv.w, self.lv.w)
    }

    /// The interior grid as cell numbers, for the header comment.
    fn picture(&self) -> Vec<String> {
        (0..self.lv.h)
            .map(|y| {
                let row: Vec<String> = (0..self.lv.w).map(|x| { let c = y * self.lv.w + x; if self.lv.walls.contains(&c) { l("##") } else { format!("{c:2}") } }).collect();
                format!("#   {}", row.join(" "))
            })
            .collect()
    }

    fn counts(&self) -> (usize, usize) {
        let floor = self.n - self.lv.walls.len();
        let k = self.objects().len();
        let valid: usize = (0..k).map(|i| floor - i).product();
        (self.n.pow(k as u32), valid)
    }

    fn on_goal(&self, b: &str) -> String {
        self.lv.goals.iter().map(|g| format!("{b} == {g}")).collect::<Vec<_>>().join(" or ")
    }

    /// Every box on a goal, read through this generator's prefix.
    fn all_on_goals(&self) -> String {
        self.boxes.iter().map(|b| format!("({})", self.on_goal(&self.var(b)))).collect::<Vec<_>>().join(" and ")
    }

    /// The event a machine wires: both boxes came to stand on goals (and were not both there before).
    fn solved_emit(&self) -> String {
        format!("    emits: {{ solved: \"{} and not ({})\" }}", self.all_on_goals(), self.old().all_on_goals())
    }

    /// The new state is the old one unchanged (the `hint` fallback), or what `move` makes of it for some `dir`.
    fn legal_single_move(&self) -> String {
        let old = self.old();
        let same: Vec<String> = self.objects().iter().map(|f| format!("{f} == old.{f}")).collect();
        let moves: Vec<String> = DIRS
            .iter()
            .map(|&(_, dx, dy)| {
                let mut terms = vec![old.guard(dx, dy), format!("p == {}", old.shifted(old.offset(dx, dy)))];
                terms.extend(self.boxes.iter().map(|b| format!("{b} == {}", old.box_push(b, dx, dy))));
                format!("({})", terms.join(" and "))
            })
            .collect();
        format!("({}) or {}", same.join(" and "), moves.join(" or "))
    }

    /// A board as a spec state, `{p: 6, b1: 7, b2: 13}`.
    fn board_text(&self, b: &Board) -> String {
        let cells = std::iter::once(b.p).chain(b.boxes.iter().copied());
        format!("{{{}}}", self.objects().iter().zip(cells).map(|(f, c)| format!("{f}: {c}")).collect::<Vec<_>>().join(", "))
    }
}

pub fn spec() -> String {
    let g = Gen::new();
    let (tuples, valid) = g.counts();
    let sol = solve(&g.lv);
    let step = g.by_dir(&|dx, dy| format!("{}", g.offset(dx, dy)));
    let objects = g.objects();
    let mut distinct = Vec::new();
    for (i, x) in objects.iter().enumerate() {
        for y in &objects[i + 1..] {
            distinct.push(format!("{x} != {y}"));
        }
    }
    let (w, n) = (g.lv.w, g.n);
    let mut lines = vec![
        l("# Dragology's sokoban: drag the player one cell onto floor; a box in the way is pushed one cell"),
        l("# further, but only onto floor with no box on it. Only the player is draggable."),
        l("# Scaled down: Dragology's 8 x 9 level with seven boxes is too big to enumerate, so this is its top"),
        l("# three rows (first two boxes and goals, player where Dragology puts it), walled off below:"),
        l("#"),
    ];
    lines.extend(LEVEL.split('\n').map(|row| format!("#   {row}")));
    lines.push(l("#"));
    lines.push(format!("# Each object is one cell number, y * {w} + x over the interior ({w} x {}, `##` = wall):", g.lv.h));
    lines.push(l("#"));
    lines.extend(g.picture());
    lines.push(l("#"));
    lines.push(format!("# Goals: cells {} (the demo draws them and has no win condition, so they are", g.lv.goals.iter().map(usize::to_string).collect::<Vec<_>>().join(" and ")));
    lines.push(l("# not state; `move` and `hint` emit `solved` when both boxes come to stand on them). Boxes b1, b2"));
    lines.push(l("# are Dragology's box objects in reading order. One drag is one step: Dragology's `chain: true`"));
    lines.push(l("# (keep walking within one drag) has no Nerve counterpart yet."));
    lines.push(format!("# {} cell tuples, {} of them boards (every object on floor, none sharing a cell).", commas(tuples), commas(valid)));
    lines.push(l("# `hint` is the next move of a shortest solution, given only by example (one per board that can be"));
    lines.push(format!("# solved, the solved ones staying put: {} examples; the other {} boards cannot be, a box stuck against the walls or the two", commas(sol.next.len()), commas(sol.unsolvable)));
    lines.push(format!("# blocking each other). The longest solution takes {} moves, the initial board's {}.", sol.longest, sol.from_initial));
    lines.push(generated_by("sokoban"));
    lines.push(l("nerve: 1"));
    lines.push(l("name: sokoban"));
    lines.push(l("state:"));
    lines.push(format!("  p: {{type: int, range: [0, {}]}}    # player's cell", n - 1));
    lines.extend(g.boxes.iter().map(|b| format!("  {b}: {{type: int, range: [0, {}]}}   # cell of box {}", n - 1, &b[1..])));
    lines.push(l("actions:"));
    lines.push(format!("  move: {{params: {{dir: {{type: enum, values: [{}]}}}}}}", DIRS.iter().map(|d| d.0).collect::<Vec<_>>().join(", ")));
    lines.push(l("  hint: {}   # the next move of a shortest solution: shown by examples, never written down"));
    lines.push(l("transitions:"));
    lines.push(l("  move:"));
    // Dragology offers only the directions this guard allows; any other `dir` is a no-op.
    lines.push(format!("    when: \"{}\"", g.by_dir(&|dx, dy| g.guard(dx, dy))));
    lines.push(format!("    p: \"{}\"", g.by_dir(&|dx, dy| g.shifted(g.offset(dx, dy)))));
    lines.extend(g.boxes.iter().map(|b| format!("    {b}: \"{}\"", g.box_update(b))));
    lines.push(g.solved_emit());
    // The fallback the `solve` hole is learned on top of: stay put. Its `solved` fires once the learned hint lands the last box.
    lines.push(l("  hint:"));
    lines.extend(objects.iter().map(|f| format!("    {f}: \"{f}\"")));
    lines.push(g.solved_emit());
    lines.push(l("invariants:"));
    lines.push(format!("  - \"{}\"   # nothing stands on a wall", objects.iter().map(|f| g.on_floor(f)).collect::<Vec<_>>().join(" and ")));
    lines.push(format!("  - \"{}\"   # one object per cell", distinct.join(" and ")));
    lines.push(l("ensures:"));
    lines.push(format!("  - {{ after: move, holds: \"p == old.p or (p - old.p == ({step}) and (p // {w} == old.p // {w} or p % {w} == old.p % {w}))\" }}   # one step the dragged way, never wrapping a row"));
    lines.extend(g.boxes.iter().map(|b| format!("  - {{ after: move, holds: \"{b} == old.{b} or ({b} - old.{b} == p - old.p and p == old.{b})\" }}   # {b} moves only when pushed, the way the player went")));
    lines.push(format!("  - {{ after: hint, holds: \"{}\" }}   # a hint may only stay put or make one legal move", g.legal_single_move()));
    lines.push(l("scene:"));
    lines.push(format!("  box: [{}, {}, {}, {}]", ORIGIN - 36, ORIGIN - 36, ORIGIN + TILE * (w - 1) + 36, ORIGIN + TILE * (g.lv.h - 1) + 36));
    lines.push(l("  shapes:"));
    let (bx, by) = (ORIGIN as f64 - 30.0 + TILE as f64 * w as f64 / 2.0, ORIGIN as f64 - 30.0 + TILE as f64 * g.lv.h as f64 / 2.0);
    lines.push(format!("    - {{rect: [{bx}, {by}, {}, {}], r: 6, color: panel, stroke: line, width: 1.5}}", TILE * w, TILE * g.lv.h));
    for i in 1..w {
        let x = ORIGIN - 30 + TILE * i;
        lines.push(format!("    - {{line: [{x}, {}, {x}, {}], color: line, width: 1.5}}", ORIGIN - 30, ORIGIN - 30 + TILE * g.lv.h));
    }
    for j in 1..g.lv.h {
        let y = ORIGIN - 30 + TILE * j;
        lines.push(format!("    - {{line: [{}, {y}, {}, {y}], color: line, width: 1.5}}", ORIGIN - 30, ORIGIN - 30 + TILE * w));
    }
    for &c in &g.lv.walls {
        lines.push(format!("    - {{rect: [{}, {}, 58, 58], r: 2, color: track}}", ORIGIN + TILE * (c % w), ORIGIN + TILE * (c / w)));
    }
    for &c in &g.lv.goals {
        lines.push(format!("    - {{circle: [{}, {}, 14], color: none, stroke: f1, width: 2}}", ORIGIN + TILE * (c % w), ORIGIN + TILE * (c / w)));
    }
    lines.push(l("render:"));
    lines.push(format!("  player: {{ {}, shapes: [{{circle: [0, 0, 17], color: f0, stroke: acc, width: 2}}, {{label: you, at: [0, 0], color: white, size: 1.6}}] }}", g.position("p")));
    lines.extend(g.boxes.iter().map(|b| {
        format!(
            "  box{}: {{ {}, shapes: [{{rect: [0, 0, 44, 44], r: 5, color: f9, when: \"not ({})\"}}, {{rect: [0, 0, 44, 44], r: 5, color: f1, when: \"{}\"}}, {{rect: [0, 0, 44, 44], r: 5, color: none, stroke: ink, width: 1.5}}, {{line: [-14, -14, 14, 14], color: white, width: 2, opacity: 0.5}}, {{line: [14, -14, -14, 14], color: white, width: 2, opacity: 0.5}}] }}",
            &b[1..],
            g.position(b),
            g.on_goal(b),
            g.on_goal(b)
        )
    }));
    lines.push(l("futures:"));
    lines.push(l("  # Dragology: closest over the four neighbouring moves (and staying); the guard decides which exist"));
    lines.push(l("  drag_player: { kind: closest, over: p, grab: player, action: move, stickiness: 6 }"));
    lines.push(l("learn:"));
    lines.push(l("  - name: solve"));
    lines.push(l("    action: hint"));
    lines.push(l("    # the fallback is the `hint` transition above (stay put); a solved board's example is to stay put, unsolvable boards have none"));
    lines.push(l("    examples:"));
    lines.extend(sol.next.iter().map(|(b, nx)| format!("      - {{ state: {}, next: {} }}", g.board_text(b), g.board_text(nx))));
    lines.push(format!("initial: {}", g.board_text(&g.lv.initial())));
    text(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::Oracle;
    use crate::spec::{ActionInst, Spec};
    use crate::value::Value;

    fn state(b: &Board) -> Vec<Value> {
        std::iter::once(b.p).chain(b.boxes.iter().copied()).map(|c| Value::Int(c as i64)).collect()
    }

    #[test]
    fn the_search_moves_the_way_the_move_transition_does() {
        // The examples are only as good as the rule they were searched over: it must be the spec's own.
        let sp = Spec::parse(&spec()).unwrap();
        let o = Oracle::new(&sp);
        let lv = parse(LEVEL);
        let boards = lv.boards();
        assert_eq!(boards.len(), Gen::new().counts().1);
        for b in &boards {
            for (name, dx, dy) in DIRS {
                let a = ActionInst { name: l("move"), params: vec![(l("dir"), Value::Str(name.to_string()))] };
                let want = lv.step(b, dx, dy).unwrap_or_else(|| b.clone());
                assert_eq!(o.step(&state(b), &a).unwrap(), state(&want), "{b:?} {name}");
            }
        }
    }

    #[test]
    fn every_hint_example_is_one_legal_move_closer_to_solved() {
        let sp = Spec::parse(&spec()).unwrap();
        let o = Oracle::new(&sp);
        let lv = parse(LEVEL);
        let sol = solve(&lv);
        assert!(!sol.next.is_empty() && sol.from_initial > 0 && sol.from_initial <= sol.longest);
        let hint = ActionInst { name: l("hint"), params: Vec::new() };
        for (b, nx) in &sol.next {
            if lv.solved(b) {
                assert_eq!(b, nx, "a solved board stays put");
            } else {
                assert!(DIRS.iter().any(|&(_, dx, dy)| lv.step(b, dx, dy).as_ref() == Some(nx)), "{b:?} -> {nx:?} is not one move");
            }
            let v = o.violations(&state(b), Some(&hint), &state(nx), 0.01);
            assert!(v.is_empty(), "{b:?} -> {nx:?}: {v:?}");
        }
        assert_eq!(sol.next.len() + sol.unsolvable, lv.boards().len(), "every board is solvable (with an example) or not");
    }
}
