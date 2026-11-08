//! Dragology's `sokoban` demo: drag the player one cell left, right, up or down onto floor; a box in
//! the way is pushed one cell further, but only onto floor that has no box on it. Only the player is
//! draggable. Scaled down: this spec keeps the top three rows of Dragology's level (its first two
//! boxes and two goals, player where Dragology puts it) and walls them off below. The level is the
//! `LEVEL` string in Dragology's own notation (`#` wall, `p` player, `b` box, `g` goal, `B` box on a
//! goal); everything else (the cell grid, the guards, the render) is derived from it here.

use super::{commas, generated_by, l, text};
use std::collections::HashMap;

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

struct Gen {
    lv: Level,
    n: usize,
    boxes: Vec<String>,
}

impl Gen {
    fn new() -> Self {
        let lv = parse(LEVEL);
        let n = lv.w * lv.h;
        let boxes = (1..=lv.boxes.len()).map(|i| format!("b{i}")).collect();
        Self { lv, n, boxes }
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
        let w = self.lv.w;
        if dx < 0 {
            format!("p % {w} >= {k}")
        } else if dx > 0 {
            format!("p % {w} <= {}", w - 1 - k)
        } else if dy < 0 {
            format!("p >= {}", w * k)
        } else {
            format!("p <= {}", self.n - 1 - w * k)
        }
    }

    fn shifted(&self, k: i64) -> String {
        match k {
            0 => l("p"),
            k if k > 0 => format!("p + {k}"),
            k => format!("p - {}", -k),
        }
    }

    /// Cell `p + k` is floor (not a wall); walls that `p + k` could only hit from outside the grid are left out.
    fn floor_at(&self, k: i64) -> String {
        let terms: Vec<String> = self.lv.walls.iter().map(|&w| w as i64 - k).filter(|&c| c >= 0 && c < self.n as i64).map(|c| format!("p != {c}")).collect();
        if terms.is_empty() { l("True") } else { terms.join(" and ") }
    }

    fn box_at(&self, cell: &str) -> String {
        format!("({})", self.boxes.iter().map(|b| format!("{b} == {cell}")).collect::<Vec<_>>().join(" or "))
    }

    fn no_box_at(&self, cell: &str) -> String {
        self.boxes.iter().map(|b| format!("{b} != {cell}")).collect::<Vec<_>>().join(" and ")
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

    fn box_update(&self, b: &str) -> String {
        self.by_dir(&|dx, dy| format!("({} if {b} == {} else {b})", self.shifted(2 * self.offset(dx, dy)), self.shifted(self.offset(dx, dy))))
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
}

pub fn spec() -> String {
    let g = Gen::new();
    let (tuples, valid) = g.counts();
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
    lines.push(l("# not state). Boxes b1, b2 are Dragology's box objects in reading order. One drag is one step:"));
    lines.push(l("# Dragology's `chain: true` (keep walking within one drag) has no Nerve counterpart yet."));
    lines.push(format!("# {} cell tuples, {} of them boards (every object on floor, none sharing a cell).", commas(tuples), commas(valid)));
    lines.push(generated_by("sokoban"));
    lines.push(l("nerve: 1"));
    lines.push(l("name: sokoban"));
    lines.push(l("state:"));
    lines.push(format!("  p: {{type: int, range: [0, {}]}}    # player's cell", n - 1));
    lines.extend(g.boxes.iter().map(|b| format!("  {b}: {{type: int, range: [0, {}]}}   # cell of box {}", n - 1, &b[1..])));
    lines.push(l("actions:"));
    lines.push(format!("  move: {{params: {{dir: {{type: enum, values: [{}]}}}}}}", DIRS.iter().map(|d| d.0).collect::<Vec<_>>().join(", ")));
    lines.push(l("transitions:"));
    lines.push(l("  move:"));
    // Dragology offers only the directions this guard allows; any other `dir` is a no-op.
    lines.push(format!("    when: \"{}\"", g.by_dir(&|dx, dy| g.guard(dx, dy))));
    lines.push(format!("    p: \"{}\"", g.by_dir(&|dx, dy| g.shifted(g.offset(dx, dy)))));
    lines.extend(g.boxes.iter().map(|b| format!("    {b}: \"{}\"", g.box_update(b))));
    lines.push(l("invariants:"));
    lines.push(format!("  - \"{}\"   # nothing stands on a wall", objects.iter().map(|f| g.on_floor(f)).collect::<Vec<_>>().join(" and ")));
    lines.push(format!("  - \"{}\"   # one object per cell", distinct.join(" and ")));
    lines.push(l("ensures:"));
    lines.push(format!("  - {{ after: move, holds: \"p == old.p or (p - old.p == ({step}) and (p // {w} == old.p // {w} or p % {w} == old.p % {w}))\" }}   # one step the dragged way, never wrapping a row"));
    lines.extend(g.boxes.iter().map(|b| format!("  - {{ after: move, holds: \"{b} == old.{b} or ({b} - old.{b} == p - old.p and p == old.{b})\" }}   # {b} moves only when pushed, the way the player went")));
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
    let mut init = vec![format!("p: {}", g.lv.player)];
    init.extend(g.boxes.iter().zip(&g.lv.boxes).map(|(b, c)| format!("{b}: {c}")));
    lines.push(format!("initial: {{{}}}", init.join(", ")));
    text(lines)
}
