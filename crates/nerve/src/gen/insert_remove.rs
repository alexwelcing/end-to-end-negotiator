//! Dragology's `insert-and-remove` demo: a list of fruit tiles under a toolbar. Dragging a list tile
//! lifts it out and offers every insert position in the rest of the list, or the trash (it is
//! removed); dragging a toolbar tile offers every insert position in the list, or the trash (nothing
//! is added). Far from all of those, the drag falls back to the state it started in (`whenFar`).
//! Scaled down: four tiles with fixed identities (a and b apples, c a banana, d grapes); a tile's field
//! is its list index, or OUT: off the list, drawn in its own toolbar slot, which is both Dragology's
//! store and its trash. `move_<tile>(to)` is the drop.

use super::{generated_by, l, text};

const TILES: [char; 4] = ['a', 'b', 'c', 'd'];
const START: [char; 3] = ['a', 'b', 'c'];
const OUT: usize = TILES.len();
const TILE_PX: usize = 50;
const X0: usize = 30;
const TOOLBAR_Y: usize = 30;
const LIST_Y: usize = 110;
const FAR: usize = 50;

fn label(t: char) -> &'static str {
    match t {
        'c' => "banana",
        'd' => "grapes",
        _ => "apple",
    }
}

fn p(t: char) -> String {
    format!("p{t}")
}

/// How many tiles (other than `skip`) are in the list.
fn in_list_count(skip: Option<char>) -> String {
    let terms: Vec<String> = TILES.iter().filter(|&&t| Some(t) != skip).map(|&t| format!("int({} < {OUT})", p(t))).collect();
    format!("({})", terms.join(" + "))
}

/// Index of tile `x` after `move_<moved>(to)`: close the gap `moved` leaves, then make room at `to`.
fn update(moved: char, x: char) -> String {
    if x == moved {
        return l("to");
    }
    let (px, pm) = (p(x), p(moved));
    let closed = format!("({px} - 1 if {pm} < {px} < {OUT} else {px})");
    format!("{closed} + 1 if to <= {closed} < {OUT} else {closed}")
}

/// The other tiles stay in (or out of) the list, and those in it keep their order.
fn others_unchanged(moved: char) -> String {
    let rest: Vec<char> = TILES.iter().copied().filter(|&t| t != moved).collect();
    let mut terms: Vec<String> = rest.iter().map(|&t| format!("({} == {OUT}) == (old.{} == {OUT})", p(t), p(t))).collect();
    for (i, &x) in rest.iter().enumerate() {
        for &y in &rest[i + 1..] {
            terms.push(format!("(old.{} == {OUT} or old.{} == {OUT} or ({} < {}) == (old.{} < old.{}))", p(x), p(y), p(x), p(y), p(x), p(y)));
        }
    }
    terms.join(" and ")
}

/// (tuples, lists): every index tuple, and the packed injective ones.
fn counts() -> (usize, usize) {
    let n = TILES.len();
    // Lists: ordered selections of any length from n tiles.
    let lists: usize = (0..=n).map(|k| (0..k).map(|i| n - i).product::<usize>()).sum();
    ((OUT + 1).pow(n as u32), lists)
}

fn fruit(t: char) -> &'static str {
    match t {
        'c' => "{rect: [0, 0, 26, 10], r: 5, color: f6}",
        'd' => "{circle: [-5, -2, 5], color: f3}, {circle: [5, -2, 5], color: f3}, {circle: [0, 6, 5], color: f3}, {rect: [0, -10, 3, 6], r: 1, color: f1}",
        _ => "{circle: [0, 2, 11], color: f4}, {rect: [0, -10, 3, 6], r: 1, color: f1}",
    }
}

pub fn spec() -> String {
    let (tuples, boards) = counts();
    let mut distinct = Vec::new();
    for (i, &x) in TILES.iter().enumerate() {
        for &y in &TILES[i + 1..] {
            distinct.push(format!("({} == {OUT} or {} != {})", p(x), p(x), p(y)));
        }
    }
    let packed: Vec<String> = TILES.iter().map(|&t| format!("({} == {OUT} or {} < {})", p(t), p(t), in_list_count(None))).collect();
    let initial = |t: char| START.iter().position(|&s| s == t).unwrap_or(OUT);
    let mut lines = vec![
        l("# Dragology's insert-and-remove: fruit tiles in a list under a toolbar. Drag a tile to any place"),
        l("# in the list (the rest make room) or to the trash; pull one from the toolbar into any place, or"),
        l("# let it go back. Far from every option (whenFar, 50 px) the drag falls back to where it began."),
        l("# Scaled down: four tiles with fixed identities instead of an unbounded list and a toolbar that"),
        l("# mints new ones. a, b = apple and c = banana (Dragology's starting list), d = grapes. A tile's"),
        format!("# field is its list index, or {OUT} = out of the list, drawn in its own toolbar slot: that slot is both"),
        l("# Dragology's store (pull from it) and its trash (drop on it). The invariants say the tiles in the"),
        format!("# list are packed from index 0 with none sharing one, so {boards} of the {tuples} index tuples are"),
        l("# lists. `move_<tile>(to)` lifts the tile out (the ones after it close up) and inserts it at `to`."),
        generated_by("insert_remove"),
        l("nerve: 1"),
        l("name: insert_remove"),
        l("state:"),
    ];
    lines.extend(TILES.iter().map(|&t| format!("  {}: {{type: int, range: [0, {OUT}]}}   # list index of tile {t} ({}); {OUT} = in the toolbar", p(t), label(t))));
    lines.push(l("actions:"));
    lines.extend(TILES.iter().map(|&t| format!("  move_{t}: {{params: {{to: {{type: int, range: [0, {OUT}]}}}}}}   # drop {t} at list index `to`, or {OUT}: remove it")));
    lines.push(l("transitions:"));
    for &t in &TILES {
        lines.push(format!("  move_{t}:"));
        // an insert index may be at most the length of the list without this tile
        lines.push(format!("    when: \"to == {OUT} or to <= {}\"", in_list_count(Some(t))));
        lines.extend(TILES.iter().map(|&x| format!("    {}: \"{}\"", p(x), update(t, x))));
        // The event a machine can wire: a tile went back to the toolbar and the list got shorter.
        let before: Vec<String> = TILES.iter().map(|&x| format!("int(old.{} < {OUT})", p(x))).collect();
        lines.push(format!("    emits: {{ shrank: \"{} < ({})\" }}", in_list_count(None), before.join(" + ")));
    }
    lines.push(l("invariants:"));
    lines.push(format!("  - \"{}\"   # no two tiles share an index", distinct.join(" and ")));
    lines.push(format!("  - \"{}\"   # the list is packed from index 0", packed.join(" and ")));
    lines.push(l("ensures:"));
    lines.extend(TILES.iter().map(|&t| format!("  - {{ after: move_{t}, holds: \"{} == to or {} == old.{}\" }}   # the tile lands where it was dropped, or stays", p(t), p(t), p(t))));
    lines.extend(TILES.iter().map(|&t| format!("  - {{ after: move_{t}, holds: \"{}\" }}   # nothing else joins or leaves, and the rest keep their order", others_unchanged(t))));
    lines.push(l("scene:"));
    lines.push(l("  box: [-10, -12, 240, 150]"));
    lines.push(l("  shapes:"));
    lines.push(format!("    - {{rect: [105, {TOOLBAR_Y}, 220, 48], r: 8, color: panel, stroke: line, width: 1.5}}"));
    lines.push(format!("    - {{rect: [105, {LIST_Y}, 220, 48], r: 8, color: panel, stroke: line, width: 1.5}}"));
    lines.push(l("    - {label: toolbar, at: [105, -2], color: mut, size: 1.8}"));
    lines.push(l("    - {label: list, at: [105, 78], color: mut, size: 1.8}"));
    for k in 0..TILES.len() {
        let x = X0 + TILE_PX * k;
        lines.push(format!("    - {{rect: [{x}, {TOOLBAR_Y}, 40, 40], r: 6, color: none, stroke: line, width: 1.5}}"));
        lines.push(format!("    - {{rect: [{x}, {LIST_Y}, 40, 40], r: 6, color: none, stroke: line, width: 1.5}}"));
    }
    lines.push(l("render:"));
    lines.extend(TILES.iter().enumerate().map(|(k, &t)| {
        format!(
            "  {t}: {{ x: \"{X0} + {TILE_PX} * ({k} if {} == {OUT} else {})\", y: \"{TOOLBAR_Y} if {} == {OUT} else {LIST_Y}\", shapes: [{{rect: [0, 0, 40, 40], r: 8, color: panel, stroke: ink, width: 1.5}}, {}] }}",
            p(t),
            p(t),
            p(t),
            fruit(t)
        )
    }));
    lines.push(l("futures:"));
    lines.push(l("  # one gesture per tile: every list index it can be dropped at, or its toolbar slot (out)"));
    lines.extend(TILES.iter().map(|&t| format!("  drag_{t}: {{ kind: closest, over: {}, grab: {t}, action: move_{t}, stickiness: 6, far: {FAR}, floating: true }}", p(t))));
    lines.push(format!("initial: {{{}}}", TILES.iter().map(|&t| format!("{}: {}", p(t), initial(t))).collect::<Vec<_>>().join(", ")));
    text(lines)
}
