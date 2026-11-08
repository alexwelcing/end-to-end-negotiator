//! Dragology's `kanban` demo: four cards in three columns (to do, doing, done). There is no list type
//! yet (ROADMAP 7), so each card has one slot, column * 4 + position, and invariants say no two cards
//! share a slot and every column is packed from the top. Of the 12^4 = 20,736 slot tuples, 360 are
//! boards. `move_<card>(to)` takes the card out of its column (the cards below it close the gap) and
//! inserts it at slot `to` (the cards at and below that position make room). Each card has its own
//! action so that dragging a card means moving that card, not moving another one out of its way.

use super::{generated_by, l, text};

const CARDS: [char; 4] = ['a', 'b', 'c', 'd'];
const COLS: usize = 3;
const DEPTH: usize = 4;
const SLOTS: usize = COLS * DEPTH;

fn s(c: char) -> String {
    format!("s{c}")
}

/// How many cards (other than `skip`) sit in column `col`.
fn count_in_column(col: &str, skip: Option<char>) -> String {
    let terms: Vec<String> = CARDS.iter().filter(|&&c| Some(c) != skip).map(|&c| format!("int({} // {DEPTH} == {col})", s(c))).collect();
    format!("({})", terms.join(" + "))
}

/// Slot of card `x` after `move_<card>(to)`: close the gap, then make room.
fn update(card: char, x: char) -> String {
    if x == card {
        return l("to");
    }
    let (frm, sx) = (s(card), s(x));
    let closed = format!("({sx} - 1 if {sx} // {DEPTH} == {frm} // {DEPTH} and {sx} > {frm} else {sx})");
    format!("{closed} + 1 if {closed} // {DEPTH} == to // {DEPTH} and {closed} >= to else {closed}")
}

pub fn spec() -> String {
    let mut distinct = Vec::new();
    for (i, &x) in CARDS.iter().enumerate() {
        for &y in &CARDS[i + 1..] {
            distinct.push(format!("{} != {}", s(x), s(y)));
        }
    }
    let packed: Vec<String> = CARDS.iter().map(|&x| format!("{} % {DEPTH} < {}", s(x), count_in_column(&format!("{} // {DEPTH}", s(x)), None))).collect();
    let mut lines = vec![
        l("# Dragology's kanban board: four cards in three columns (to do, doing, done). A card's slot is"),
        l("# column * 4 + position. The invariants say no two cards share a slot and every column is packed"),
        l("# from the top, so 360 of the 20,736 slot tuples are boards. `move_<card>(to)` lifts the card out"),
        l("# (the cards below close the gap) and drops it at `to` (the cards there make room)."),
        generated_by("kanban"),
        l("nerve: 1"),
        l("name: kanban"),
        l("state:"),
    ];
    lines.extend(CARDS.iter().map(|&c| format!("  {}: {{type: int, range: [0, {}]}}   # slot of card {c}", s(c), SLOTS - 1)));
    lines.push(l("actions:"));
    lines.extend(CARDS.iter().map(|&c| format!("  move_{c}: {{params: {{to: {{type: int, range: [0, {}]}}}}}}", SLOTS - 1)));
    lines.push(l("transitions:"));
    for &c in &CARDS {
        lines.push(format!("  move_{c}:"));
        // the drop position may be at most one past the other cards already in that column
        lines.push(format!("    when: \"to % {DEPTH} <= {}\"", count_in_column(&format!("to // {DEPTH}"), Some(c))));
        lines.extend(CARDS.iter().map(|&x| format!("    {}: \"{}\"", s(x), update(c, x))));
    }
    lines.push(l("invariants:"));
    lines.push(format!("  - \"{}\"", distinct.join(" and ")));
    lines.push(format!("  - \"{}\"", packed.join(" and ")));
    lines.push(l("ensures:"));
    lines.extend(CARDS.iter().map(|&c| format!("  - {{ after: move_{c}, holds: \"{} == to or {} == old.{}\" }}   # the card lands where it was dropped, or stays", s(c), s(c), s(c))));
    lines.push(l("scene:"));
    lines.push(l("  box: [-10, 0, 435, 245]"));
    lines.push(l("  shapes:"));
    for (i, head) in ["to do", "doing", "done"].iter().enumerate() {
        lines.push(format!("    - {{rect: [{}, 122, 132, 238], r: 10, color: panel, stroke: line, width: 1.5}}", 60 + 150 * i));
        lines.push(format!("    - {{label: {head}, at: [{}, 16], color: mut, size: 1.8}}", 60 + 150 * i));
    }
    lines.push(l("render:"));
    lines.extend(CARDS.iter().map(|&c| {
        format!("  {c}: {{ x: \"60 + 150 * ({} // {DEPTH})\", y: \"50 + 46 * ({} % {DEPTH})\", shapes: [{{rect: [0, 0, 112, 36], r: 8, color: panel, stroke: ink, width: 1.5}}, {{label: {c}, at: [0, 0], color: ink, size: 2.6}}] }}", s(c), s(c))
    }));
    lines.push(l("futures:"));
    lines.push(l("  # one gesture per card: every slot it can be dropped into, in any column"));
    lines.extend(CARDS.iter().map(|&c| format!("  drag_{c}: {{ kind: closest, over: {}, grab: {c}, action: move_{c}, stickiness: 6, floating: true }}", s(c))));
    // a, b in "to do"; c in "doing"; d in "done"
    lines.push(l("initial: {sa: 0, sb: 1, sc: 4, sd: 8}"));
    text(lines)
}
