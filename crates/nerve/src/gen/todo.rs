//! Dragology's `todo` demo (src/demos/todo.tsx): a list of cards, each with a `completed` checkbox.
//! Dragging a card splices it out and back in at any index (`statesWith`); ticking the box flips that
//! card's `completed` and nothing else. The text box and the Add button need a text type, so cards
//! are not added, typed or renamed here. There is no list type yet (ROADMAP 7), so the list is one
//! position per card plus "no two share a place", and one done flag per card: 4! orders x 2^4
//! completions = 384 of the 4^4 * 2^4 = 4,096 tuples are lists. Each card has its own `move_<card>(to)`
//! (so dragging a card moves that card) and its own `toggle_<card>` (the checkbox).

use super::{generated_by, l, text};

const ITEMS: [char; 4] = ['a', 'b', 'c', 'd'];
const DONE: char = 'b';
const TOP: usize = 60;
const ROW: usize = 55;
const LEFT: usize = 60;
const INDENT: usize = 50;

fn p(x: char) -> String {
    format!("p{x}")
}

fn d(x: char) -> String {
    format!("d{x}")
}

/// Position of `x` after `move_<card>(to)`: splice `card` out, then back in at `to`.
fn update(card: char, x: char) -> String {
    if x == card {
        return l("to");
    }
    let (frm, px) = (p(card), p(x));
    format!("{px} - 1 if {frm} < {px} <= to else ({px} + 1 if to <= {px} < {frm} else {px})")
}

fn unchanged(fields: &[String]) -> String {
    fields.iter().map(|f| format!("{f} == old.{f}")).collect::<Vec<_>>().join(" and ")
}

pub fn spec() -> String {
    let n = ITEMS.len();
    let mut distinct = Vec::new();
    for (i, &x) in ITEMS.iter().enumerate() {
        for &y in &ITEMS[i + 1..] {
            distinct.push(format!("{} != {}", p(x), p(y)));
        }
    }
    let all_d: Vec<String> = ITEMS.iter().map(|&y| d(y)).collect();
    let mut lines = vec![
        l("# Dragology's todo list (src/demos/todo.tsx): drag a card to reorder the list, tick its box to complete it."),
        l("# The rule: a drag lifts the card out and splices it back in at any index (the cards in between close up"),
        l("# and make room), and a tick flips that card's `completed` and nothing else. Scaled down: four cards with"),
        l("# no text. Dragology's text box and Add button need a text type, so cards are not added, typed or renamed;"),
        l("# Dragology starts with a, b (completed), c, and d stands in for one more card. With no list type"),
        l("# (ROADMAP 7) a card's place is a position field, so 384 of the 4,096 tuples are lists (24 orders x 16"),
        l("# completions). The drag is Dragology's `useFloating` variant, closest + floating; its `whenFar(stateWithout)`"),
        l("# (the list without the card) is not a state here. A completed card is drawn 50 px to the right."),
        generated_by("todo"),
        l("nerve: 1"),
        l("name: todo"),
        l("state:"),
    ];
    lines.extend(ITEMS.iter().map(|&x| format!("  {}: {{type: int, range: [0, {}]}}   # position of card {x}, top to bottom", p(x), n - 1)));
    lines.extend(ITEMS.iter().map(|&x| format!("  {}: {{type: bool}}   # card {x} is completed", d(x))));
    lines.push(l("actions:"));
    lines.extend(ITEMS.iter().map(|&x| format!("  move_{x}: {{params: {{to: {{type: int, range: [0, {}]}}}}}}   # drag card {x} to index `to`", n - 1)));
    lines.extend(ITEMS.iter().map(|&x| format!("  toggle_{x}: {{}}   # tick card {x}'s checkbox")));
    lines.push(l("transitions:"));
    for &x in &ITEMS {
        lines.push(format!("  move_{x}:"));
        lines.extend(ITEMS.iter().map(|&y| format!("    {}: \"{}\"", p(y), update(x, y))));
    }
    lines.extend(ITEMS.iter().map(|&x| format!("  toggle_{x}: {{ {}: \"not {}\" }}", d(x), d(x))));
    lines.push(l("invariants:"));
    lines.push(format!("  - \"{}\"", distinct.join(" and ")));
    lines.push(l("ensures:"));
    lines.extend(ITEMS.iter().map(|&x| {
        format!("  - {{ after: move_{x}, holds: \"{} == to and {}\" }}   # the card lands where it was dropped, and a drag completes nothing", p(x), unchanged(&all_d))
    }));
    lines.extend(ITEMS.iter().map(|&x| {
        let mut rest: Vec<String> = ITEMS.iter().filter(|&&y| y != x).map(|&y| d(y)).collect();
        rest.extend(ITEMS.iter().map(|&y| p(y)));
        format!("  - {{ after: toggle_{x}, holds: \"{} != old.{} and {}\" }}   # a tick completes (or reopens) that card only, and never reorders", d(x), d(x), unchanged(&rest))
    }));
    lines.push(l("scene:"));
    lines.push(l("  box: [20, 25, 225, 265]"));
    lines.push(l("  shapes:"));
    lines.push(l("    - {rect: [122, 142, 200, 236], r: 10, color: panel, stroke: line, width: 1.5}"));
    lines.extend((0..n).map(|i| format!("    - {{line: [36, {}, 208, {}], color: line, width: 1.5}}", 88 + 55 * i, 88 + 55 * i)));
    lines.push(l("    - {label: to do, at: [60, 36], color: mut, size: 1.8}"));
    lines.push(l("    - {label: done, at: [110, 36], color: mut, size: 1.8}"));
    lines.push(l("render:"));
    lines.extend(ITEMS.iter().map(|&x| {
        let dx = d(x);
        format!(
            "  {x}: {{ x: \"{LEFT} + {INDENT} * int({dx})\", y: \"{TOP} + {ROW} * {}\", shapes: [{{rect: [30, 0, 120, 40], r: 8, color: panel, stroke: ink, width: 1.5}}, {{rect: [-14, 0, 18, 18], r: 4, color: none, stroke: mut, width: 1.5, when: \"not {dx}\"}}, {{rect: [-14, 0, 18, 18], r: 4, color: f1, when: {dx}}}, {{label: x, at: [-14, 0], color: white, size: 1.8, when: {dx}}}, {{label: {x}, at: [24, 0], color: ink, size: 2.4}}, {{line: [8, 0, 42, 0], color: mut, width: 2, when: {dx}}}] }}",
            p(x)
        )
    }));
    lines.push(l("futures:"));
    lines.push(l("  # one gesture per card: every index it can be spliced back in at (Dragology's `statesWith`)"));
    lines.extend(ITEMS.iter().map(|&x| format!("  drag_{x}: {{ kind: closest, over: {}, grab: {x}, action: move_{x}, stickiness: 6, floating: true }}", p(x))));
    let mut init: Vec<String> = ITEMS.iter().enumerate().map(|(i, &x)| format!("{}: {i}", p(x))).collect();
    init.extend(ITEMS.iter().map(|&x| format!("{}: {}", d(x), if x == DONE { "true" } else { "false" })));
    lines.push(format!("initial: {{{}}}", init.join(", ")));
    text(lines)
}
