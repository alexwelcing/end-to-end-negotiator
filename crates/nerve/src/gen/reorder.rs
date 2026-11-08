//! Dragology's reorderable list, five items, as position fields.
//!
//! The spec language has no `list<T>` yet (ROADMAP 7), so a list is one position per item plus an
//! all-distinct invariant; `move(item, to)` takes the item out and shifts the items in between by one,
//! exactly `moveItem(items, i, j)` in Dragology's demo.

use super::{generated_by, l, text};

const ITEMS: [char; 5] = ['a', 'b', 'c', 'd', 'e'];

fn p(x: char) -> String {
    format!("p{x}")
}

/// The position of the moved item, by its name.
fn from() -> String {
    let parts: Vec<String> = ITEMS[..ITEMS.len() - 1].iter().map(|&x| format!("{} if item == '{x}'", p(x))).collect();
    format!("{} else {}", parts.join(" else "), p(ITEMS[ITEMS.len() - 1]))
}

fn update(x: char) -> String {
    let (px, f) = (p(x), from());
    format!("to if item == '{x}' else ({px} - 1 if ({f}) < {px} <= to else ({px} + 1 if to <= {px} < ({f}) else {px}))")
}

pub fn spec() -> String {
    let n = ITEMS.len();
    let f = from();
    let mut lines = vec![
        l("# Dragology's reorderable list: five items, drag one to a new place and the others make room."),
        l("# There is no list type yet (ROADMAP 7), so the list is one position per item and an invariant"),
        l("# that no two share a place. `move` is Dragology's moveItem(items, i, j): the drag's futures are"),
        l("# every order one move away, derived from this rule instead of written out by hand."),
        generated_by("reorder"),
        l("nerve: 1"),
        l("name: reorder"),
        l("state:"),
    ];
    lines.extend(ITEMS.iter().map(|&x| format!("  {}: {{type: int, range: [0, {}]}}   # position of item {x}", p(x), n - 1)));
    lines.push(l("actions:"));
    lines.push(format!("  move: {{params: {{item: {{type: enum, values: [{}]}}, to: {{type: int, range: [0, {}]}}}}}}", ITEMS.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(", "), n - 1));
    lines.push(l("transitions:"));
    lines.push(l("  move:"));
    lines.extend(ITEMS.iter().map(|&x| format!("    {}: \"{}\"", p(x), update(x))));
    // The event a machine can wire: the list came back to its initial order (and was not there before).
    let home: Vec<String> = ITEMS.iter().enumerate().map(|(i, &x)| format!("{} == {i}", p(x))).collect();
    let was_home: Vec<String> = ITEMS.iter().enumerate().map(|(i, &x)| format!("old.{} == {i}", p(x))).collect();
    lines.push(format!("    emits: {{ home: \"{} and not ({})\" }}", home.join(" and "), was_home.join(" and ")));
    lines.push(l("invariants:"));
    let mut distinct = Vec::new();
    for (i, &x) in ITEMS.iter().enumerate() {
        for &y in &ITEMS[i + 1..] {
            distinct.push(format!("{} != {}", p(x), p(y)));
        }
    }
    lines.push(format!("  - \"{}\"", distinct.join(" and ")));
    lines.push(l("ensures:"));
    lines.push(format!("  - {{ after: move, holds: \"({f}) == to\" }}   # the moved item lands where it was dropped"));
    lines.push(l("scene:"));
    lines.push(l("  box: [0, 0, 330, 80]"));
    lines.push(l("  shapes:"));
    lines.push(l("    - {rect: [168, 40, 316, 68], r: 10, color: panel, stroke: line, width: 1.5}"));
    lines.extend((0..n).map(|i| format!("    - {{rect: [{}, 40, 54, 54], r: 8, color: none, stroke: line, width: 1.5}}", 40 + 64 * i)));
    lines.push(l("render:"));
    lines.extend(ITEMS.iter().map(|&x| {
        format!("  {x}: {{ x: \"40 + 64 * {}\", y: 40, shapes: [{{rect: [0, 0, 50, 50], r: 8, color: panel, stroke: ink, width: 1.5}}, {{label: {x}, at: [0, 0], color: ink, size: 2.6}}] }}", p(x))
    }));
    lines.push(l("futures:"));
    lines.push(l("  # one gesture per item: grab it, and every order one move away competes for the pointer"));
    lines.extend(ITEMS.iter().map(|&x| format!("  drag_{x}: {{ kind: closest, over: {}, grab: {x}, action: move, stickiness: 6, floating: true }}", p(x))));
    lines.push(format!("initial: {{{}}}", ITEMS.iter().enumerate().map(|(i, &x)| format!("{}: {i}", p(x))).collect::<Vec<_>>().join(", ")));
    text(lines)
}
