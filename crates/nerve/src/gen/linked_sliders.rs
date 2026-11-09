//! Dragology's `linked-sliders` demo: a stack of sliders on tracks of width W. Dragging slider i
//! shifts it and every slider below it by the same delta, under the constraint that every value stays
//! on its track, so the group moves rigidly and stops as soon as any member reaches an end of its
//! track. `set_<i>(to)` asks for slider i at `to`; the delta is `to - v_i` clamped to what keeps the
//! whole group on the tracks, and every slider in the group moves by it. Dragology has five sliders;
//! this has three (the continuous trainer descends over all three fields at once).

use super::{generated_by, l, text};

const N: usize = 3;
const W: usize = 240;
const X0: usize = 40;
const Y0: usize = 30;
const SPACING: usize = 40;
const TOL: &str = "0.001";

fn v(i: usize) -> String {
    format!("v{}", i + 1)
}

/// Slider i (0-based) and every slider below it: the ones a drag of slider i moves.
fn group(i: usize) -> Vec<String> {
    (i..N).map(v).collect()
}

fn lowest(names: &[String]) -> String {
    if names.len() == 1 { names[0].clone() } else { format!("min({})", names.join(", ")) }
}

fn highest(names: &[String]) -> String {
    if names.len() == 1 { names[0].clone() } else { format!("max({})", names.join(", ")) }
}

/// Dragology's delta for slider i: as far towards `to` as the group can go and stay on the tracks.
fn delta(i: usize) -> String {
    let g = group(i);
    format!("clamp(to - {}, -{}, {W} - {})", v(i), lowest(&g), highest(&g))
}

fn ensures(i: usize) -> Vec<String> {
    let (g, vi) = (group(i), v(i));
    let mut out = Vec::new();
    if i > 0 {
        let still: Vec<String> = (0..i).map(|j| format!("{} == old.{}", v(j), v(j))).collect();
        out.push(format!("  - {{ after: set_{}, holds: \"{}\" }}   # the sliders above do not move", i + 1, still.join(" and ")));
    }
    for w in &g[1..] {
        out.push(format!("  - {{ after: set_{}, holds: \"abs(({w} - {vi}) - (old.{w} - old.{vi})) < {TOL}\" }}   # {w} moves with {vi}", i + 1));
    }
    let reached = format!("abs({vi} - to) < {TOL} or (to > old.{vi} and {} == {W}) or (to < old.{vi} and {} == 0)", highest(&g), lowest(&g));
    out.push(format!("  - {{ after: set_{}, holds: \"{reached}\" }}   # it reaches `to`, or the group is at an end", i + 1));
    out
}

pub fn spec() -> String {
    let on_tracks: Vec<String> = (0..N).map(|i| format!("0 <= {} <= {W}", v(i))).collect();
    let initial: Vec<String> = (0..N).map(|i| format!("{}: {:?}", v(i), (i + 1) as f64 / (N + 1) as f64 * W as f64)).collect();
    let mut lines = vec![
        l("# Dragology's linked-sliders (src/demos/linked-sliders.tsx): a stack of sliders on tracks."),
        l("# Rule: dragging slider i moves it and every slider below it by the same amount, and the whole group"),
        l("# stops as soon as any of them reaches an end of its track (`d.varyFunc` with an `inOrder([0, v, W])`"),
        l("# constraint on every value). Scaled down from five sliders to three; vertical spacing 36 -> 40 px."),
        l("# `set_<i>(to)` asks for slider i at `to` and moves the group by `to - v_i`, clamped to keep it on the tracks."),
        generated_by("linked_sliders"),
        l("nerve: 1"),
        l("name: linked_sliders"),
        l("state:"),
    ];
    lines.extend((0..N).map(|i| format!("  {}: {{type: float, range: [0, {W}]}}   # slider {}", v(i), i + 1)));
    lines.push(l("actions:"));
    lines.extend((0..N).map(|i| format!("  set_{}: {{params: {{to: {{type: float, range: [0, {W}]}}}}}}", i + 1)));
    lines.push(l("transitions:"));
    for i in 0..N {
        lines.push(format!("  set_{}:", i + 1));
        lines.extend(group(i).iter().map(|w| format!("    {w}: \"clamp({w} + {}, 0, {W})\"", delta(i))));
        if i == 0 {
            // The event a machine can wire: the whole group has just reached the right end of its tracks.
            let all = group(0).join(", ");
            let all_before = group(0).iter().map(|w| format!("old.{w}")).collect::<Vec<_>>().join(", ");
            lines.push(format!("    emits: {{ at_end: \"max({all}) >= {W} and max({all_before}) < {W}\" }}"));
        }
    }
    lines.push(l("invariants:"));
    lines.push(format!("  - \"{}\"", on_tracks.join(" and ")));
    lines.push(l("ensures:"));
    for i in 0..N {
        lines.extend(ensures(i));
    }
    lines.push(l("scene:"));
    lines.push(l("  box: [20, 10, 300, 130]"));
    lines.push(l("  shapes:"));
    for i in 0..N {
        let y = Y0 + i * SPACING;
        lines.push(format!("    - {{line: [{X0}, {y}, {}, {y}], color: track, width: 6}}", X0 + W));
        lines.push(format!("    - {{rect: [\"{X0} + {} / 2\", {y}, {}, 6], color: acc}}", v(i), v(i)));
        lines.push(format!("    - {{label: {}, at: [26, {y}], color: mut, size: 1.8}}", v(i)));
    }
    lines.push(l("render:"));
    lines.extend((0..N).map(|i| format!("  thumb{}: {{ x: \"{X0} + {}\", y: {}, shape: {{circle: [0, 0, 11], color: panel, stroke: acc, width: 2.5}} }}", i + 1, v(i), Y0 + i * SPACING)));
    lines.push(l("futures:"));
    lines.push(l("  # one vary per thumb, realised through its own set_<i> so the group moves and the clamp applies"));
    lines.extend((0..N).map(|i| format!("  drag_{}: {{ kind: vary, over: {}, grab: thumb{}, action: set_{}, param: to }}", i + 1, v(i), i + 1, i + 1)));
    // Dragology's initial state: values[i] = (i + 1) / (N + 1) * W
    lines.push(format!("initial: {{{}}}", initial.join(", ")));
    text(lines)
}
