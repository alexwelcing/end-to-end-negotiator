//! Dragology's `hanoi` demo with four disks. The spec writes the rules (only a top disk moves, never
//! onto a smaller one), and a `learn` hole, `hint`, gives the optimal next move toward peg 2 only by
//! example: one example per position, found here by breadth-first search. A training run holds some
//! of those examples out, so the network is scored on positions whose solution it never saw.

use super::{generated_by, l, text};
use std::collections::{HashMap, VecDeque};

const DISKS: usize = 4;
const PEGS: usize = 3;
const GOAL_PEG: usize = 2;
/// "No disk on this peg" compares larger than every disk.
const NONE: usize = DISKS + 1;

type Pos = [usize; DISKS];

fn f(k: usize) -> String {
    format!("d{k}")
}

/// Smallest disk on `peg`, or NONE: the disk a move from `peg` would lift.
fn top_expr(peg: &str) -> String {
    let chain: Vec<String> = (1..=DISKS).map(|k| format!("{k} if {} == {peg}", f(k))).collect();
    format!("({} else {NONE})", chain.join(" else "))
}

fn top(s: &Pos, peg: usize) -> usize {
    (1..=DISKS).find(|&k| s[k - 1] == peg).unwrap_or(NONE)
}

fn moves(s: &Pos) -> Vec<Pos> {
    let mut out = Vec::new();
    for src in 0..PEGS {
        for dst in 0..PEGS {
            let k = top(s, src);
            if src != dst && k < NONE && k < top(s, dst) {
                let mut n = *s;
                n[k - 1] = dst;
                out.push(n);
            }
        }
    }
    out
}

/// The optimal next position toward all disks on GOAL_PEG, for every position; unique for Hanoi.
fn solution() -> Vec<(Pos, Pos)> {
    let goal: Pos = [GOAL_PEG; DISKS];
    let mut dist: HashMap<Pos, usize> = HashMap::from([(goal, 0)]);
    let mut queue = VecDeque::from([goal]);
    while let Some(s) = queue.pop_front() {
        for n in moves(&s) {
            // moves are reversible, so search outward from the goal
            if !dist.contains_key(&n) {
                dist.insert(n, dist[&s] + 1);
                queue.push_back(n);
            }
        }
    }
    assert_eq!(dist.len(), PEGS.pow(DISKS as u32), "every position reaches the goal");
    let mut best: Vec<(Pos, Pos)> = Vec::new();
    for (s, &d) in &dist {
        if d == 0 {
            best.push((*s, *s));
        } else {
            let closer: Vec<Pos> = moves(s).into_iter().filter(|n| dist[n] == d - 1).collect();
            assert_eq!(closer.len(), 1, "{s:?}: {} optimal moves", closer.len());
            best.push((*s, closer[0]));
        }
    }
    best.sort();
    best
}

/// At most one disk moved, and if one did it was on top before and lands on a larger disk or an empty peg.
fn legal_single_move() -> String {
    let moved: Vec<String> = (1..=DISKS).map(|k| format!("int({} != old.{})", f(k), f(k))).collect();
    let per_disk: Vec<String> = (1..=DISKS)
        .map(|k| {
            let fk = f(k);
            let clear: Vec<String> = (1..k).map(|j| format!("old.{} != old.{fk} and old.{} != {fk}", f(j), f(j))).collect();
            let clear = if clear.is_empty() { l("True") } else { clear.join(" and ") };
            format!("({fk} == old.{fk} or ({clear}))")
        })
        .collect();
    format!("({}) <= 1 and {}", moved.join(" + "), per_disk.join(" and "))
}

fn stack_height(k: usize) -> String {
    let larger: Vec<String> = (k + 1..=DISKS).map(|j| format!("int({} == {})", f(j), f(k))).collect();
    if larger.is_empty() { l("0") } else { larger.join(" + ") }
}

pub fn spec() -> String {
    let best = solution();
    let pos = |s: &Pos| (1..=DISKS).map(|k| format!("{}: {}", f(k), s[k - 1])).collect::<Vec<_>>().join(", ");
    let examples: Vec<String> = best.iter().map(|(s, n)| format!("      - {{ state: {{{}}}, next: {{{}}} }}", pos(s), pos(n))).collect();
    let mut lines = vec![
        l("# Dragology's Tower of Hanoi, four disks (d1 smallest). The rules are written down: only a top"),
        l("# disk moves, and never onto a smaller one, so a drag can only offer legal drops. `hint` is not:"),
        l("# it is the optimal next move toward peg 2, given only by example (one per position). Training"),
        l("# holds some examples out, and the network is scored on positions whose solution it never saw."),
        generated_by("hanoi"),
        l("nerve: 1"),
        l("name: hanoi"),
        l("state:"),
    ];
    lines.extend((1..=DISKS).map(|k| format!("  {}: {{type: int, range: [0, {}]}}   # peg of disk {k}", f(k), PEGS - 1)));
    lines.push(l("actions:"));
    lines.push(format!("  move: {{params: {{src: {{type: int, range: [0, {}]}}, dst: {{type: int, range: [0, {}]}}}}}}", PEGS - 1, PEGS - 1));
    lines.push(l("  hint: {}   # the optimal next move toward peg 2: shown by examples, never written down"));
    lines.push(l("transitions:"));
    lines.push(l("  move:"));
    lines.push(format!("    when: \"{} < {NONE} and {} < {}\"", top_expr("src"), top_expr("src"), top_expr("dst")));
    lines.extend((1..=DISKS).map(|k| format!("    {}: \"dst if {} == {k} else {}\"", f(k), top_expr("src"), f(k))));
    lines.push(l("ensures:"));
    lines.push(format!("  - {{ after: move, holds: \"{}\" }}", legal_single_move()));
    lines.push(format!("  - {{ after: hint, holds: \"{}\" }}   # a hint may only make a legal move", legal_single_move()));
    lines.push(l("scene:"));
    lines.push(l("  box: [0, 75, 420, 240]"));
    lines.push(l("  shapes:"));
    lines.push(l("    - {rect: [210, 220, 396, 12], r: 4, color: track}"));
    lines.extend((0..PEGS).map(|i| format!("    - {{rect: [{}, 154, 8, 124], r: 3, color: track}}", 70 + 140 * i)));
    lines.extend(["left", "middle", "right"].iter().enumerate().map(|(i, name)| format!("    - {{label: {name} peg, at: [{}, 236], color: mut, size: 1.8}}", 70 + 140 * i)));
    lines.push(l("render:"));
    lines.extend((1..=DISKS).map(|k| {
        format!(
            "  disk{k}: {{ x: \"70 + 140 * {}\", y: \"200 - 30 * ({})\", shapes: [{{rect: [0, 0, {}, 24], r: 8, color: f{}, stroke: ink, width: 1.5}}, {{label: '{k}', at: [0, 0], color: white, size: 2.2}}] }}",
            f(k),
            stack_height(k),
            36 + 22 * (k - 1),
            k - 1
        )
    }));
    lines.push(l("futures:"));
    lines.push(l("  # one gesture per disk: only a top disk has anywhere to go, and only onto a larger disk"));
    lines.extend((1..=DISKS).map(|k| format!("  drag_disk{k}: {{ kind: closest, over: {}, grab: disk{k}, action: move, stickiness: 8, floating: true }}", f(k))));
    lines.push(l("learn:"));
    lines.push(l("  - name: solve"));
    lines.push(l("    action: hint"));
    lines.push(format!("    fallback: {{ {}: \"{}\" }}", f(1), f(1)));
    lines.push(l("    examples:"));
    lines.extend(examples);
    lines.push(format!("initial: {{{}}}", (1..=DISKS).map(|k| format!("{}: 0", f(k))).collect::<Vec<_>>().join(", ")));
    text(lines)
}
