//! The negotiation, the homage to the repo this branch grew out of: Deal or No Deal, scaled to a
//! finite game. Two agents split 2 books and 2 hats; a proposal is what the proposer keeps. The rules
//! (`propose`, `accept`, `walk`) are written down. `auto` is not: it is the move of whichever agent is
//! on turn, from a policy shown only by example, one per state: open by keeping everything, concede
//! one item a round, take any offer that leaves you at least half, and walk when the last round's
//! offer does not. A machine pulses `auto` every tick and the two sides haggle until they shake hands.

use super::{generated_by, l, text};

const BOOKS: usize = 2;
const HATS: usize = 2;
/// Proposals allowed before the table closes.
const ROUNDS: usize = 3;
/// An offer is fair to the mover when the proposer keeps no more than half of the items.
const HALF: usize = (BOOKS + HATS) / 2;
/// What the proposer keeps in rounds 0, 1 and 2: everything, then one item less each round (a hat
/// first, then a book).
const OFFERS: [(usize, usize); ROUNDS] = [(BOOKS, HATS), (BOOKS, HATS - 1), (BOOKS - 1, HATS - 1)];

const TURNS: [&str; 2] = ["a", "b"];
const OPEN: &str = "open";
const DEAL: &str = "deal";
const NO_DEAL: &str = "no_deal";
const STATUSES: [&str; 3] = [OPEN, DEAL, NO_DEAL];
const FIELDS: [&str; 5] = ["turn", "books", "hats", "status", "rounds"];

/// One state of the negotiation, in the spec's field order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct St {
    turn: &'static str,
    books: usize,
    hats: usize,
    status: &'static str,
    rounds: usize,
}

const INITIAL: St = St { turn: "a", books: 0, hats: 0, status: OPEN, rounds: 0 };

impl St {
    fn yaml(&self) -> String {
        format!("{{turn: {}, books: {}, hats: {}, status: {}, rounds: {}}}", self.turn, self.books, self.hats, self.status, self.rounds)
    }
}

fn other(turn: &str) -> &'static str {
    if turn == "a" {
        "b"
    } else {
        "a"
    }
}

/// Every state, in the order the spec enumerates them.
fn states() -> Vec<St> {
    let mut out = Vec::new();
    for turn in TURNS {
        for books in 0..=BOOKS {
            for hats in 0..=HATS {
                for status in STATUSES {
                    for rounds in 0..=ROUNDS {
                        out.push(St { turn, books, hats, status, rounds });
                    }
                }
            }
        }
    }
    out
}

/// The policy `auto` shows: what the agent named by `turn` does from `s`. Each step is a legal
/// `propose` (open, rounds < 3, turn flips), `accept` (open, rounds > 0) or `walk` (open) of the
/// spec, or the identity once the table is closed.
fn play(s: St) -> St {
    if s.status != OPEN {
        return s; // terminal states are absorbing
    }
    // `books`/`hats` are what the last proposer keeps; the standing offer is fair when it leaves the mover half.
    if s.rounds > 0 && s.books + s.hats <= HALF {
        return St { status: DEAL, ..s };
    }
    if s.rounds < ROUNDS {
        let (books, hats) = OFFERS[s.rounds];
        return St { turn: other(s.turn), books, hats, status: OPEN, rounds: s.rounds + 1 };
    }
    St { status: NO_DEAL, ..s }
}

/// How many `auto` ticks the policy takes from the opening state to a deal.
fn ticks_to_deal() -> usize {
    let mut s = INITIAL;
    let mut ticks = 0;
    while s.status == OPEN {
        s = play(s);
        ticks += 1;
        assert!(ticks <= ROUNDS + 1, "the policy must close the table within the rounds it has");
    }
    assert_eq!(s.status, DEAL, "the policy must reach a deal from the opening state");
    ticks
}

fn unchanged(fields: &[&str]) -> String {
    fields.iter().map(|f| format!("{f} == old.{f}")).collect::<Vec<_>>().join(" and ")
}

/// An `auto` step is a legal propose, accept or walk, or the identity: the fallback, and the only
/// step once the negotiation is closed, since each legal move needs `old.status == 'open'`.
fn legal_auto_step() -> String {
    let identity = unchanged(&FIELDS);
    let offer = unchanged(&["turn", "books", "hats", "rounds"]);
    let propose = format!("(old.status == 'open' and old.rounds < {ROUNDS} and status == 'open' and turn != old.turn and rounds == old.rounds + 1)");
    let accept = format!("(old.status == 'open' and old.rounds > 0 and status == 'deal' and {offer})");
    let walk = format!("(old.status == 'open' and status == 'no_deal' and {offer})");
    format!("({identity}) or {propose} or {accept} or {walk}")
}

pub fn spec() -> String {
    let examples: Vec<String> = states().iter().map(|s| format!("      - {{ state: {}, next: {} }}", s.yaml(), play(*s).yaml())).collect();
    let ticks = ticks_to_deal();
    let deal_emit = l(r#"emits: { deal: "status == 'deal' and old.status != 'deal'" }"#);
    let mut lines = vec![
        l("# Homage to the repo this branch grew out of: Deal or No Deal, scaled to a"),
        l(r#"# finite game. Two agents split 2 books and 2 hats. A proposal is "what I keep"."#),
        format!("# `auto` is the move of whichever agent is on turn, shown only by example (one per state, {}):", examples.len()),
        l("# open by keeping everything, concede one item a round (a hat, then a book), take any offer that"),
        l("# leaves you at least half, and walk when the last round's offer does not. A machine pulses `auto`"),
        format!("# every tick and the two sides haggle until they shake hands (`deal`, {ticks} ticks from the opening state)."),
        generated_by("negotiate"),
        l("nerve: 1"),
        l("name: negotiate"),
        l("state:"),
        l("  turn:   {type: enum, values: [a, b]}"),
        l("  books:  {type: int, range: [0, 2]}     # books proposer keeps"),
        l("  hats:   {type: int, range: [0, 2]}     # hats proposer keeps"),
        l("  status: {type: enum, values: [open, deal, no_deal]}"),
        l("  rounds: {type: int, range: [0, 3]}"),
        l("actions:"),
        l("  propose: {params: {b: {type: int, range: [0, 2]}, h: {type: int, range: [0, 2]}}}"),
        l("  accept: {}"),
        l("  walk: {}"),
        l("  auto: {}   # the agent whose turn it is acts, from a policy shown by examples (see `learn`)"),
        l("transitions:"),
        l("  propose:"),
        l(r#"    when: "status == 'open' and rounds < 3""#),
        l(r#"    books: "b""#),
        l(r#"    hats: "h""#),
        l(r#"    turn: "'b' if turn == 'a' else 'a'""#),
        l(r#"    rounds: "rounds + 1""#),
        l("  accept:"),
        l(r#"    when: "status == 'open' and rounds > 0""#),
        l(r#"    status: "'deal'""#),
        format!("    {deal_emit}"),
        l("  walk:"),
        l(r#"    when: "status == 'open'""#),
        l(r#"    status: "'no_deal'""#),
        l("  auto:   # the fallback the network is learned on top of: nothing happens; a deal it reaches still emits"),
    ];
    lines.extend(FIELDS.iter().map(|f| format!("    {f}: \"{f}\"")));
    lines.push(format!("    {deal_emit}"));
    lines.extend([
        l("invariants:"),
        l(r#"  - "0 <= books <= 2 and 0 <= hats <= 2""#),
        l(r#"  - "rounds <= 3""#),
        l("ensures:"),
        l(r#"  - { after: propose, holds: "turn != old.turn or old.status != 'open' or old.rounds >= 3" }"#),
        l(r#"  - { after: accept,  holds: "status != 'open' or old.rounds == 0" }"#),
        format!("  - {{ after: auto,    holds: \"{}\" }}   # an auto step is a legal propose, accept or walk, or the identity", legal_auto_step()),
        l(r#"  - { holds: "status == old.status or old.status == 'open'" }   # terminal states are absorbing"#),
        l("initial: {turn: a, books: 0, hats: 0, status: open, rounds: 0}"),
        l("scene:"),
        l("  box: [0, 0, 270, 165]"),
        l("  shapes:"),
        l("    - {line: [40, 40, 200, 40], color: track, width: 4}"),
        l("    - {line: [40, 120, 200, 120], color: track, width: 4}"),
        l("    - {label: keep 0, at: [40, 66], color: mut, size: 1.6}"),
        l("    - {label: keep 1, at: [120, 66], color: mut, size: 1.6}"),
        l("    - {label: keep 2, at: [200, 66], color: mut, size: 1.6}"),
        l("    - {label: keep 0, at: [40, 146], color: mut, size: 1.6}"),
        l("    - {label: keep 1, at: [120, 146], color: mut, size: 1.6}"),
        l("    - {label: keep 2, at: [200, 146], color: mut, size: 1.6}"),
        l("    - {label: books, at: [240, 40], color: mut, size: 1.8}"),
        l("    - {label: hats, at: [240, 120], color: mut, size: 1.8}"),
        l("    - {label: 'turn ', text: turn, at: [70, 155], color: mut, size: 1.6}"),
        l("    - {text: status, at: [150, 155], color: mut, size: 1.6}"),
        l("    - {label: 'round ', text: rounds, at: [225, 155], color: mut, size: 1.6}"),
        l("render:                    # proposal markers on two 0..2 tracks: drag what you want to keep"),
        l(r#"  books: { x: "40 + 80 * books", y: 40, shapes: [{rect: [0, 0, 44, 34], r: 8, color: panel, stroke: ink, width: 1.5}, {rect: [-8, 2, 10, 20], r: 1, color: f0}, {rect: [3, 4, 10, 16], r: 1, color: f4}, {rect: [13, 0, 8, 24], r: 1, color: f1}] }"#),
        l(r#"  hats:  { x: "40 + 80 * hats", y: 120, shapes: [{rect: [0, 0, 44, 34], r: 8, color: panel, stroke: ink, width: 1.5}, {rect: [0, 8, 32, 5], r: 2, color: ink}, {rect: [0, -3, 18, 18], r: 2, color: ink}] }"#),
        l("futures:"),
        l("  # each drag is a `propose`; the oracle decides which counts are still legal (open, rounds < 3)"),
        l("  drag_books: { kind: closest, over: books, grab: books, action: propose, stickiness: 6, far: 70 }"),
        l("  drag_hats:  { kind: closest, over: hats, grab: hats, action: propose, stickiness: 6, far: 70 }"),
        l("learn:"),
        l("  # The policy the two sides play, one example per state: `auto` proposes, accepts or walks for the"),
        l("  # agent on turn. The spec bounds it (a legal move, or nothing once the table is closed) but never"),
        l("  # writes it; the `auto` transition above is the fallback."),
        l("  - name: play"),
        l("    action: auto"),
        l("    examples:"),
    ]);
    lines.extend(examples);
    text(lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_policy_concedes_one_item_a_round_until_the_deal() {
        assert_eq!(ticks_to_deal(), ROUNDS + 1);
        let mut s = INITIAL;
        let mut kept = Vec::new();
        while s.status == OPEN {
            s = play(s);
            kept.push((s.books, s.hats));
        }
        assert_eq!(kept, vec![(2, 2), (2, 1), (1, 1), (1, 1)]);
        assert_eq!(s, St { turn: "b", books: 1, hats: 1, status: DEAL, rounds: 3 });
    }

    #[test]
    fn every_state_has_one_example_and_closed_tables_stay_put() {
        let all = states();
        assert_eq!(all.len(), 2 * (BOOKS + 1) * (HATS + 1) * STATUSES.len() * (ROUNDS + 1));
        for s in all {
            let n = play(s);
            if s.status != OPEN {
                assert_eq!(n, s, "{s:?}");
            } else if n.status == OPEN {
                assert!(s.rounds < ROUNDS && n.turn != s.turn && n.rounds == s.rounds + 1, "{s:?} -> {n:?} is not a legal propose");
            } else {
                assert_eq!((n.turn, n.books, n.hats, n.rounds), (s.turn, s.books, s.hats, s.rounds), "{s:?} -> {n:?} changed the offer");
                assert!(n.status == NO_DEAL || s.rounds > 0, "{s:?} accepted before any offer");
                assert!(n.status == DEAL || s.rounds == ROUNDS, "{s:?} walked while a proposal was still allowed");
            }
        }
    }
}
