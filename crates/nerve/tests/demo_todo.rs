//! Dragology's `todo` demo (src/demos/todo.tsx) as `specs/todo.nerve`.
//!
//! Dragology writes a card's futures by hand: splice the card out of the list, then splice it back in at every
//! index (`statesWith`). Here that rule is computed independently on a plain list and compared with what the
//! spec's drag derives from `move_<card>` through the oracle.

use nerve::drag::{drag, DragSession};
use nerve::oracle::Oracle;
use nerve::spec::{ActionInst, Spec, State};
use rand::SeedableRng;
use serde_json::{json, Map, Value as J};
use std::path::Path;

fn spec() -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/todo.nerve")).unwrap()
}

/// One card as Dragology holds it (its text cannot be expressed, so the id stands for it).
#[derive(Clone, Debug)]
struct Todo {
    id: char,
    completed: bool,
}

/// Cards top to bottom, e.g. `list("cadb", "ab")` with a and b completed.
fn list(order: &str, done: &str) -> Vec<Todo> {
    order.chars().map(|id| Todo { id, completed: done.contains(id) }).collect()
}

/// The spec's state for a list: each card's position `p<id>` and its done flag `d<id>`.
fn state(sp: &Spec, todos: &[Todo]) -> State {
    let mut j = Map::new();
    for (i, t) in todos.iter().enumerate() {
        j.insert(format!("p{}", t.id), json!(i));
        j.insert(format!("d{}", t.id), json!(t.completed));
    }
    sp.state_from_json(&J::Object(j)).unwrap()
}

/// Dragology's `dragologyOnDrag` for the card at `idx`: `stateWithout` splices it out, and `statesWith` splices
/// it back in at every index `0..todos.length`.
fn states_with(todos: &[Todo], idx: usize) -> Vec<Vec<Todo>> {
    let mut without = todos.to_vec();
    let todo = without.remove(idx);
    (0..todos.len())
        .map(|new_idx| {
            let mut s = without.clone();
            s.insert(new_idx, todo.clone());
            s
        })
        .collect()
}

fn sorted(sp: &Spec, mut v: Vec<State>) -> Vec<State> {
    v.sort_by_key(|s| sp.state_key(s));
    v
}

/// Starting lists: different orders, and every card both completed and not somewhere.
fn starts() -> [Vec<Todo>; 4] {
    [list("abcd", "b"), list("dbca", ""), list("cadb", "acd"), list("bdac", "abcd")]
}

#[test]
fn every_card_drag_offers_exactly_dragologys_states_with() {
    let sp = spec();
    for todos in &starts() {
        let start = state(&sp, todos);
        for (idx, t) in todos.iter().enumerate() {
            let f = sp.future(&format!("drag_{}", t.id)).unwrap();
            assert!(f.floating, "Dragology's useFloating variant");
            let sess = DragSession::new(&sp, f, start.clone(), None).unwrap();
            let got = sorted(&sp, sess.candidates().iter().map(|c| c.state.clone()).collect());
            let want = sorted(&sp, states_with(todos, idx).iter().map(|l| state(&sp, l)).collect());
            assert_eq!(got, want, "dragging {} in {todos:?}", t.id);
            for c in sess.candidates() {
                if let Some(a) = &c.action {
                    assert_eq!(a.name, format!("move_{}", t.id), "dragging a card moves that card");
                }
            }
        }
    }
}

#[test]
fn ticking_a_box_completes_that_card_and_nothing_else() {
    let sp = spec();
    let oracle = Oracle::new(&sp);
    for todos in &starts() {
        for t in todos {
            let mut want = todos.clone();
            want.iter_mut().filter(|x| x.id == t.id).for_each(|x| x.completed = !x.completed);
            let got = oracle.step(&state(&sp, todos), &ActionInst::new(&format!("toggle_{}", t.id))).unwrap();
            assert_eq!(got, state(&sp, &want), "ticking {} in {todos:?}", t.id);
        }
    }
}

#[test]
fn dragging_a_card_commits_the_splice_under_the_pointer() {
    let sp = spec();
    assert_eq!(sp.initial, state(&sp, &list("abcd", "b")), "Dragology's list, b completed, plus d");
    // Cards sit at x = 60 (+50 when completed), y = 60 + 55 * index. Carry a from the top to index 2.
    let (frames, fin) = drag(&sp, "drag_a", sp.initial.clone(), &[(60.0, 60.0), (64.0, 120.0), (61.0, 168.0)]).unwrap();
    assert_eq!(frames.len(), 3);
    assert_eq!(fin, state(&sp, &list("bcad", "b")));
    // The completed card b, drawn indented, goes to the top and stays completed.
    let (_, fin) = drag(&sp, "drag_b", sp.initial.clone(), &[(110.0, 115.0), (108.0, 70.0), (110.0, 58.0)]).unwrap();
    assert_eq!(fin, state(&sp, &list("bacd", "b")));
    assert!(Oracle::new(&sp).violations(&sp.initial, None, &fin, 0.01).is_empty());
}

#[test]
fn every_order_with_every_completion_is_a_list() {
    let sp = spec();
    let oracle = Oracle::new(&sp);
    let all = sp.state_space().unwrap();
    assert_eq!(all.len(), 4usize.pow(4) * 2usize.pow(4));
    assert_eq!(all.iter().filter(|s| !oracle.breaks_invariant(s)).count(), 24 * 16);
    assert_eq!(sp.action_instances().unwrap().len(), 4 * 4 + 4);
}

#[test]
fn the_spec_passes_its_own_self_check() {
    let sp = spec();
    let problems = Oracle::new(&sp).self_check(&mut rand::rngs::StdRng::seed_from_u64(0)).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}
