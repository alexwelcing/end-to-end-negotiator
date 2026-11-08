//! Dragology's `study-three-way-switch` as `specs/three_way_switch.nerve`: the knob commits to the
//! nearest of the places it may reach, all three by default and only the next one clockwise in
//! "clockwise only" mode.

use nerve::drag::{drag, DragSession};
use nerve::oracle::Oracle;
use nerve::spec::{ActionInst, Spec, State};
use nerve::value::Value;
use rand::SeedableRng;

fn spec() -> Spec {
    Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/three_way_switch.nerve")).unwrap()
}

/// Dragology's geometry: `translate(60, 30)` plus `POS[name]`.
fn dragology_pos(name: &str) -> (f64, f64) {
    let (x, y) = match name {
        "r" => (50.0, 20.0),
        "g" => (16.0, 78.0),
        "b" => (84.0, 78.0),
        other => panic!("no position for {other}"),
    };
    (60.0 + x, 30.0 + y)
}

/// The demo's clockwise order on screen (y down): r (top) -> b (bottom right) -> g (bottom left) -> r.
fn next_clockwise(name: &str) -> &'static str {
    match name {
        "r" => "b",
        "b" => "g",
        "g" => "r",
        other => panic!("no state {other}"),
    }
}

fn state(name: &str, clockwise_only: bool) -> State {
    vec![Value::str(name), Value::Bool(clockwise_only)]
}

/// Which of `allowed` Dragology's `between` commits to on release at `p`: the nearest, or `None`
/// when two are equally near (a tie is the engine's tie-break, not the demo's rule).
fn nearest<'a>(allowed: &[&'a str], p: (f64, f64)) -> Option<&'a str> {
    let d = |n: &str| {
        let (x, y) = dragology_pos(n);
        (x - p.0).hypot(y - p.1)
    };
    let mut by_gap: Vec<(f64, &str)> = allowed.iter().map(|&n| (d(n), n)).collect();
    by_gap.sort_by(|a, b| a.0.total_cmp(&b.0));
    (by_gap.len() < 2 || by_gap[1].0 - by_gap[0].0 > 0.5).then_some(by_gap[0].1)
}

/// What a single-frame drag from `start` to `p` commits.
fn release_at(sp: &Spec, start: &State, p: (f64, f64)) -> State {
    let mut sess = DragSession::new(sp, sp.future("drag_knob").unwrap(), start.clone(), None).unwrap();
    sess.frame(p).unwrap();
    sess.release().unwrap()
}

/// Pointer positions around the switch: at each place, between each pair, and off to the sides.
const POINTERS: [(f64, f64); 10] = [
    (110.0, 50.0),
    (76.0, 108.0),
    (144.0, 108.0),
    (100.0, 80.0),
    (95.0, 90.0),
    (128.0, 75.0),
    (115.0, 112.0),
    (20.0, 140.0),
    (200.0, 20.0),
    (104.0, 4.0),
];

#[test]
fn render_matches_dragologys_positions() {
    let sp = spec();
    for name in ["r", "g", "b"] {
        let p = nerve::drag::element_position(&sp, "knob", &state(name, false)).unwrap();
        assert_eq!(p, dragology_pos(name), "{name}");
    }
}

#[test]
fn free_mode_futures_are_all_three_places() {
    let sp = spec();
    for start in ["r", "g", "b"] {
        let sess = DragSession::new(&sp, sp.future("drag_knob").unwrap(), state(start, false), None).unwrap();
        let mut got: Vec<String> = sess.candidates().iter().map(|c| c.state[0].py_str()).collect();
        got.sort();
        assert_eq!(got, vec!["b", "g", "r"], "from {start}");
    }
}

#[test]
fn free_mode_release_commits_to_the_nearest_place() {
    let sp = spec();
    let mut checked = 0;
    for start in ["r", "g", "b"] {
        for p in POINTERS {
            let Some(want) = nearest(&["r", "g", "b"], p) else { continue };
            assert_eq!(release_at(&sp, &state(start, false), p), state(want, false), "from {start}, released at {p:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 3 * POINTERS.len(), "no pointer is a tie among all three");
    // Spelled out: from r, a release near g lands on g and one near b lands on b.
    assert_eq!(release_at(&sp, &state("r", false), (80.0, 100.0)), state("g", false));
    assert_eq!(release_at(&sp, &state("r", false), (140.0, 100.0)), state("b", false));
    assert_eq!(release_at(&sp, &state("g", false), (108.0, 58.0)), state("r", false));
}

#[test]
fn clockwise_only_futures_are_the_start_and_the_next_place() {
    let sp = spec();
    for start in ["r", "g", "b"] {
        let sess = DragSession::new(&sp, sp.future("drag_knob").unwrap(), state(start, true), None).unwrap();
        let mut got: Vec<State> = sess.candidates().iter().map(|c| c.state.clone()).collect();
        let mut want = vec![state(start, true), state(next_clockwise(start), true)];
        got.sort_by_key(|s| sp.state_key(s));
        want.sort_by_key(|s| sp.state_key(s));
        assert_eq!(got, want, "Dragology: d.between([state, next]) from {start}");
    }
}

#[test]
fn clockwise_only_release_never_steps_counter_clockwise() {
    let sp = spec();
    let mut checked = 0;
    for start in ["r", "g", "b"] {
        let allowed = [start, next_clockwise(start)];
        for p in POINTERS {
            // From b, a release right on r is equally near b and g: skip that tie.
            let Some(want) = nearest(&allowed, p) else { continue };
            assert_eq!(release_at(&sp, &state(start, true), p), state(want, true), "from {start}, released at {p:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 3 * POINTERS.len() - 1);
    // From r the knob may go to b; released right on g (counter-clockwise) it stays on r.
    assert_eq!(release_at(&sp, &state("r", true), dragology_pos("b")), state("b", true));
    assert_eq!(release_at(&sp, &state("r", true), dragology_pos("g")), state("r", true));
    // From g the only step is up to r; released next to b it stays on g, where free mode goes to b.
    assert_eq!(release_at(&sp, &state("g", true), dragology_pos("r")), state("r", true));
    assert_eq!(release_at(&sp, &state("g", true), (135.0, 112.0)), state("g", true));
    assert_eq!(release_at(&sp, &state("g", false), (135.0, 112.0)), state("b", false));
}

#[test]
fn a_whole_drag_from_r_to_b_commits_b() {
    let sp = spec();
    let path = [(110.0, 50.0), (118.0, 66.0), (127.0, 84.0), (138.0, 101.0), (143.0, 107.0)];
    let (frames, fin) = drag(&sp, "drag_knob", state("r", false), &path).unwrap();
    assert_eq!(fin, state("b", false));
    assert_eq!(frames.len(), path.len());
    assert_eq!(frames[0].branch, Some(Value::str("r")));
    assert_eq!(frames.last().unwrap().branch, Some(Value::str("b")));
    // `between` interpolates: every frame's weights cover the three places and sum to one.
    for f in &frames {
        assert_eq!(f.weights.len(), 3);
        assert!((f.weights.iter().map(|(_, w)| w).sum::<f64>() - 1.0).abs() < 1e-9);
    }
    // In clockwise-only mode the same path is the one legal step, r -> b.
    let (_, fin) = drag(&sp, "drag_knob", state("r", true), &path).unwrap();
    assert_eq!(fin, state("b", true));
}

#[test]
fn the_checkbox_is_an_action_that_keeps_the_knob_in_place() {
    let sp = spec();
    let o = Oracle::new(&sp);
    let toggle = ActionInst::new("toggle_clockwise_only");
    assert_eq!(o.step(&state("g", false), &toggle).unwrap(), state("g", true));
    assert_eq!(o.step(&state("g", true), &toggle).unwrap(), state("g", false));
    let set = |to: &str| ActionInst { name: "set".into(), params: vec![("to".into(), Value::str(to))] };
    assert_eq!(o.step(&state("g", false), &set("b")).unwrap(), state("b", false));
    assert_eq!(o.step(&state("g", true), &set("b")).unwrap(), state("g", true), "b is counter-clockwise of g");
    assert_eq!(o.step(&state("g", true), &set("r")).unwrap(), state("r", true));
}

#[test]
fn spec_passes_its_own_self_check() {
    let sp = spec();
    assert_eq!(sp.state_space().unwrap().len(), 6);
    let problems = Oracle::new(&sp).self_check(&mut rand::rngs::StdRng::seed_from_u64(0)).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}
