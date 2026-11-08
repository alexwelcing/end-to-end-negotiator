//! Dragology's `linked-sliders` as `specs/linked_sliders.nerve`: dragging slider i moves it and every
//! slider below it by the same delta, and the group stops as soon as one of them reaches a track end.

use nerve::drag::{drag, DragSession};
use nerve::oracle::Oracle;
use nerve::spec::{ActionInst, Spec, State};
use nerve::value::Value;
use rand::{Rng, SeedableRng};

const W: f64 = 240.0;
const X0: f64 = 40.0;

fn spec() -> Spec {
    Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/linked_sliders.nerve")).unwrap()
}

fn state(v: [f64; 3]) -> State {
    v.iter().map(|&x| Value::Float(x)).collect()
}

fn values(s: &State) -> [f64; 3] {
    [0, 1, 2].map(|k| s[k].as_f64().unwrap())
}

/// Dragology's rule, written out: `d.varyFunc([0], ([delta]) => values[j] += delta for j >= i)` under
/// `inOrder([0, v, W])` for every value. The pointer asks for slider i at `want`; the delta is the one
/// nearest `want - v[i]` that keeps the whole group on the tracks.
fn dragology(v: [f64; 3], i: usize, want: f64) -> [f64; 3] {
    let group = &v[i..];
    let lo = -group.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = W - group.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let delta = (want - v[i]).clamp(lo, hi);
    let mut out = v;
    for x in out.iter_mut().skip(i) {
        *x += delta;
    }
    out
}

fn assert_close(got: [f64; 3], want: [f64; 3], tol: f64, what: &str) {
    for k in 0..3 {
        assert!((got[k] - want[k]).abs() <= tol, "{what}: got {got:?}, want {want:?}");
    }
}

/// The spec draws slider i's thumb at `(40 + v_i, 30 + 40 * i)`.
fn row(i: usize) -> f64 {
    30.0 + 40.0 * i as f64
}

/// Where one pointer frame of `drag_<i+1>` from `start` puts the sliders (aiming slider i at `want`).
fn drag_to(sp: &Spec, start: [f64; 3], i: usize, want: f64) -> [f64; 3] {
    let f = sp.future(&format!("drag_{}", i + 1)).unwrap();
    let mut sess = DragSession::new(sp, f, state(start), None).unwrap();
    let fr = sess.frame((X0 + want, row(i) + 3.0)).unwrap();
    values(&fr.preview)
}

fn set(i: usize, to: f64) -> ActionInst {
    ActionInst { name: format!("set_{}", i + 1), params: vec![("to".into(), Value::Float(to))] }
}

#[test]
fn render_puts_each_thumb_on_its_own_track() {
    let sp = spec();
    let s = state([60.0, 120.0, 180.0]);
    assert_eq!(sp.initial, s, "Dragology's initial values (i + 1) / (N + 1) * W");
    for (i, x) in [60.0, 120.0, 180.0].into_iter().enumerate() {
        let p = nerve::drag::element_position(&sp, &format!("thumb{}", i + 1), &s).unwrap();
        assert_eq!(p, (X0 + x, row(i)));
    }
}

#[test]
fn dragging_a_slider_moves_it_and_every_slider_below_it() {
    let sp = spec();
    let starts = [[60.0, 120.0, 180.0], [200.0, 30.0, 90.0], [10.0, 10.0, 10.0], [120.0, 200.0, 50.0]];
    for start in starts {
        for i in 0..3 {
            for want in [start[i] - 25.0, start[i] + 17.5, start[i] + 3.0] {
                let got = drag_to(&sp, start, i, want);
                assert_close(got, dragology(start, i, want), 0.05, &format!("drag_{} from {start:?} to {want}", i + 1));
                for k in 0..i {
                    assert_eq!(got[k], start[k], "slider {} is above slider {} and must not move", k + 1, i + 1);
                }
            }
        }
    }
    // Spelled out from Dragology's initial state: slider 1 drags the other two along; slider 3 moves alone.
    assert_close(drag_to(&sp, [60.0, 120.0, 180.0], 0, 90.0), [90.0, 150.0, 210.0], 0.05, "slider 1 +30");
    assert_close(drag_to(&sp, [60.0, 120.0, 180.0], 1, 100.0), [60.0, 100.0, 160.0], 0.05, "slider 2 -20");
    assert_close(drag_to(&sp, [60.0, 120.0, 180.0], 2, 150.0), [60.0, 120.0, 150.0], 0.05, "slider 3 -30");
}

#[test]
fn the_group_stops_when_any_member_reaches_a_track_end() {
    let sp = spec();
    // From (60, 120, 180) slider 1 can go right by 60 (slider 3 hits 240) and left by 60 (slider 1 hits 0).
    assert_close(drag_to(&sp, [60.0, 120.0, 180.0], 0, 230.0), [120.0, 180.0, 240.0], 0.05, "right, stopped by slider 3");
    assert_close(drag_to(&sp, [60.0, 120.0, 180.0], 0, -40.0), [0.0, 60.0, 120.0], 0.05, "left, stopped by slider 1");
    assert_close(drag_to(&sp, [60.0, 120.0, 180.0], 1, 400.0), [60.0, 180.0, 240.0], 0.05, "slider 2 right, stopped by slider 3");
    assert_close(drag_to(&sp, [60.0, 120.0, 180.0], 2, -100.0), [60.0, 120.0, 0.0], 0.05, "slider 3 alone to 0");
    // A lower slider can be the one that stops the group on the left too: slider 2 at 30 limits slider 1.
    assert_close(drag_to(&sp, [200.0, 30.0, 90.0], 0, 0.0), [170.0, 0.0, 60.0], 0.05, "left, stopped by slider 2");
    assert_close(drag_to(&sp, [200.0, 30.0, 90.0], 0, 300.0), [240.0, 70.0, 130.0], 0.05, "right, stopped by slider 1");
    // One slider at each end pins slider 1's group completely; slider 2's group can still go left.
    let pinned = [0.0, 100.0, 240.0];
    for want in [-50.0, 50.0, 200.0] {
        assert_close(drag_to(&sp, pinned, 0, want), pinned, 0.05, &format!("slider 1 pinned, aimed at {want}"));
    }
    assert_close(drag_to(&sp, pinned, 1, 20.0), [0.0, 20.0, 160.0], 0.05, "slider 2 left");
    assert_close(drag_to(&sp, pinned, 1, 230.0), pinned, 0.05, "slider 2 cannot go right: slider 3 is at 240");
}

#[test]
fn a_whole_drag_of_slider_two_commits_the_moved_group() {
    let sp = spec();
    // Grab slider 2 at 120 and pull it right past where slider 3 runs out of track.
    let path = [(160.0, 70.0), (175.0, 72.0), (200.0, 69.0), (250.0, 71.0), (300.0, 70.0)];
    let (frames, fin) = drag(&sp, "drag_2", sp.initial.clone(), &path).unwrap();
    assert_eq!(frames.len(), path.len());
    assert_close(values(&frames[1].preview), [60.0, 135.0, 195.0], 0.05, "mid-drag");
    assert_close(values(&fin), [60.0, 180.0, 240.0], 0.05, "released past the end");
    assert_eq!(values(&fin)[0], 60.0, "slider 1 is above and never moves");
    let a = set(1, values(&fin)[1]);
    assert!(Oracle::new(&sp).violations(&sp.initial, Some(&a), &fin, 0.01).is_empty());
}

#[test]
fn every_set_matches_dragologys_rule() {
    let sp = spec();
    let o = Oracle::new(&sp);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    for _ in 0..3000 {
        let v = [rng.gen_range(0.0..=W), rng.gen_range(0.0..=W), rng.gen_range(0.0..=W)];
        let i = rng.gen_range(0..3);
        let to = rng.gen_range(0.0..=W);
        let got = values(&o.step(&state(v), &set(i, to)).unwrap());
        assert_close(got, dragology(v, i, to), 1e-9, &format!("set_{}({to}) from {v:?}", i + 1));
        assert!(got.iter().all(|x| (0.0..=W).contains(x)), "off the track: {got:?}");
    }
}

#[test]
fn spec_passes_its_own_self_check() {
    let sp = spec();
    assert!(!sp.finite());
    assert_eq!(sp.continuous_fields().len(), 3);
    let problems = Oracle::new(&sp).self_check(&mut rand::rngs::StdRng::seed_from_u64(0)).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}
