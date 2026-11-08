//! Dragology's `scheduler` demo (src/demos/scheduler.tsx) as `specs/scheduler.nerve`, scaled to 3 meetings,
//! 2 rooms and a 3-slot timeline.
//!
//! Dragology's drags, computed here independently on a plain list of meetings: the bar goes to every room
//! (`statesChangeRoom`), and a dot varies its end under `lessThan(0, start)`, `lessThan(start, end -
//! MIN_LENGTH)`, `lessThan(end, END_TIME)` (each `a - b <= 0`), which on the slot grid is every slot that keeps
//! those constraints. The spec's drags derive their futures from `set_room<i>` / `set_start<i>` / `set_end<i>`.

use nerve::drag::{drag, DragSession};
use nerve::oracle::Oracle;
use nerve::spec::{Spec, State};
use rand::SeedableRng;
use serde_json::{json, Map, Value as J};
use std::path::Path;

const NUM_TRACKS: i64 = 2;
const END_TIME: i64 = 3;
const MIN_LENGTH: i64 = 1;

fn spec() -> Spec {
    Spec::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/scheduler.nerve")).unwrap()
}

/// Dragology's `Meeting`, in slots.
#[derive(Clone, Copy, Debug)]
struct Meeting {
    room: i64,
    start: i64,
    end: i64,
}

const fn m(room: i64, start: i64, end: i64) -> Meeting {
    Meeting { room, start, end }
}

fn state(sp: &Spec, ms: &[Meeting]) -> State {
    let mut j = Map::new();
    for (i, x) in ms.iter().enumerate() {
        j.insert(format!("room{i}"), json!(x.room));
        j.insert(format!("start{i}"), json!(x.start));
        j.insert(format!("end{i}"), json!(x.end));
    }
    sp.state_from_json(&J::Object(j)).unwrap()
}

/// Dragology's endpoint constraint, `lessThan(a, b)` meaning `a - b <= 0`.
fn fits(x: &Meeting) -> bool {
    0 <= x.start && x.start <= x.end - MIN_LENGTH && x.end <= END_TIME
}

/// `statesChangeRoom`: meeting `i` in each room, nothing else touched.
fn states_change_room(ms: &[Meeting], i: usize) -> Vec<Vec<Meeting>> {
    (0..NUM_TRACKS)
        .map(|t| {
            let mut s = ms.to_vec();
            s[i].room = t;
            s
        })
        .collect()
}

/// `d.vary(state, param("meetings", i, endpoint), {constraint})` on the slot grid: every slot of that end the
/// constraint allows, the other end and the room held.
fn vary_endpoint(ms: &[Meeting], i: usize, endpoint: &str) -> Vec<Vec<Meeting>> {
    (0..=END_TIME)
        .filter_map(|v| {
            let mut s = ms.to_vec();
            match endpoint {
                "start" => s[i].start = v,
                _ => s[i].end = v,
            }
            fits(&s[i]).then_some(s)
        })
        .collect()
}

fn sorted(sp: &Spec, mut v: Vec<State>) -> Vec<State> {
    v.sort_by_key(|s| sp.state_key(s));
    v
}

/// Starting schedules: the initial one, a room triple-booked, short meetings, and one long meeting.
fn starts() -> [[Meeting; 3]; 4] {
    [
        [m(0, 0, 2), m(1, 1, 3), m(1, 0, 1)],
        [m(0, 0, 3), m(0, 1, 2), m(0, 2, 3)],
        [m(0, 1, 2), m(1, 1, 2), m(0, 0, 1)],
        [m(1, 2, 3), m(0, 0, 1), m(1, 0, 3)],
    ]
}

#[test]
fn every_handle_offers_exactly_dragologys_futures() {
    let sp = spec();
    for ms in &starts() {
        let start = state(&sp, ms);
        for i in 0..ms.len() {
            let gestures: [(String, String, Vec<Vec<Meeting>>); 3] = [
                (format!("drag_bar{i}"), format!("set_room{i}"), states_change_room(ms, i)),
                (format!("drag_left{i}"), format!("set_start{i}"), vary_endpoint(ms, i, "start")),
                (format!("drag_right{i}"), format!("set_end{i}"), vary_endpoint(ms, i, "end")),
            ];
            for (gesture, action, futures) in gestures {
                let sess = DragSession::new(&sp, sp.future(&gesture).unwrap(), start.clone(), None).unwrap();
                let got = sorted(&sp, sess.candidates().iter().map(|c| c.state.clone()).collect());
                let want = sorted(&sp, futures.iter().map(|s| state(&sp, s)).collect());
                assert_eq!(got, want, "{gesture} from {ms:?}");
                for c in sess.candidates() {
                    if let Some(a) = &c.action {
                        assert_eq!(a.name, action, "{gesture} moves only meeting {i}'s own handle");
                    }
                }
            }
        }
    }
}

#[test]
fn dragging_handles_commits_what_dragology_would() {
    let sp = spec();
    let initial = [m(0, 0, 2), m(1, 1, 3), m(1, 0, 1)];
    assert_eq!(sp.initial, state(&sp, &initial));
    // Dots sit at x = 60 + 100 * slot; meeting i's row is y = 40 + 48 * i + 170 * room.
    // Meeting 0's right dot (260, 40) pulled to the end of the timeline.
    let (_, fin) = drag(&sp, "drag_right0", sp.initial.clone(), &[(260.0, 40.0), (300.0, 42.0), (356.0, 40.0)]).unwrap();
    assert_eq!(fin, state(&sp, &[m(0, 0, 3), initial[1], initial[2]]));
    // Meeting 1's left dot (160, 258) pushed past its right end stops one slot short of it.
    let (_, fin) = drag(&sp, "drag_left1", sp.initial.clone(), &[(160.0, 258.0), (300.0, 258.0), (420.0, 258.0)]).unwrap();
    assert_eq!(fin, state(&sp, &[initial[0], m(1, 2, 3), initial[2]]));
    // Meeting 2's bar (110, 306) carried up into room 0, where it double-books with meeting 0: allowed.
    let (frames, fin) = drag(&sp, "drag_bar2", sp.initial.clone(), &[(110.0, 306.0), (112.0, 220.0), (110.0, 140.0)]).unwrap();
    assert_eq!(fin, state(&sp, &[initial[0], initial[1], m(0, 0, 1)]));
    assert_eq!(frames.last().unwrap().weights.len(), 2, "`between` interpolates the two rooms");
    assert!(Oracle::new(&sp).violations(&sp.initial, None, &fin, 0.01).is_empty());
}

#[test]
fn every_meeting_on_the_grid_is_a_schedule_even_double_booked() {
    let sp = spec();
    let oracle = Oracle::new(&sp);
    let all = sp.state_space().unwrap();
    assert_eq!(all.len(), (2 * 3 * 3usize).pow(3));
    // Per meeting: 2 rooms x the 6 (start, end) pairs with start + 1 <= end.
    assert_eq!(all.iter().filter(|s| !oracle.breaks_invariant(s)).count(), 12usize.pow(3));
    assert!(!oracle.breaks_invariant(&state(&sp, &[m(0, 0, 3), m(0, 0, 3), m(0, 1, 2)])), "overlaps are not forbidden");
    assert_eq!(sp.action_instances().unwrap().len(), 3 * (2 + 3 + 3));
}

#[test]
fn the_spec_passes_its_own_self_check() {
    let sp = spec();
    let problems = Oracle::new(&sp).self_check(&mut rand::rngs::StdRng::seed_from_u64(0)).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}
