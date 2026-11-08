//! Dragology's `insert-and-remove` as `specs/insert_remove.nerve`: a tile dragged from the list may be
//! dropped at any place in the rest of the list or in the trash; a tile pulled from the toolbar may be dropped
//! at any place in the list or let go. Here the toolbar slot of each of the four tiles is both its store and
//! its trash. The rule is re-implemented on plain lists, as `insert-and-remove.tsx` writes it with `amb`.

use nerve::drag::{drag, element_position, DragSession};
use nerve::oracle::Oracle;
use nerve::spec::{Spec, State};
use nerve::value::Value;
use rand::SeedableRng;

const TILES: [char; 4] = ['a', 'b', 'c', 'd'];
/// Field value of a tile that is not in the list (it sits in the toolbar).
const OUT: i64 = 4;

fn spec() -> Spec {
    Spec::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../specs/insert_remove.nerve")).unwrap()
}

/// The spec's `[pa, pb, pc, pd]` for a list: each tile's index, or OUT.
fn state(list: &[char]) -> State {
    TILES.iter().map(|t| Value::Int(list.iter().position(|x| x == t).map_or(OUT, |i| i as i64))).collect()
}

fn list(s: &State) -> Vec<char> {
    let mut placed: Vec<(i64, char)> =
        TILES.iter().zip(s).map(|(&t, v)| (v.as_f64().unwrap() as i64, t)).filter(|&(i, _)| i != OUT).collect();
    placed.sort();
    placed.into_iter().map(|(_, t)| t).collect()
}

/// `produceAmb(state, draft => draft.items.splice(amb(range(len + 1)), 0, tile))`: every way to insert.
fn insertions(items: &[char], tile: char) -> Vec<Vec<char>> {
    (0..=items.len())
        .map(|i| {
            let mut out = items.to_vec();
            out.insert(i, tile);
            out
        })
        .collect()
}

/// Dragology's futures for dragging `tile`. A list tile: `dropIntoListAndTrash(stateWithout, tile)`, i.e.
/// every insertion into the list without it, or the trash (the list without it); far away, the start. A
/// toolbar tile: `dropIntoListAndTrash(state, tile)`, every insertion into the list, or the trash and the far
/// fallback, both the list as it was.
fn dragology_futures(items: &[char], tile: char) -> Vec<State> {
    let without: Vec<char> = items.iter().copied().filter(|&x| x != tile).collect();
    let mut want: Vec<State> = insertions(&without, tile).iter().map(|l| state(l)).collect();
    want.push(state(&without));
    want.push(state(items));
    sort(&mut want);
    want
}

fn sort(states: &mut Vec<State>) {
    states.sort_by_key(|s| s.iter().map(|v| v.as_f64().unwrap() as i64).collect::<Vec<_>>());
    states.dedup();
}

fn candidate_states(sp: &Spec, start: &State, tile: char) -> Vec<State> {
    let sess = DragSession::new(sp, sp.future(&format!("drag_{tile}")).unwrap(), start.clone(), None).unwrap();
    for c in sess.candidates() {
        assert!(c.action.as_ref().is_none_or(|a| a.name == format!("move_{tile}")), "dragging {tile} ran {:?}", c.action);
    }
    let mut got: Vec<State> = sess.candidates().iter().map(|c| c.state.clone()).collect();
    sort(&mut got);
    got
}

#[test]
fn the_spec_starts_with_dragologys_list_and_counts_every_list() {
    let sp = spec();
    assert_eq!(list(&sp.initial), vec!['a', 'b', 'c'], "apple, apple, banana; grapes waits in the toolbar");
    let oracle = Oracle::new(&sp);
    let space = sp.state_space().unwrap();
    assert_eq!(space.len(), 625);
    let lists: Vec<&State> = space.iter().filter(|s| !oracle.breaks_invariant(s)).collect();
    // ordered selections of 0..4 of the 4 tiles: 1 + 4 + 12 + 24 + 24
    assert_eq!(lists.len(), 65);
    for s in lists {
        assert_eq!(state(&list(s)), *s, "{s:?} is a packed list");
    }
}

#[test]
fn tile_futures_are_dragologys_from_named_lists() {
    let sp = spec();
    let named: [&[char]; 5] = [&['a', 'b', 'c'], &[], &['d', 'c', 'b', 'a'], &['b'], &['c', 'a']];
    for items in named {
        let start = state(items);
        for tile in TILES {
            let want = dragology_futures(items, tile);
            // in the list of n: n places in the rest, plus out; in the toolbar: n + 1 places, plus staying out
            let n = items.len() + usize::from(!items.contains(&tile));
            assert_eq!(want.len(), n + 1, "the test's own count for {tile} on {items:?}");
            assert_eq!(candidate_states(&sp, &start, tile), want, "dragging {tile} on {items:?}");
        }
    }
}

#[test]
fn tile_futures_are_dragologys_on_every_list() {
    let sp = spec();
    let oracle = Oracle::new(&sp);
    let mut checked = 0;
    for s in sp.state_space().unwrap() {
        if oracle.breaks_invariant(&s) {
            continue;
        }
        for tile in TILES {
            assert_eq!(candidate_states(&sp, &s, tile), dragology_futures(&list(&s), tile), "dragging {tile} on {:?}", list(&s));
        }
        checked += 1;
    }
    assert_eq!(checked, 65);
}

/// Where tile `t` is drawn in `s`.
fn at(sp: &Spec, t: char, s: &State) -> (f64, f64) {
    element_position(sp, &t.to_string(), s).unwrap()
}

#[test]
fn pulling_a_tile_from_the_toolbar_inserts_it_where_it_is_dropped() {
    let sp = spec();
    let start = sp.initial.clone();
    // Grapes sits in its toolbar slot; drop it between the two apples.
    let target = state(&['a', 'd', 'b', 'c']);
    let (x0, y0) = at(&sp, 'd', &start);
    let (x1, y1) = at(&sp, 'd', &target);
    let path = [(x0, y0), ((x0 + x1) / 2.0, (y0 + y1) / 2.0), (x1 + 3.0, y1 - 2.0)];
    let (frames, fin) = drag(&sp, "drag_d", start, &path).unwrap();
    assert_eq!(list(&fin), vec!['a', 'd', 'b', 'c']);
    assert!(frames.iter().all(|f| !f.far));
}

#[test]
fn dropping_a_list_tile_on_its_toolbar_slot_removes_it() {
    let sp = spec();
    let start = sp.initial.clone();
    let gone = state(&['a', 'c']);
    let (x0, y0) = at(&sp, 'b', &start);
    let (x1, y1) = at(&sp, 'b', &gone);
    let path = [(x0, y0), (x0, (y0 + y1) / 2.0), (x1 - 2.0, y1 + 3.0)];
    let (_, fin) = drag(&sp, "drag_b", start, &path).unwrap();
    assert_eq!(fin, gone, "b is out of the list and a, c have closed up");
    // ... and it can be pulled straight back in, at the end this time.
    let back = state(&['a', 'c', 'b']);
    let (x2, y2) = at(&sp, 'b', &back);
    let (_, fin) = drag(&sp, "drag_b", gone.clone(), &[(x1, y1), (x2, y2)]).unwrap();
    assert_eq!(fin, back);
}

#[test]
fn a_drag_far_from_every_option_falls_back_to_the_start() {
    let sp = spec();
    let start = sp.initial.clone();
    let (x, y) = at(&sp, 'a', &start);
    // Dragology's `whenFar(d.fixed(state))`: well below the list, nothing is near enough to drop into.
    let (frames, fin) = drag(&sp, "drag_a", start.clone(), &[(x, y), (x + 20.0, y + 30.0), (x + 40.0, y + 140.0)]).unwrap();
    assert!(frames.last().unwrap().far);
    assert_eq!(fin, start);
    // The same far drop for a toolbar tile leaves it in the toolbar: nothing is added.
    let (x, y) = at(&sp, 'd', &start);
    let (frames, fin) = drag(&sp, "drag_d", start.clone(), &[(x, y), (x + 200.0, y + 10.0)]).unwrap();
    assert!(frames.last().unwrap().far);
    assert_eq!(fin, start);
}

#[test]
fn spec_passes_its_own_check() {
    let sp = spec();
    let problems = Oracle::new(&sp).self_check(&mut rand::rngs::StdRng::seed_from_u64(0)).unwrap();
    assert!(problems.is_empty(), "{problems:?}");
}
