"""Drags as energy minimisation over spec-derived futures (nerve/spec/drag.py, docs/dragology.md)."""
import pytest

from nerve.spec import load_spec, Oracle
from nerve.spec.drag import DragSession, drag, element_position
from nerve.spec.schema import Future


def test_toggle_between_commits_past_the_midpoint():
    sp = load_spec("specs/toggle.nerve")
    frames, final = drag(sp, "drag_knob", {"status": "off"}, [(30, 30), (50, 30), (59, 30)])
    assert final == {"status": "off"}                               # never crossed the midpoint (x = 60)
    assert frames[-1].weights == pytest.approx({"off": 1 - 29 / 60, "on": 29 / 60})
    frames, final = drag(sp, "drag_knob", {"status": "off"}, [(30, 30), (61, 30), (95, 30)])
    assert final == {"status": "on"}
    assert frames[1].branch == "on" and frames[0].branch == "off"


def test_candidates_are_derived_from_the_spec_not_listed():
    sp = load_spec("specs/counter.nerve")
    sess = DragSession(sp, sp.futures["drag_dial"], {"count": 3, "armed": False})
    c = sess.candidates()
    assert set(c) == set(range(10))                                 # every value `set` can reach
    assert c[7].action == ("set", {"to": 7}) and c[3].action is None   # staying needs no action
    assert all(not Oracle(sp).violations({"count": 3, "armed": False}, cand.action, cand.state) for cand in c.values() if cand.action)


def test_closest_with_stickiness_holds_the_incumbent():
    sp = load_spec("specs/counter.nerve")                           # dial at x = 20 + 20*count, stickiness 4 px
    sess = DragSession(sp, sp.futures["drag_dial"], {"count": 3, "armed": True})
    assert sess.frame((80, 40)).branch == 3                         # exactly on 3
    assert sess.frame((91, 40)).branch == 3                         # 1 px past the midpoint to 4: incumbent holds
    assert sess.frame((93, 40)).branch == 4                         # 3 px past: 4 is better by more than stickiness
    assert sess.release() == {"count": 4, "armed": True}
    # without stickiness the same path flips at the midpoint
    f = Future("d", "closest", "count", list(range(10)), grab="dial", action="set")
    sess2 = DragSession(sp, f, {"count": 3, "armed": True})
    sess2.frame((80, 40)); assert sess2.frame((91, 40)).branch == 4


def test_vary_tracks_the_pointer_through_the_spec_clamp():
    sp = load_spec("specs/slider_continuous.nerve")
    s = {"value": 10.0, "grabbed": True}
    frames, final = drag(sp, "drag_handle", s, [(50, 0), (137.2, 0)])
    assert final["value"] == pytest.approx(137.2, abs=0.05)
    frames, final = drag(sp, "drag_handle", s, [(400, 0)])          # past the range: the spec's clamp decides
    assert final["value"] == pytest.approx(240.0, abs=0.05)
    # the guard is the spec's too: not grabbed -> `move` is a no-op, the drag goes nowhere
    frames, final = drag(sp, "drag_handle", {"value": 10.0, "grabbed": False}, [(137.2, 0)])
    assert final["value"] == 10.0


def test_gridded_vary_snaps_to_the_grid():
    sp = load_spec("specs/slider.nerve")
    _, final = drag(sp, "drag_handle", {"value": 0.0, "grabbed": True}, [(133, 0)])
    assert final["value"] == 130.0


def test_far_falls_back_to_the_start_with_hysteresis():
    sp = load_spec("specs/toggle.nerve")
    f = Future("d", "between", "status", ["off", "on"], grab="knob", far=40.0)
    sess = DragSession(sp, f, {"status": "off"})
    assert sess.frame((95, 30)).branch == "on"                     # near the `on` knob position (90, 30)
    fr = sess.frame((90, 120))                                     # 90 px below: far -> back to the start state
    assert fr.far and fr.preview == {"status": "off"}
    assert sess.frame((90, 75)).far                                # 45 px: still far (hysteresis, comes back at 40)
    assert not sess.frame((90, 65)).far and sess.release() == {"status": "on"}


def test_render_block_is_checked():
    from nerve.spec.parse import parse_spec, SpecError
    import yaml
    doc = yaml.safe_load(open("specs/toggle.nerve"))
    doc["render"] = {"knob": {"x": "__import__('os')", "y": 0}}
    with pytest.raises(Exception):
        parse_spec(doc)
    doc["render"] = {"knob": {"x": "1"}}
    with pytest.raises(SpecError):
        parse_spec(doc)
    assert element_position(load_spec("specs/toggle.nerve"), "knob", {"status": "on"}) == (90.0, 30.0)
