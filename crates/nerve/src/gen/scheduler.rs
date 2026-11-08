//! Dragology's `scheduler` demo (src/demos/scheduler.tsx): meetings on a timeline, one track per room.
//! Each meeting is a bar with a dot at each end. Dragging the bar moves the meeting to any room;
//! dragging a dot moves that end alone, never off the timeline and never closer than MIN_LENGTH to the
//! other end. Overlaps are allowed. Scaled down so every tuple can be enumerated: 3 meetings, 2 rooms,
//! a timeline of 3 whole slots with a one-slot minimum length.

use super::{generated_by, l, text};

const MEETINGS: usize = 3;
const ROOMS: usize = 2;
const SLOTS: usize = 3;
const MIN_LEN: usize = 1;
/// (room, start, end): Dragology's first three meetings overlap in a chain without a double booking.
const INITIAL: [(usize, usize, usize); 3] = [(0, 0, 2), (1, 1, 3), (1, 0, 1)];
const X0: usize = 60;
const SLOT_W: usize = 100;
const Y0: usize = 40;
const ROOM_H: usize = 170;
const ROW_H: usize = 48;

fn fields(i: usize) -> Vec<String> {
    vec![format!("room{i}"), format!("start{i}"), format!("end{i}")]
}

fn unchanged(names: &[String]) -> String {
    names.iter().map(|f| format!("{f} == old.{f}")).collect::<Vec<_>>().join(" and ")
}

fn others(i: usize) -> Vec<String> {
    (0..MEETINGS).filter(|&j| j != i).flat_map(fields).collect()
}

fn row_y(i: usize) -> String {
    format!("{} + {ROOM_H} * room{i}", Y0 + ROW_H * i)
}

pub fn spec() -> String {
    let mut lines = vec![
        l("# Dragology's meeting scheduler (src/demos/scheduler.tsx): meetings as bars on a timeline, one track per room."),
        l("# The rule: drag a bar to move its meeting to any room; drag the dot at either end to move that end alone,"),
        l("# never off the timeline and never closer than MIN_LENGTH to the other end. Meetings may overlap, even in"),
        l("# one room: Dragology draws the overlap graph and marks a double booking in red, it never forbids one (the"),
        l("# graph is not drawn here, render has points only). Scaled down: 3 meetings (of 5), 2 rooms (of 4), a"),
        l("# 3-slot timeline (of 250 px) with ends on whole slots and a one-slot minimum, so 1,728 of the 5,832 tuples"),
        l("# are schedules. On the grid an end's `vary` becomes `closest` over its slot; the bar's drag is `between`."),
        l("# Each meeting has its own row inside a room's band so that no two handles ever coincide."),
        generated_by("scheduler"),
        l("nerve: 1"),
        l("name: scheduler"),
        l("state:"),
    ];
    for i in 0..MEETINGS {
        lines.push(format!("  room{i}: {{type: int, range: [0, {}]}}   # meeting {i}'s room", ROOMS - 1));
        lines.push(format!("  start{i}: {{type: int, range: [0, {}]}}   # meeting {i} starts at this slot", SLOTS - MIN_LEN));
        lines.push(format!("  end{i}: {{type: int, range: [{MIN_LEN}, {SLOTS}]}}   # ... and ends at this one"));
    }
    lines.push(l("actions:"));
    for i in 0..MEETINGS {
        lines.push(format!("  set_room{i}: {{params: {{to: {{type: int, range: [0, {}]}}}}}}   # drag meeting {i}'s bar", ROOMS - 1));
        lines.push(format!("  set_start{i}: {{params: {{to: {{type: int, range: [0, {}]}}}}}}   # drag its left dot", SLOTS - MIN_LEN));
        lines.push(format!("  set_end{i}: {{params: {{to: {{type: int, range: [{MIN_LEN}, {SLOTS}]}}}}}}   # drag its right dot"));
    }
    lines.push(l("transitions:"));
    for i in 0..MEETINGS {
        lines.push(format!("  set_room{i}: {{ room{i}: \"to\" }}"));
        lines.push(format!("  set_start{i}: {{ when: \"to + {MIN_LEN} <= end{i}\", start{i}: \"to\" }}"));
        lines.push(format!("  set_end{i}: {{ when: \"start{i} + {MIN_LEN} <= to\", end{i}: \"to\" }}"));
    }
    lines.push(l("invariants:"));
    lines.push(l("  # Dragology's vary constraints: lessThan(0, start), lessThan(start, end - MIN_LENGTH), lessThan(end, END_TIME)"));
    lines.extend((0..MEETINGS).map(|i| format!("  - \"0 <= start{i} and start{i} + {MIN_LEN} <= end{i} <= {SLOTS}\"")));
    lines.push(l("ensures:"));
    for i in 0..MEETINGS {
        let mut keep_time = vec![format!("start{i}"), format!("end{i}")];
        keep_time.extend(others(i));
        lines.push(format!("  - {{ after: set_room{i}, holds: \"room{i} == to and {}\" }}   # the meeting changes room, keeps its time, and no other meeting moves", unchanged(&keep_time)));
        let mut keep_start = vec![format!("room{i}"), format!("end{i}")];
        keep_start.extend(others(i));
        lines.push(format!("  - {{ after: set_start{i}, holds: \"(start{i} == to or start{i} == old.start{i}) and {}\" }}   # only that end moves", unchanged(&keep_start)));
        let mut keep_end = vec![format!("room{i}"), format!("start{i}")];
        keep_end.extend(others(i));
        lines.push(format!("  - {{ after: set_end{i}, holds: \"(end{i} == to or end{i} == old.end{i}) and {}\" }}   # only that end moves", unchanged(&keep_end)));
    }
    lines.push(l("scene:"));
    lines.push(l("  box: [35, 5, 385, 345]"));
    lines.push(l("  shapes:"));
    for room in 0..ROOMS {
        let y0 = 15 + ROOM_H * room;
        lines.push(format!("    - {{rect: [210, {}, 330, 150], r: 10, color: panel, stroke: line, width: 1.5}}", y0 + 72));
        lines.push(format!("    - {{label: room {room}, at: [70, {}], color: mut, size: 1.8}}", y0 + 8));
        for s in 0..=SLOTS {
            let x = X0 + SLOT_W * s;
            lines.push(format!("    - {{line: [{x}, {}, {x}, {}], color: line, width: 1.5}}", y0 + 14, y0 + 146));
            lines.push(format!("    - {{label: slot {s}, at: [{x}, {}], color: mut, size: 1.6}}", y0 + 160));
        }
    }
    lines.push(l("render:"));
    for i in 0..MEETINGS {
        lines.push(format!(
            "  bar{i}: {{ x: \"{X0} + {} * (start{i} + end{i})\", y: \"{}\", shapes: [{{rect: [0, 0, \"{SLOT_W} * (end{i} - start{i})\", 26], r: 8, color: f{}, opacity: 0.55}}, {{label: meeting {i}, at: [0, 0], color: ink, size: 1.6}}] }}   # middle of the bar",
            SLOT_W / 2,
            row_y(i),
            i + 2
        ));
        lines.push(format!("  left{i}: {{ x: \"{X0} + {SLOT_W} * start{i}\", y: \"{}\", shape: {{circle: [0, 0, 8], color: f{}, stroke: panel, width: 2}} }}", row_y(i), i + 2));
        lines.push(format!("  right{i}: {{ x: \"{X0} + {SLOT_W} * end{i}\", y: \"{}\", shape: {{circle: [0, 0, 8], color: f{}, stroke: panel, width: 2}} }}", row_y(i), i + 2));
    }
    lines.push(l("futures:"));
    lines.push(l("  # per meeting: the bar goes to any room (Dragology `between`, no snap radius here); each dot to every slot"));
    lines.push(l("  # its end may take (Dragology `vary` with constraints, on the slot grid)"));
    for i in 0..MEETINGS {
        lines.push(format!("  drag_bar{i}: {{ kind: between, over: room{i}, grab: bar{i}, action: set_room{i} }}"));
        lines.push(format!("  drag_left{i}: {{ kind: closest, over: start{i}, grab: left{i}, action: set_start{i}, stickiness: 6 }}"));
        lines.push(format!("  drag_right{i}: {{ kind: closest, over: end{i}, grab: right{i}, action: set_end{i}, stickiness: 6 }}"));
    }
    lines.push(format!(
        "initial: {{{}}}",
        INITIAL.iter().enumerate().map(|(i, (r, s, e))| format!("room{i}: {r}, start{i}: {s}, end{i}: {e}")).collect::<Vec<_>>().join(", ")
    ));
    text(lines)
}
