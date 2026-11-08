// Skins: how each interface draws itself, in the drag lab and in the machine room. Positions still come
// from the spec's `render` block (the spec is the only source of truth for where things are); a skin adds
// the scenery under them (pegs, boards, tracks, columns) and gives each element the shape of the thing it
// is (a disc, a tile, a car, a card). Everything is drawn in the spec's own pixel coordinates; `c.k` is
// screen pixels per spec unit so strokes and text keep a readable size; `c.state`, `c.positions` and
// `c.active` (this element is being dragged) are available. An element skin returns true when it drew
// the element, else the page draws its default circle and label. `bounds` widens the page's view.
window.NERVE_SKINS = (() => {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v); if (parent) parent.append(e); return e; };
  const text = (parent, x, y, s, attrs = {}) => { const t = el("text", {x, y, "text-anchor": "middle", "dominant-baseline": "central", ...attrs}, parent); t.textContent = s; return t; };
  const rrect = (parent, cx, cy, w, h, r, attrs = {}) => el("rect", {x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: r, ...attrs}, parent);
  const PAL = i => `var(--f${((i % 10) + 10) % 10})`;
  const fs = (c, px = 11) => px / c.k;
  const hair = (c, px = 1.5) => px / c.k;
  const outline = c => ({stroke: c.active ? "var(--acc)" : "var(--ink)", "stroke-width": hair(c, c.active ? 2.5 : 1.5), "stroke-opacity": c.active ? 1 : 0.55});
  const label = (parent, x, y, s, c, px = 12, fill = "#fff") => text(parent, x, y, s, {"font-size": fs(c, px), fill, "font-weight": 600, "pointer-events": "none"});
  const dim = (parent, x, y, s, c, anchor = "middle") => text(parent, x, y, s, {"font-size": fs(c, 10), fill: "var(--mut)", "text-anchor": anchor});
  const line = (parent, x1, y1, x2, y2, c, attrs = {}) => el("line", {x1, y1, x2, y2, stroke: "var(--track)", "stroke-width": hair(c), ...attrs}, parent);
  const grid = (parent, x0, y0, cols, rows, cell, c) => {
    el("rect", {x: x0, y: y0, width: cols * cell, height: rows * cell, rx: 6, fill: "var(--panel)", stroke: "var(--line)", "stroke-width": hair(c)}, parent);
    for (let i = 1; i < cols; i++) line(parent, x0 + i * cell, y0, x0 + i * cell, y0 + rows * cell, c);
    for (let j = 1; j < rows; j++) line(parent, x0, y0 + j * cell, x0 + cols * cell, y0 + j * cell, c);
  };
  const track = (parent, x1, y1, x2, y2, c, w = 6) => el("line", {x1, y1, x2, y2, stroke: "var(--track)", "stroke-width": w, "stroke-linecap": "round"}, parent);
  const knob = (parent, p, c, r = 12, fill) => el("circle", {cx: p.x, cy: p.y, r, fill: fill || (c.active ? "var(--acc)" : "var(--panel)"), stroke: "var(--acc)", "stroke-width": hair(c, 2.5)}, parent);

  const skins = {
    toggle: {
      bounds: {x0: 0, y0: 0, x1: 120, y1: 60},
      scenery(g, c) {
        const on = c.state && c.state.status === "on";
        rrect(g, 60, 30, 92, 34, 17, {fill: on ? "var(--acc)" : "var(--track)", "fill-opacity": on ? 0.9 : 1});
        label(g, on ? 44 : 76, 30, on ? "on" : "off", c, 11, on ? "#fff" : "var(--ink)");
      },
      element(g, name, p, c) { knob(g, p, c, 14, "var(--panel)"); return true; },
    },

    three_way_switch: {
      bounds: {x0: 40, y0: 20, x1: 180, y1: 140},
      scenery(g, c) {
        const spots = [["r", 110, 50, PAL(4)], ["g", 76, 108, PAL(1)], ["b", 144, 108, PAL(0)]];
        el("polygon", {points: "110,50 76,108 144,108", fill: "none", stroke: "var(--line)", "stroke-width": hair(c), "stroke-dasharray": `${4 / c.k} ${4 / c.k}`}, g);
        for (const [n, x, y, col] of spots) {
          el("circle", {cx: x, cy: y, r: 17, fill: col, "fill-opacity": 0.18, stroke: col, "stroke-width": hair(c)}, g);
          dim(g, x, y + 30, n, c);
        }
        if (c.state && c.state.clockwise_only) dim(g, 110, 136, "clockwise only", c);
      },
      element(g, name, p, c) { knob(g, p, c, 12, c.active ? "var(--acc)" : "var(--ink)"); return true; },
    },

    counter: {
      bounds: {x0: 0, y0: 5, x1: 220, y1: 75},
      scenery(g, c) {
        const armed = !c.state || c.state.armed;
        track(g, 20, 40, 200, 40, c, 4);
        for (let i = 0; i <= 9; i++) { line(g, 20 + 20 * i, 34, 20 + 20 * i, 46, c, {stroke: "var(--mut)"}); dim(g, 20 + 20 * i, 60, String(i), c); }
        if (!armed) dim(g, 110, 16, "not armed: clicks do nothing", c);
      },
      element(g, name, p, c) {
        el("polygon", {points: `${p.x},${p.y - 4} ${p.x - 8},${p.y - 18} ${p.x + 8},${p.y - 18}`, fill: c.active ? "var(--acc)" : "var(--ink)"}, g);
        knob(g, p, c, 9);
        return true;
      },
    },

    slider: { bounds: {x0: -14, y0: -34, x1: 254, y1: 30}, scenery: sliderScenery, element: sliderKnob },
    slider_continuous: { bounds: {x0: -14, y0: -34, x1: 254, y1: 30}, scenery: sliderScenery, element: sliderKnob },

    linked_sliders: {
      bounds: {x0: 20, y0: 10, x1: 300, y1: 130},
      scenery(g, c) {
        [30, 70, 110].forEach((y, i) => {
          track(g, 40, y, 280, y, c, 6);
          const v = c.state ? c.state[`v${i + 1}`] : null;
          if (v !== null && v !== undefined) track(g, 40, y, 40 + Math.max(0, Math.min(240, v)), y, c, 6).setAttribute("stroke", "var(--acc)");
          dim(g, 28, y, `v${i + 1}`, c, "end");
        });
      },
      element(g, name, p, c) { knob(g, p, c, 11); return true; },
    },

    tracks: {
      bounds: {x0: -20, y0: -30, x1: 262, y1: 192},
      scenery(g, c) {
        for (const y of [0, 80, 160]) track(g, 0, y, 240, y, c, 8);
        track(g, 60, 80, 180, 80, c, 8).setAttribute("stroke", "var(--ghost)");
        dim(g, 120, 104, "the middle track only reaches from 60 to 180", c);
        for (const x of [0, 60, 120, 180, 240]) dim(g, x, -16, String(x), c);
      },
      element(g, name, p, c) { rrect(g, p.x, p.y, 30, 30, 7, {fill: c.active ? "var(--acc)" : PAL(2), ...outline(c)}); return true; },
    },

    reorder: {
      bounds: {x0: 0, y0: 0, x1: 300, y1: 80},
      scenery(g, c) {
        rrect(g, 168, 40, 316, 68, 10, {fill: "var(--panel)", stroke: "var(--line)", "stroke-width": hair(c)});
        for (let i = 0; i < 5; i++) rrect(g, 40 + 64 * i, 40, 54, 54, 8, {fill: "none", stroke: "var(--line)", "stroke-width": hair(c), "stroke-dasharray": `${3 / c.k} ${3 / c.k}`});
      },
      element(g, name, p, c) { card(g, p, name, c, 50, 50); return true; },
    },

    insert_remove: {
      bounds: {x0: -10, y0: -5, x1: 240, y1: 150},
      scenery(g, c) {
        rrect(g, 105, 30, 220, 48, 8, {fill: "var(--panel)", stroke: "var(--line)", "stroke-width": hair(c)});
        rrect(g, 105, 110, 220, 48, 8, {fill: "var(--panel)", stroke: "var(--line)", "stroke-width": hair(c)});
        dim(g, 222, 30, "toolbar", c, "start"); dim(g, 222, 110, "list", c, "start");
        for (let i = 0; i < 4; i++) { rrect(g, 30 + 50 * i, 30, 40, 40, 6, {fill: "none", stroke: "var(--line)", "stroke-width": hair(c), "stroke-dasharray": `${3 / c.k} ${3 / c.k}`}); rrect(g, 30 + 50 * i, 110, 40, 40, 6, {fill: "none", stroke: "var(--line)", "stroke-width": hair(c), "stroke-dasharray": `${3 / c.k} ${3 / c.k}`}); }
      },
      element(g, name, p, c) {
        const fruit = {a: "🍎", b: "🍎", c: "🍌", d: "🍇"}[name] || name;
        rrect(g, p.x, p.y, 40, 40, 8, {fill: "var(--panel)", ...outline(c)});
        text(g, p.x, p.y + 1, fruit, {"font-size": fs(c, 20), "pointer-events": "none"});
        return true;
      },
    },

    fifteen: {
      bounds: {x0: 14, y0: 14, x1: 206, y1: 146},
      scenery(g, c) { grid(g, 20, 20, 3, 2, 60, c); },
      element(g, name, p, c) {
        if (name === "blank") { rrect(g, p.x, p.y, 52, 52, 6, {fill: "var(--bg)", stroke: "var(--mut)", "stroke-width": hair(c), "stroke-dasharray": `${4 / c.k} ${3 / c.k}`}); return true; }
        const n = name.replace("tile", "");
        rrect(g, p.x, p.y, 52, 52, 6, {fill: c.active ? "var(--acc)" : PAL(+n), ...outline(c)});
        label(g, p.x, p.y, n, c, 20);
        return true;
      },
    },

    hanoi: {
      bounds: {x0: 0, y0: 75, x1: 420, y1: 240},
      scenery(g, c) {
        rrect(g, 210, 220, 396, 12, 4, {fill: "var(--track)"});
        for (let i = 0; i < 3; i++) { rrect(g, 70 + 140 * i, 154, 8, 124, 3, {fill: "var(--track)"}); dim(g, 70 + 140 * i, 236, ["left", "middle", "right"][i] + " peg", c); }
      },
      element(g, name, p, c) {
        const i = +name.replace("disk", "");
        rrect(g, p.x, p.y, 36 + 22 * (i - 1), 24, 8, {fill: c.active ? "var(--acc)" : PAL(i - 1), ...outline(c)});
        label(g, p.x, p.y, String(i), c, 11);
        return true;
      },
    },

    rush_hour: {
      bounds: {x0: 20, y0: 20, x1: 345, y1: 290},
      scenery(g, c) {
        grid(g, 30, 30, 5, 5, 50, c);
        // The exit: row 2 opens to the right, where only the red car may drive out.
        el("rect", {x: 279, y: 132, width: 4, height: 46, fill: "var(--bg)"}, g);
        el("polygon", {points: "292,155 318,155 318,147 334,155 318,163 318,155", fill: PAL(4), "fill-opacity": 0.6}, g);
        dim(g, 313, 180, "exit", c);
      },
      element(g, name, p, c) {
        const size = {red: [92, 40], a: [142, 40], b: [40, 92], c: [40, 92], d: [92, 40], e: [92, 40]}[name] || [40, 40];
        const fill = name === "red" ? PAL(4) : {a: PAL(0), b: PAL(1), c: PAL(5), d: PAL(2), e: PAL(3)}[name];
        rrect(g, p.x, p.y, size[0], size[1], 9, {fill: c.active ? "var(--acc)" : fill, ...outline(c)});
        label(g, p.x, p.y, name === "red" ? "red" : name, c, 11);
        return true;
      },
    },

    sokoban: {
      bounds: {x0: 14, y0: 14, x1: 326, y1: 206},
      scenery(g, c) {
        grid(g, 20, 20, 5, 3, 60, c);
        for (const cell of [0, 1, 10, 11]) rrect(g, 50 + 60 * (cell % 5), 50 + 60 * Math.floor(cell / 5), 58, 58, 2, {fill: "var(--track)"});
        for (const cell of [5, 14]) el("circle", {cx: 50 + 60 * (cell % 5), cy: 50 + 60 * Math.floor(cell / 5), r: 14, fill: "none", stroke: PAL(1), "stroke-width": hair(c, 2), "stroke-dasharray": `${4 / c.k} ${3 / c.k}`}, g);
      },
      element(g, name, p, c) {
        if (name === "player") { knob(g, p, c, 17, c.active ? "var(--acc)" : PAL(0)); label(g, p.x, p.y, "you", c, 10); return true; }
        const onGoal = c.state && [5, 14].includes(c.state[name === "box1" ? "b1" : "b2"]);
        rrect(g, p.x, p.y, 44, 44, 5, {fill: onGoal ? PAL(1) : PAL(9), ...outline(c)});
        line(g, p.x - 14, p.y - 14, p.x + 14, p.y + 14, c, {stroke: "#fff", "stroke-opacity": 0.5});
        line(g, p.x + 14, p.y - 14, p.x - 14, p.y + 14, c, {stroke: "#fff", "stroke-opacity": 0.5});
        return true;
      },
    },

    kanban: {
      bounds: {x0: -10, y0: 0, x1: 435, y1: 245},
      scenery(g, c) {
        ["to do", "doing", "done"].forEach((h, i) => {
          rrect(g, 60 + 150 * i, 122, 132, 238, 10, {fill: "var(--panel)", stroke: "var(--line)", "stroke-width": hair(c)});
          text(g, 60 + 150 * i, 16, h, {"font-size": fs(c, 11), fill: "var(--mut)", "font-weight": 600, "letter-spacing": "0.08em"});
        });
      },
      element(g, name, p, c) { card(g, p, name, c, 112, 36); return true; },
    },

    todo: {
      bounds: {x0: 20, y0: 25, x1: 225, y1: 265},
      scenery(g, c) {
        rrect(g, 122, 142, 200, 236, 10, {fill: "var(--panel)", stroke: "var(--line)", "stroke-width": hair(c)});
        for (let i = 0; i < 4; i++) line(g, 36, 88 + 55 * i, 208, 88 + 55 * i, c, {stroke: "var(--line)"});
        dim(g, 60, 36, "to do", c); dim(g, 110, 36, "done", c);
      },
      element(g, name, p, c) {
        const done = p.x > 85;
        rrect(g, p.x + 30, p.y, 120, 40, 8, {fill: "var(--panel)", ...outline(c)});
        rrect(g, p.x - 14, p.y, 18, 18, 4, {fill: done ? PAL(1) : "none", stroke: done ? PAL(1) : "var(--mut)", "stroke-width": hair(c)});
        if (done) label(g, p.x - 14, p.y, "✓", c, 12);
        text(g, p.x + 24, p.y, name, {"font-size": fs(c, 13), fill: done ? "var(--mut)" : "var(--ink)", "text-decoration": done ? "line-through" : null, "pointer-events": "none"});
        return true;
      },
    },

    scheduler: {
      bounds: {x0: 35, y0: 5, x1: 385, y1: 345},
      scenery(g, c) {
        [0, 1].forEach(room => {
          const y0 = 15 + 170 * room;
          rrect(g, 210, y0 + 72, 330, 150, 10, {fill: "var(--panel)", stroke: "var(--line)", "stroke-width": hair(c)});
          dim(g, 50, y0 + 8, `room ${room}`, c, "start");
          for (let s = 0; s <= 3; s++) { line(g, 60 + 100 * s, y0 + 14, 60 + 100 * s, y0 + 146, c); dim(g, 60 + 100 * s, y0 + 160, `slot ${s}`, c); }
        });
      },
      element(g, name, p, c) {
        const n = name.replace(/\D/g, "");
        if (name.startsWith("bar")) {
          const l = c.positions[`left${n}`], r = c.positions[`right${n}`];
          if (l && r) rrect(g, (l.x + r.x) / 2, p.y, Math.abs(r.x - l.x), 26, 8, {fill: PAL(+n + 2), "fill-opacity": c.active ? 0.95 : 0.55, ...outline(c)});
          label(g, p.x, p.y, `meeting ${n}`, c, 10, "var(--ink)");
          return true;
        }
        el("circle", {cx: p.x, cy: p.y, r: 8, fill: c.active ? "var(--acc)" : PAL(+n + 2), stroke: "var(--panel)", "stroke-width": hair(c, 2)}, g);
        return true;
      },
    },

    negotiate: {
      bounds: {x0: 0, y0: 0, x1: 270, y1: 165},
      scenery(g, c) {
        [["books", 40], ["hats", 120]].forEach(([n, y]) => {
          track(g, 40, y, 200, y, c, 4);
          for (let i = 0; i <= 2; i++) dim(g, 40 + 80 * i, y + 26, `keep ${i}`, c);
          dim(g, 240, y, n, c);
        });
        if (c.state) dim(g, 130, 155, `${c.state.turn} proposes · ${String(c.state.status).replace("_", " ")} · round ${c.state.rounds}`, c);
      },
      element(g, name, p, c) {
        rrect(g, p.x, p.y, 44, 34, 8, {fill: "var(--panel)", ...outline(c)});
        text(g, p.x, p.y + 1, name === "books" ? "📚" : "🎩", {"font-size": fs(c, 18), "pointer-events": "none"});
        return true;
      },
    },

    clock: {
      bounds: {x0: 0, y0: 5, x1: 185, y1: 55},
      scenery(g, c) {
        track(g, 24, 30, 168, 30, c, 4);
        for (let i = 0; i < 10; i++) line(g, 24 + 16 * i, 25, 24 + 16 * i, 35, c, {stroke: "var(--mut)"});
      },
      element(g, name, p, c) { el("polygon", {points: `${p.x},${p.y + 2} ${p.x - 7},${p.y - 14} ${p.x + 7},${p.y - 14}`, fill: "var(--acc)"}, g); return true; },
    },
  };

  function sliderScenery(g, c) {
    track(g, 0, 0, 240, 0, c, 6);
    const v = c.state ? Number(c.state.value) : null;
    if (v !== null && !Number.isNaN(v)) track(g, 0, 0, Math.max(0, Math.min(240, v)), 0, c, 6).setAttribute("stroke", "var(--acc)");
    for (let x = 0; x <= 240; x += 40) { line(g, x, -6, x, 6, c, {stroke: "var(--mut)"}); dim(g, x, 20, String(x), c); }
  }
  function sliderKnob(g, name, p, c) {
    knob(g, p, c, 12, c.state && c.state.grabbed ? "var(--acc)" : undefined);
    if (c.state && c.state.value !== undefined) dim(g, p.x, -22, Number(c.state.value).toFixed(c.state.value % 1 ? 1 : 0), c);
    return true;
  }
  function card(g, p, name, c, w, h) {
    rrect(g, p.x, p.y, w, h, 8, {fill: c.active ? "var(--acc)" : "var(--panel)", ...outline(c)});
    text(g, p.x, p.y, name, {"font-size": fs(c, 14), "font-weight": 600, fill: c.active ? "#fff" : "var(--ink)", "pointer-events": "none"});
  }

  return { get: name => skins[name] || null };
})();
