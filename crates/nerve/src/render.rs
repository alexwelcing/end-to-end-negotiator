//! The render oracle: what an interface looks like, declared in the spec and rasterised exactly.
//!
//! A spec's `scene:` gives the picture's box, background and static shapes (pegs, a board, a track);
//! each `render:` element may carry `shape`/`shapes` drawn at the element's position (a disc, a tile,
//! a card). Shapes are rectangles, circles, lines and text in a small bitmap font, in named colours;
//! a shape may have a `when` predicate over the state and text may be an expression. This module turns
//! a state into pixels the same way the oracle turns a state and an action into the next state: it is
//! the teacher of the render network (`picture`) and the check on what that network draws.

use crate::expr::Expr;
use crate::oracle::StateScope;
use crate::spec::{Spec, State};
use crate::value::Value;
use anyhow::{bail, ensure, Context, Result};
use serde_yaml::{Mapping, Value as Yaml};

/// RGB in 0..1.
pub type Color = [f32; 3];

/// The longest side of a raster, in pixels; both sides are multiples of `RASTER_STEP`.
pub const RASTER_MAX: usize = 192;
pub const RASTER_STEP: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    Rect,
    Circle,
    Line,
    Text,
}

/// Geometry is a number or an expression over the state, so a bar can be as long as a value.
#[derive(Debug, Clone)]
pub struct Shape {
    pub kind: Kind,
    /// Centre (rect, circle, text) or start (line); relative to the element for element shapes.
    pub x: Expr,
    pub y: Expr,
    /// Line end.
    pub x2: Expr,
    pub y2: Expr,
    pub w: Expr,
    pub h: Expr,
    /// Corner radius (rect), radius (circle), half line width (line), cell size (text).
    pub r: Expr,
    /// False for `color: none`: only the stroke is drawn.
    pub fill: bool,
    pub color: Color,
    pub opacity: f32,
    pub stroke: Option<(Color, f64)>,
    pub when: Option<Expr>,
    /// Static text, or an expression whose value is drawn.
    pub label: Option<String>,
    pub text: Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct Scene {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
    pub background: Color,
    pub shapes: Vec<Shape>,
}

impl Scene {
    /// The raster this scene is drawn into: at most `RASTER_MAX` on the long side, sides multiples of
    /// `RASTER_STEP`, as near the scene's aspect as that allows.
    pub fn raster_size(&self) -> (usize, usize) {
        let (bw, bh) = (self.x1 - self.x0, self.y1 - self.y0);
        let s = RASTER_MAX as f64 / bw.max(bh).max(1.0);
        let fit = |v: f64| (((v * s / RASTER_STEP as f64).round() as usize).max(1)) * RASTER_STEP;
        (fit(bw), fit(bh))
    }
}

/// The lab's light palette, by the names the pages use, plus `#rrggbb`.
pub fn color(name: &str) -> Result<Color> {
    let hex = match name {
        "bg" => "#f6f7fb",
        "panel" | "white" => "#ffffff",
        "ink" => "#141824",
        "mut" | "mute" => "#646b7d",
        "line" => "#dfe3ec",
        "track" => "#c9cfdc",
        "acc" => "#3b6cf6",
        "learn" => "#9b5cf6",
        "ok" => "#12a17a",
        "bad" => "#e5484d",
        "ghost" => "#8da2d8",
        "f0" => "#3b6cf6",
        "f1" => "#12a17a",
        "f2" => "#e8833a",
        "f3" => "#9b5cf6",
        "f4" => "#e5484d",
        "f5" => "#0aa2c0",
        "f6" => "#c2a20a",
        "f7" => "#d6409f",
        "f8" => "#5c7a29",
        "f9" => "#7a5c3a",
        "black" => "#000000",
        h if h.starts_with('#') && h.len() == 7 => h,
        other => bail!("unknown colour {other:?}"),
    };
    let v = u32::from_str_radix(&hex[1..], 16).with_context(|| format!("colour {hex}"))?;
    Ok([((v >> 16) & 255) as f32 / 255.0, ((v >> 8) & 255) as f32 / 255.0, (v & 255) as f32 / 255.0])
}

fn get<'a>(m: &'a Mapping, k: &str) -> Option<&'a Yaml> {
    m.get(Yaml::String(k.to_string()))
}

fn num(y: &Yaml) -> Result<f64> {
    y.as_f64().or_else(|| y.as_i64().map(|i| i as f64)).with_context(|| format!("expected a number, got {y:?}"))
}

fn nums(y: &Yaml, n: usize, what: &str) -> Result<Vec<f64>> {
    let Yaml::Sequence(s) = y else { bail!("{what} must be a list of {n} numbers") };
    ensure!(s.len() == n, "{what} must have {n} numbers, has {}", s.len());
    s.iter().map(num).collect()
}

/// A number, or a string holding an expression over the state.
fn num_expr(y: &Yaml) -> Result<Expr> {
    match y {
        Yaml::String(e) => Expr::compile(e),
        other => Expr::compile(&crate::value::float_repr(num(other)?)),
    }
}

fn exprs(y: &Yaml, n: usize, what: &str) -> Result<Vec<Expr>> {
    let Yaml::Sequence(s) = y else { bail!("{what} must be a list of {n} numbers or expressions") };
    ensure!(s.len() == n, "{what} must have {n} entries, has {}", s.len());
    s.iter().map(num_expr).collect()
}

fn zero() -> Expr {
    Expr::compile("0").expect("a constant")
}

/// `color: none` draws no fill (the stroke, if any, still shows).
fn opt_color(m: &Mapping, k: &str, default: &str) -> Result<(Color, bool)> {
    match get(m, k) {
        Some(Yaml::String(s)) if s == "none" => Ok((color(default)?, false)),
        Some(Yaml::String(s)) => Ok((color(s)?, true)),
        None => Ok((color(default)?, true)),
        Some(other) => bail!("{k} must be a colour name, got {other:?}"),
    }
}

fn expr_src(y: &Yaml) -> Result<String> {
    Ok(match y {
        Yaml::String(s) => s.clone(),
        Yaml::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Yaml::Number(n) => n.to_string(),
        other => bail!("expected an expression, got {other:?}"),
    })
}

impl Shape {
    /// One shape from its YAML mapping: `{rect: [cx, cy, w, h]}`, `{circle: [cx, cy, r]}`,
    /// `{line: [x1, y1, x2, y2]}`, `{label: "text", at: [x, y]}` or `{text: "expr", at: [x, y]}`,
    /// with `color`, `opacity`, `stroke` and `width`, `r` (corner radius), `size` (text) and `when`.
    pub fn parse(y: &Yaml) -> Result<Shape> {
        let Yaml::Mapping(m) = y else { bail!("a shape must be a mapping, got {y:?}") };
        let mut s = Shape {
            kind: Kind::Rect,
            x: zero(),
            y: zero(),
            x2: zero(),
            y2: zero(),
            w: zero(),
            h: zero(),
            r: zero(),
            fill: true,
            color: color("ink")?,
            opacity: 1.0,
            stroke: None,
            when: None,
            label: None,
            text: None,
        };
        let constant = |v: f64| Expr::compile(&crate::value::float_repr(v)).expect("a constant");
        if let Some(v) = get(m, "rect") {
            let mut n = exprs(v, 4, "rect")?;
            s.kind = Kind::Rect;
            s.h = n.pop().unwrap();
            s.w = n.pop().unwrap();
            s.y = n.pop().unwrap();
            s.x = n.pop().unwrap();
            s.r = get(m, "r").map(num_expr).transpose()?.unwrap_or_else(zero);
        } else if let Some(v) = get(m, "circle") {
            let mut n = exprs(v, 3, "circle")?;
            s.kind = Kind::Circle;
            s.r = n.pop().unwrap();
            s.y = n.pop().unwrap();
            s.x = n.pop().unwrap();
        } else if let Some(v) = get(m, "line") {
            let mut n = exprs(v, 4, "line")?;
            s.kind = Kind::Line;
            s.y2 = n.pop().unwrap();
            s.x2 = n.pop().unwrap();
            s.y = n.pop().unwrap();
            s.x = n.pop().unwrap();
            s.r = constant(get(m, "width").map(num).transpose()?.unwrap_or(2.0) / 2.0);
        } else if get(m, "label").is_some() || get(m, "text").is_some() {
            s.kind = Kind::Text;
            let mut at = get(m, "at").map(|v| exprs(v, 2, "at")).transpose()?.unwrap_or_else(|| vec![zero(), zero()]);
            s.y = at.pop().unwrap();
            s.x = at.pop().unwrap();
            s.r = constant(get(m, "size").map(num).transpose()?.unwrap_or(2.2));
            if let Some(l) = get(m, "label") {
                s.label = Some(match l {
                    Yaml::String(t) => t.clone(),
                    other => expr_src(other)?,
                });
            }
            if let Some(t) = get(m, "text") {
                s.text = Some(Expr::compile(&expr_src(t)?)?);
            }
        } else {
            bail!("a shape needs one of rect, circle, line, label or text: {y:?}");
        }
        (s.color, s.fill) = opt_color(m, "color", if s.kind == Kind::Text { "ink" } else { "acc" })?;
        s.opacity = get(m, "opacity").map(num).transpose()?.unwrap_or(1.0) as f32;
        if let Some(Yaml::String(c)) = get(m, "stroke") {
            s.stroke = Some((color(c)?, get(m, "width").map(num).transpose()?.unwrap_or(2.0)));
        }
        if let Some(w) = get(m, "when") {
            s.when = Some(Expr::compile(&expr_src(w)?)?);
        }
        Ok(s)
    }
}

impl Scene {
    /// `scene: {box: [x0, y0, x1, y1], background: bg, shapes: [...]}`.
    pub fn parse(y: &Yaml) -> Result<Scene> {
        let Yaml::Mapping(m) = y else { bail!("scene must be a mapping") };
        let b = nums(get(m, "box").context("scene needs `box: [x0, y0, x1, y1]`")?, 4, "scene box")?;
        ensure!(b[2] > b[0] && b[3] > b[1], "scene box must have positive size");
        let shapes = get(m, "shapes").and_then(Yaml::as_sequence).map(|s| s.iter().map(Shape::parse).collect::<Result<Vec<_>>>()).transpose()?.unwrap_or_default();
        Ok(Scene { x0: b[0], y0: b[1], x1: b[2], y1: b[3], background: opt_color(m, "background", "bg")?.0, shapes })
    }
}

/// A 3x5 bitmap font: digits, lowercase letters and a few marks, as rows of three bits.
fn glyph(c: char) -> Option<[u8; 5]> {
    Some(match c.to_ascii_lowercase() {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        'a' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'b' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'c' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'd' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'e' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'f' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'g' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'h' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'i' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'j' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'k' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'l' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'm' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'n' => [0b110, 0b101, 0b101, 0b101, 0b101],
        'o' => [0b010, 0b101, 0b101, 0b101, 0b010],
        'p' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'q' => [0b010, 0b101, 0b101, 0b111, 0b011],
        'r' => [0b110, 0b101, 0b110, 0b101, 0b101],
        's' => [0b011, 0b100, 0b010, 0b001, 0b110],
        't' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'u' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'v' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'w' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'x' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '=' => [0b000, 0b111, 0b000, 0b111, 0b000],
        '+' => [0b000, 0b010, 0b111, 0b010, 0b000],
        '%' => [0b101, 0b001, 0b010, 0b100, 0b101],
        '<' => [0b001, 0b010, 0b100, 0b010, 0b001],
        '>' => [0b100, 0b010, 0b001, 0b010, 0b100],
        ' ' => [0; 5],
        _ => return None,
    })
}

/// A rasteriser over one RGB buffer in CHW order.
struct Canvas {
    w: usize,
    h: usize,
    px: Vec<f32>,
    /// Spec units to pixels, and the spec-space origin of pixel (0, 0).
    s: f64,
    ox: f64,
    oy: f64,
}

impl Canvas {
    fn to_px(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - self.ox) * self.s, (y - self.oy) * self.s)
    }

    fn blend(&mut self, i: usize, j: usize, c: Color, a: f32) {
        if a <= 0.0 {
            return;
        }
        let n = self.w * self.h;
        let k = j * self.w + i;
        for (ch, &value) in c.iter().enumerate() {
            let p = &mut self.px[ch * n + k];
            *p = *p * (1.0 - a) + value * a;
        }
    }

    /// Fills every pixel whose 2x2 subsamples fall `inside`, with coverage as alpha; `inside` takes
    /// pixel-space coordinates. Only the pixels in the box are visited.
    #[allow(clippy::too_many_arguments)]
    fn fill(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, c: Color, opacity: f32, inside: &dyn Fn(f64, f64) -> bool) {
        let (i0, i1) = ((x0.floor().max(0.0)) as usize, (x1.ceil().min(self.w as f64)) as usize);
        let (j0, j1) = ((y0.floor().max(0.0)) as usize, (y1.ceil().min(self.h as f64)) as usize);
        for j in j0..j1 {
            for i in i0..i1 {
                let mut hits = 0;
                for (dx, dy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                    if inside(i as f64 + dx, j as f64 + dy) {
                        hits += 1;
                    }
                }
                if hits > 0 {
                    self.blend(i, j, c, opacity * hits as f32 / 4.0);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rect(&mut self, cx: f64, cy: f64, w: f64, h: f64, r: f64, c: Color, opacity: f32) {
        let (px, py) = self.to_px(cx, cy);
        let (hw, hh, rr) = (w * self.s / 2.0, h * self.s / 2.0, (r * self.s).min(w * self.s / 2.0).min(h * self.s / 2.0));
        self.fill(px - hw, py - hh, px + hw, py + hh, c, opacity, &|x, y| {
            let (dx, dy) = ((x - px).abs() - (hw - rr), (y - py).abs() - (hh - rr));
            if dx <= 0.0 && dy <= 0.0 {
                return true;
            }
            if dx > 0.0 && dy > 0.0 {
                return dx * dx + dy * dy <= rr * rr;
            }
            dx <= rr && dy <= rr
        });
    }

    fn circle(&mut self, cx: f64, cy: f64, r: f64, c: Color, opacity: f32) {
        let (px, py) = self.to_px(cx, cy);
        let pr = r * self.s;
        self.fill(px - pr, py - pr, px + pr, py + pr, c, opacity, &|x, y| (x - px).powi(2) + (y - py).powi(2) <= pr * pr);
    }

    #[allow(clippy::too_many_arguments)]
    fn ring(&mut self, cx: f64, cy: f64, r: f64, width: f64, c: Color, opacity: f32) {
        let (px, py) = self.to_px(cx, cy);
        let (ro, ri) = ((r + width / 2.0) * self.s, ((r - width / 2.0) * self.s).max(0.0));
        self.fill(px - ro, py - ro, px + ro, py + ro, c, opacity, &|x, y| {
            let d2 = (x - px).powi(2) + (y - py).powi(2);
            d2 <= ro * ro && d2 >= ri * ri
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn rect_outline(&mut self, cx: f64, cy: f64, w: f64, h: f64, r: f64, width: f64, c: Color, opacity: f32) {
        let (px, py) = self.to_px(cx, cy);
        let (hw, hh) = (w * self.s / 2.0, h * self.s / 2.0);
        let (rr, half) = ((r * self.s).min(hw).min(hh), width * self.s / 2.0);
        let dist = move |x: f64, y: f64| -> f64 {
            let (dx, dy) = ((x - px).abs() - (hw - rr), (y - py).abs() - (hh - rr));
            let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
            outside + dx.max(dy).min(0.0) - rr
        };
        self.fill(px - hw - half, py - hh - half, px + hw + half, py + hh + half, c, opacity, &|x, y| dist(x, y).abs() <= half);
    }

    #[allow(clippy::too_many_arguments)]
    fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, half_width: f64, c: Color, opacity: f32) {
        let (ax, ay) = self.to_px(x1, y1);
        let (bx, by) = self.to_px(x2, y2);
        let hw = half_width * self.s;
        let (dx, dy) = (bx - ax, by - ay);
        let len2 = (dx * dx + dy * dy).max(1e-9);
        self.fill(ax.min(bx) - hw, ay.min(by) - hw, ax.max(bx) + hw, ay.max(by) + hw, c, opacity, &|x, y| {
            let t = (((x - ax) * dx + (y - ay) * dy) / len2).clamp(0.0, 1.0);
            let (qx, qy) = (ax + t * dx, ay + t * dy);
            (x - qx).powi(2) + (y - qy).powi(2) <= hw * hw
        });
    }

    /// Text centred at (cx, cy), `cell` spec units per font cell.
    fn text(&mut self, cx: f64, cy: f64, s: &str, cell: f64, c: Color, opacity: f32) {
        let chars: Vec<char> = s.chars().collect();
        let width = chars.len() as f64 * 4.0 * cell - cell;
        let left = cx - width / 2.0;
        let top = cy - 2.5 * cell;
        for (k, ch) in chars.iter().enumerate() {
            let Some(g) = glyph(*ch) else { continue };
            for (row, bits) in g.iter().enumerate() {
                for col in 0..3 {
                    if bits & (0b100 >> col) != 0 {
                        let x = left + (k as f64 * 4.0 + col as f64) * cell + cell / 2.0;
                        let y = top + row as f64 * cell + cell / 2.0;
                        self.rect(x, y, cell, cell, 0.0, c, opacity);
                    }
                }
            }
        }
    }

    fn shape(&mut self, sh: &Shape, dx: f64, dy: f64, scope: &StateScope) -> Result<()> {
        if let Some(w) = &sh.when {
            if !w.eval(scope)?.truthy() {
                return Ok(());
            }
        }
        let ev = |e: &Expr| -> Result<f64> { e.eval(scope)?.as_f64().context("a shape's geometry must be a number") };
        let (x, y, r) = (ev(&sh.x)? + dx, ev(&sh.y)? + dy, ev(&sh.r)?);
        match sh.kind {
            Kind::Rect => {
                let (w, h) = (ev(&sh.w)?, ev(&sh.h)?);
                if sh.fill {
                    self.rect(x, y, w, h, r, sh.color, sh.opacity);
                }
                if let Some((c, sw)) = sh.stroke {
                    self.rect_outline(x, y, w, h, r, sw, c, sh.opacity);
                }
            }
            Kind::Circle => {
                if sh.fill {
                    self.circle(x, y, r, sh.color, sh.opacity);
                }
                if let Some((c, sw)) = sh.stroke {
                    self.ring(x, y, r, sw, c, sh.opacity);
                }
            }
            Kind::Line => self.line(x, y, ev(&sh.x2)? + dx, ev(&sh.y2)? + dy, r, sh.color, sh.opacity),
            Kind::Text => {
                let mut s = sh.label.clone().unwrap_or_default();
                if let Some(e) = &sh.text {
                    s.push_str(&text_of(&e.eval(scope)?));
                }
                self.text(x, y, &s, r, sh.color, sh.opacity);
            }
        }
        Ok(())
    }
}

/// A value as the font draws it: integers plainly, floats to one decimal, strings as they are.
fn text_of(v: &Value) -> String {
    match v {
        Value::Float(f) => {
            if (f - f.round()).abs() < 1e-9 {
                format!("{}", f.round() as i64)
            } else {
                format!("{f:.1}")
            }
        }
        Value::Str(s) => s.clone(),
        Value::Bool(b) => if *b { "yes" } else { "no" }.to_string(),
        other => other.py_str(),
    }
}

/// The picture of `state`: the scene's background and shapes, then every element's shapes at the
/// element's position, as RGB floats in CHW order of a `w` x `h` raster.
pub fn rasterize(spec: &Spec, state: &State, w: usize, h: usize) -> Result<Vec<f32>> {
    let scene = spec.scene.as_ref().context("the spec has no scene to draw")?;
    let (bw, bh) = (scene.x1 - scene.x0, scene.y1 - scene.y0);
    let s = (w as f64 / bw).min(h as f64 / bh);
    // Centred, so a raster whose aspect differs a little from the scene's shows a margin, not a stretch.
    let ox = scene.x0 - (w as f64 / s - bw) / 2.0;
    let oy = scene.y0 - (h as f64 / s - bh) / 2.0;
    let mut cv = Canvas { w, h, px: Vec::with_capacity(3 * w * h), s, ox, oy };
    for ch in 0..3 {
        cv.px.extend(std::iter::repeat_n(scene.background[ch], w * h));
    }
    let params: Vec<(String, Value)> = Vec::new();
    let scope = StateScope { spec, cur: state, params: &params, old: None };
    for sh in &scene.shapes {
        cv.shape(sh, 0.0, 0.0, &scope)?;
    }
    for e in &spec.render {
        if e.shapes.is_empty() {
            continue;
        }
        let (x, y) = crate::drag::element_position(spec, &e.name, state)?;
        for sh in &e.shapes {
            cv.shape(sh, x, y, &scope)?;
        }
    }
    Ok(cv.px)
}

/// Mean absolute error between two rasters of the same size, in 0..1.
pub fn mae(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 1.0;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

/// Bytes 0..255 in HWC order, for a page's `ImageData`.
pub fn to_rgb8(px: &[f32], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    let mut out = Vec::with_capacity(3 * n);
    for k in 0..n {
        for ch in 0..3 {
            out.push((px[ch * n + k].clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    out
}

/// Standard base64, for sending a raster inside JSON without another dependency.
pub fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_with_scene() -> Spec {
        Spec::parse(
            r#"
nerve: 1
name: dot
state:
  k: {type: int, range: [0, 3]}
actions:
  step: {}
transitions:
  step: { k: "(k + 1) % 4" }
scene:
  box: [0, 0, 64, 32]
  background: white
  shapes:
    - {rect: [32, 28, 60, 4], color: track}
    - {label: "hi", at: [8, 6], color: ink, size: 2}
render:
  dot: { x: "8 + 16 * k", y: 16, shapes: [{circle: [0, 0, 6], color: f4}, {text: "k", at: [0, 0], color: white, size: 1.5, when: "k > 0"}] }
initial: {k: 0}
"#,
        )
        .unwrap()
    }

    #[test]
    fn the_scene_parses_and_sizes_its_raster() {
        let sp = spec_with_scene();
        let sc = sp.scene.as_ref().unwrap();
        assert_eq!((sc.shapes.len(), sp.render[0].shapes.len()), (2, 2));
        assert_eq!(sc.raster_size(), (192, 96));
        assert_eq!(color("f4").unwrap(), [229.0 / 255.0, 72.0 / 255.0, 77.0 / 255.0]);
        assert!(color("nope").is_err());
    }

    #[test]
    fn the_dot_is_drawn_where_the_state_puts_it() {
        let sp = spec_with_scene();
        let (w, h) = (64, 32);
        let at = |k: i64| rasterize(&sp, &vec![Value::Int(k)], w, h).unwrap();
        let (a, b) = (at(0), at(2));
        let red = |px: &Vec<f32>, x: usize, y: usize| px[y * w + x] > 0.8 && px[w * h + y * w + x] < 0.4;
        // The centre carries the white digit once k > 0, so look at the disc's left edge instead.
        assert!(red(&a, 3, 16) && !red(&a, 35, 16), "k = 0 draws the dot at x = 8");
        assert!(red(&b, 35, 16) && !red(&b, 3, 16), "k = 2 draws the dot at x = 40");
        let white = |px: &Vec<f32>, x: usize, y: usize| px[y * w + x] > 0.95 && px[w * h + y * w + x] > 0.95;
        assert!(!white(&a, 8, 16) && white(&b, 40, 16), "the digit is drawn only when k > 0");
        // Background and the track show through elsewhere; the two pictures differ.
        assert!((a[0] - 1.0).abs() < 1e-6, "top-left is the white background");
        assert!(mae(&a, &b) > 0.001 && mae(&a, &a) == 0.0);
        assert_eq!(to_rgb8(&a, w, h).len(), 3 * w * h);
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
    }
}
