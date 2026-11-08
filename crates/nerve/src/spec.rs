//! Behavior specs: typed state, actions, pure transitions, invariants,
//! `ensures`, Dragology `futures`, `learn` holes, and a `render` block.
//! The `.nerve` surface syntax is YAML.

use crate::expr::Expr;
use crate::value::{py_round, py_round_digits, Value};
use anyhow::{bail, ensure, Context, Result};
use serde_yaml::{Mapping, Value as Yaml};
use std::path::Path;

/// A state: one value per spec field, in spec order.
pub type State = Vec<Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Int,
    Enum,
    Bool,
    Float,
}

#[derive(Debug, Clone)]
pub struct FieldType {
    pub name: String,
    pub kind: Kind,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub values: Vec<String>,
    pub step: Option<f64>,
    /// Gridded floats whose `lo` and `step` are YAML integers hold ints, as in Python.
    int_grid: bool,
}

impl FieldType {
    pub fn finite(&self) -> bool {
        match self.kind {
            Kind::Enum | Kind::Bool => true,
            Kind::Int => self.lo.is_some() && self.hi.is_some(),
            Kind::Float => false,
        }
    }

    /// A ranged float with no grid: inferred by descent on the energy, not by argmin.
    pub fn continuous(&self) -> bool {
        self.kind == Kind::Float && self.step.is_none() && self.lo.is_some() && self.hi.is_some()
    }

    pub fn gridded(&self) -> bool {
        self.kind == Kind::Float && self.step.is_some()
    }

    pub fn range(&self) -> (f64, f64) {
        (self.lo.unwrap_or(0.0), self.hi.unwrap_or(0.0))
    }

    pub fn domain(&self) -> Result<Vec<Value>> {
        Ok(match self.kind {
            Kind::Bool => vec![Value::Bool(false), Value::Bool(true)],
            Kind::Enum => self.values.iter().map(|v| Value::str(v)).collect(),
            Kind::Int => {
                let (Some(lo), Some(hi)) = (self.lo, self.hi) else { bail!("int field {} has no range", self.name) };
                (lo as i64..=hi as i64).map(Value::Int).collect()
            }
            Kind::Float => {
                let (Some(lo), Some(hi), Some(step)) = (self.lo, self.hi, self.step) else {
                    bail!("float field {} needs lo/hi/step to enumerate", self.name)
                };
                let n = py_round((hi - lo) / step) as i64;
                (0..=n).map(|i| self.canon(&Value::Float(lo + i as f64 * step))).collect()
            }
        })
    }

    /// Canonical value: gridded floats snap to their grid point; integral floats in int fields become ints.
    pub fn canon(&self, v: &Value) -> Value {
        match self.kind {
            Kind::Float if self.step.is_some() && self.lo.is_some() => {
                let (lo, step) = (self.lo.unwrap_or(0.0), self.step.unwrap_or(1.0));
                let Some(x) = v.as_f64() else { return v.clone() };
                let k = py_round((x - lo) / step);
                if self.int_grid {
                    Value::Int((lo + k * step) as i64)
                } else {
                    Value::Float(py_round_digits(lo + k * step, 10))
                }
            }
            Kind::Int => match v {
                Value::Float(f) if f.fract() == 0.0 => Value::Int(*f as i64),
                other => other.clone(),
            },
            _ => v.clone(),
        }
    }

    pub fn contains(&self, v: &Value) -> bool {
        let in_range = |x: f64| self.lo.is_none_or(|lo| x >= lo) && self.hi.is_none_or(|hi| x <= hi);
        match (self.kind, v) {
            (Kind::Bool, Value::Bool(_)) => true,
            (Kind::Enum, Value::Str(s)) => self.values.contains(s),
            (Kind::Int, Value::Int(i)) => in_range(*i as f64),
            (Kind::Float, Value::Bool(_) | Value::Int(_) | Value::Float(_)) => v.as_f64().is_some_and(in_range),
            _ => false,
        }
    }

    /// Normalised position of a numeric value in the field's range.
    pub fn norm(&self, v: &Value) -> f64 {
        let (lo, hi) = self.range();
        (v.as_f64().unwrap_or(lo) - lo) / (hi - lo)
    }
}

#[derive(Debug, Clone)]
pub struct Action {
    pub name: String,
    pub params: Vec<FieldType>,
}

/// A concrete action: a name plus parameter values, in the action's parameter order.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionInst {
    pub name: String,
    pub params: Vec<(String, Value)>,
}

impl ActionInst {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), params: Vec::new() }
    }

    pub fn param(&self, key: &str) -> Option<&Value> {
        self.params.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn params_json(&self) -> serde_json::Value {
        serde_json::Value::Object(self.params.iter().map(|(k, v)| (k.clone(), v.to_json())).collect())
    }
}

impl Action {
    /// Every instance over the discrete parameters; continuous ones ride along as features.
    pub fn instances(&self) -> Result<Vec<ActionInst>> {
        let discrete: Vec<&FieldType> = self.params.iter().filter(|p| !p.continuous()).collect();
        let mut out = vec![ActionInst::new(&self.name)];
        for p in discrete {
            let dom = p.domain()?;
            out = out
                .into_iter()
                .flat_map(|inst| {
                    dom.iter().map(move |v| {
                        let mut next = inst.clone();
                        next.params.push((p.name.clone(), v.clone()));
                        next
                    })
                })
                .collect();
        }
        Ok(out)
    }

    pub fn continuous_params(&self) -> Vec<&str> {
        self.params.iter().filter(|p| p.continuous()).map(|p| p.name.as_str()).collect()
    }

    pub fn param(&self, name: &str) -> Option<&FieldType> {
        self.params.iter().find(|p| p.name == name)
    }
}

#[derive(Debug, Clone)]
pub struct Transition {
    pub action: String,
    pub when: Option<Expr>,
    /// `(field index, expression)` in the order the spec writes them.
    pub updates: Vec<(usize, Expr)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FutureKind {
    Closest,
    Between,
    Vary,
}

/// A Dragology-style declaration: the states a gesture may lead to.
#[derive(Debug, Clone)]
pub struct Future {
    pub name: String,
    pub kind: FutureKind,
    pub field: usize,
    pub values: Option<Vec<Value>>,
    pub grab: Option<String>,
    pub stickiness: f64,
    pub far: Option<f64>,
    pub action: Option<String>,
    pub param: Option<String>,
    /// Dragology `closest` with a spec per branch: one `vary` for every value of `field`
    /// the oracle can reach, realised through the vary's action with the branch fixed.
    pub each: Option<Box<Future>>,
    /// Dragology `withFloating`: the grabbed element follows the pointer while the
    /// scene previews the drop. Presentation only; the minimisation is unchanged.
    pub floating: bool,
}

#[derive(Debug, Clone)]
pub struct Ensures {
    pub after: Option<String>,
    pub holds: Expr,
}

#[derive(Debug, Clone)]
pub struct HoleExample {
    pub state: State,
    pub params: Vec<(String, Value)>,
    pub next: State,
}

/// An ink-style `learn` hole: behaviour known only by examples and a safe fallback.
#[derive(Debug, Clone)]
pub struct Hole {
    pub name: String,
    pub action: String,
    pub examples: Vec<HoleExample>,
}

#[derive(Debug, Clone)]
pub struct RenderElement {
    pub name: String,
    pub x: Expr,
    pub y: Expr,
}

#[derive(Debug, Clone)]
pub struct Spec {
    pub name: String,
    pub fields: Vec<FieldType>,
    pub actions: Vec<Action>,
    pub transitions: Vec<Transition>,
    pub invariants: Vec<Expr>,
    pub ensures: Vec<Ensures>,
    pub futures: Vec<Future>,
    pub holes: Vec<Hole>,
    pub initial: State,
    pub render: Vec<RenderElement>,
}

impl Spec {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let doc: Yaml = serde_yaml::from_str(text)?;
        parse_doc(&doc)
    }

    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    pub fn action(&self, name: &str) -> Option<&Action> {
        self.actions.iter().find(|a| a.name == name)
    }

    pub fn transition(&self, action: &str) -> Option<&Transition> {
        self.transitions.iter().find(|t| t.action == action)
    }

    pub fn future(&self, name: &str) -> Option<&Future> {
        self.futures.iter().find(|f| f.name == name)
    }

    pub fn render_element(&self, name: &str) -> Option<&RenderElement> {
        self.render.iter().find(|r| r.name == name)
    }

    pub fn hole_actions(&self) -> Vec<&str> {
        self.holes.iter().map(|h| h.action.as_str()).collect()
    }

    pub fn finite(&self) -> bool {
        self.fields.iter().all(|f| f.finite() || f.gridded())
            && !self.actions.iter().any(|a| a.params.iter().any(FieldType::continuous))
    }

    pub fn continuous_fields(&self) -> Vec<usize> {
        (0..self.fields.len()).filter(|&i| self.fields[i].continuous()).collect()
    }

    pub fn discrete_fields(&self) -> Vec<usize> {
        (0..self.fields.len()).filter(|&i| !self.fields[i].continuous()).collect()
    }

    /// Every state of a finite spec, in `itertools.product` order.
    pub fn state_space(&self) -> Result<Vec<State>> {
        ensure!(self.finite(), "spec {} has continuous fields/params", self.name);
        product(&self.fields.iter().map(FieldType::domain).collect::<Result<Vec<_>>>()?)
    }

    pub fn state_space_size(&self) -> Result<usize> {
        self.fields
            .iter()
            .map(|f| if f.continuous() { Ok(1) } else { f.domain().map(|d| d.len()) })
            .product()
    }

    pub fn action_instances(&self) -> Result<Vec<ActionInst>> {
        let mut out = Vec::new();
        for a in &self.actions {
            out.extend(a.instances()?);
        }
        Ok(out)
    }

    pub fn canon_state(&self, s: &State) -> State {
        self.fields.iter().zip(s).map(|(f, v)| f.canon(v)).collect()
    }

    /// Python `repr(spec.state_key(s))`: the identity hashed for hold-out splits.
    pub fn state_key(&self, s: &State) -> String {
        tuple_repr(&self.canon_state(s).iter().map(Value::repr).collect::<Vec<_>>())
    }

    /// Python `repr(spec.action_key(a))`: name plus sorted discrete params.
    pub fn action_key(&self, a: &ActionInst) -> String {
        let cont: Vec<&str> = self.action(&a.name).map(Action::continuous_params).unwrap_or_default();
        let mut params: Vec<&(String, Value)> = a.params.iter().filter(|(k, _)| !cont.contains(&k.as_str())).collect();
        params.sort_by(|x, y| x.0.cmp(&y.0));
        let items: Vec<String> = params.iter().map(|(k, v)| tuple_repr(&[Value::str(k).repr(), v.repr()])).collect();
        tuple_repr(&[Value::str(&a.name).repr(), tuple_repr(&items)])
    }

    pub fn state_json(&self, s: &State) -> serde_json::Value {
        serde_json::Value::Object(self.fields.iter().zip(s).map(|(f, v)| (f.name.clone(), v.to_json())).collect())
    }

    pub fn state_from_json(&self, j: &serde_json::Value) -> Result<State> {
        self.fields
            .iter()
            .map(|f| Value::from_json(j.get(&f.name).with_context(|| format!("state lacks {}", f.name))?))
            .collect()
    }

    pub fn action_from_json(&self, name: &str, params: &serde_json::Value) -> Result<ActionInst> {
        let action = self.action(name).with_context(|| format!("unknown action {name:?}"))?;
        let params = action
            .params
            .iter()
            .filter_map(|p| params.get(&p.name).map(|v| Ok((p.name.clone(), Value::from_json(v)?))))
            .collect::<Result<Vec<_>>>()?;
        Ok(ActionInst { name: name.to_string(), params })
    }
}

/// Python tuple repr: `()`, `(x,)`, `(x, y)`.
fn tuple_repr(items: &[String]) -> String {
    match items {
        [] => "()".into(),
        [one] => format!("({one},)"),
        many => format!("({})", many.join(", ")),
    }
}

/// Cartesian product, last axis fastest (as `itertools.product`).
pub fn product(axes: &[Vec<Value>]) -> Result<Vec<State>> {
    let mut out: Vec<State> = vec![Vec::new()];
    for axis in axes {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                axis.iter().map(move |v| {
                    let mut s = prefix.clone();
                    s.push(v.clone());
                    s
                })
            })
            .collect();
    }
    Ok(out)
}

// ---- parsing ----------------------------------------------------------------------------------------

fn key<'a>(m: &'a Mapping, k: &str) -> Option<&'a Yaml> {
    m.get(Yaml::String(k.to_string()))
}

fn as_str(y: &Yaml) -> Result<String> {
    Ok(match y {
        Yaml::String(s) => s.clone(),
        Yaml::Number(n) => n.to_string(),
        Yaml::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        other => bail!("expected a string, got {other:?}"),
    })
}

/// Expression source from YAML: numbers and bools become their Python literal text.
fn expr_src(y: &Yaml) -> Result<String> {
    Ok(match y {
        Yaml::Number(n) if n.is_f64() => crate::value::float_repr(n.as_f64().unwrap_or(0.0)),
        other => as_str(other)?,
    })
}

fn entries(y: Option<&Yaml>) -> Result<Vec<(String, Yaml)>> {
    match y {
        None | Some(Yaml::Null) => Ok(Vec::new()),
        Some(Yaml::Mapping(m)) => m.iter().map(|(k, v)| Ok((as_str(k)?, v.clone()))).collect(),
        Some(other) => bail!("expected a mapping, got {other:?}"),
    }
}

fn num(y: &Yaml) -> Option<(f64, bool)> {
    match y {
        Yaml::Number(n) => n.as_i64().map(|i| (i as f64, true)).or_else(|| n.as_f64().map(|f| (f, false))),
        _ => None,
    }
}

fn parse_future(name: String, fm: &Mapping, field_idx: &dyn Fn(&str) -> Option<usize>) -> Result<Future> {
    let over = as_str(key(fm, "over").with_context(|| format!("future {name} needs `over`"))?)?;
    let field = field_idx(&over).with_context(|| format!("future {name} is over unknown field {over:?}"))?;
    let kind = match key(fm, "kind").map(as_str).transpose()?.as_deref().unwrap_or("closest") {
        "closest" => FutureKind::Closest,
        "between" => FutureKind::Between,
        "vary" => FutureKind::Vary,
        other => bail!("future {name}: unknown kind {other:?}"),
    };
    let values = key(fm, "values")
        .and_then(Yaml::as_sequence)
        .map(|vs| vs.iter().map(Value::from_yaml).collect::<Result<Vec<_>>>())
        .transpose()?;
    let opt_str = |k: &str| key(fm, k).filter(|v| !v.is_null()).map(as_str).transpose();
    let grab = opt_str("grab")?;
    let each = match key(fm, "each").filter(|v| !v.is_null()) {
        None => None,
        Some(Yaml::Mapping(em)) => {
            ensure!(kind == FutureKind::Closest, "future {name}: `each` nests under `closest` only");
            for k in ["stickiness", "far", "floating", "values", "each"] {
                ensure!(key(em, k).is_none(), "future {name}: set `{k}` on the outer future, not inside `each`");
            }
            let mut inner = parse_future(format!("{name}.each"), em, field_idx)?;
            ensure!(inner.kind == FutureKind::Vary, "future {name}: `each` must be a `vary` (got {:?})", inner.kind);
            ensure!(inner.field != field, "future {name}: `each` must vary a different field than {over}");
            inner.grab = inner.grab.or_else(|| grab.clone());
            Some(Box::new(inner))
        }
        Some(_) => bail!("future {name}: `each` must be a mapping"),
    };
    let floating = match key(fm, "floating") {
        None | Some(Yaml::Null) => false,
        Some(Yaml::Bool(b)) => *b,
        Some(other) => bail!("future {name}: `floating` must be true or false, got {other:?}"),
    };
    Ok(Future {
        name,
        kind,
        field,
        values,
        grab,
        stickiness: key(fm, "stickiness").and_then(num).map_or(0.0, |x| x.0),
        far: key(fm, "far").and_then(num).map(|x| x.0),
        action: opt_str("action")?,
        param: opt_str("param")?,
        each,
        floating,
    })
}

fn parse_field(name: &str, y: &Yaml) -> Result<FieldType> {
    let m = match y {
        Yaml::String(s) => {
            let mut m = Mapping::new();
            m.insert(Yaml::String("type".into()), Yaml::String(s.clone()));
            m
        }
        Yaml::Mapping(m) => m.clone(),
        other => bail!("field {name}: bad definition {other:?}"),
    };
    let kind = key(&m, "type").map(as_str).transpose()?.unwrap_or_default();
    let mut f = FieldType { name: name.into(), kind: Kind::Bool, lo: None, hi: None, values: Vec::new(), step: None, int_grid: false };
    let range = || -> Result<(Option<(f64, bool)>, Option<(f64, bool)>)> {
        match key(&m, "range") {
            Some(Yaml::Sequence(r)) if r.len() == 2 => Ok((num(&r[0]), num(&r[1]))),
            None | Some(Yaml::Null) => Ok((None, None)),
            Some(other) => bail!("field {name}: bad range {other:?}"),
        }
    };
    match kind.as_str() {
        "bool" => {}
        "enum" => {
            f.kind = Kind::Enum;
            f.values = match key(&m, "values") {
                Some(Yaml::Sequence(vals)) if !vals.is_empty() => vals.iter().map(as_str).collect::<Result<_>>()?,
                _ => bail!("enum field {name} needs values"),
            };
        }
        "int" => {
            f.kind = Kind::Int;
            let (lo, hi) = range()?;
            f.lo = lo.map(|x| x.0);
            f.hi = hi.map(|x| x.0);
        }
        "float" => {
            f.kind = Kind::Float;
            let (lo, hi) = range()?;
            let step = key(&m, "step").and_then(num);
            f.lo = lo.map(|x| x.0);
            f.hi = hi.map(|x| x.0);
            f.step = step.map(|x| x.0);
            f.int_grid = lo.is_some_and(|x| x.1) && step.is_some_and(|x| x.1);
        }
        other => bail!("field {name}: unknown type {other:?}"),
    }
    Ok(f)
}

fn yaml_state(fields: &[FieldType], y: &Yaml, what: &str) -> Result<State> {
    let Yaml::Mapping(m) = y else { bail!("{what} must be a mapping") };
    fields
        .iter()
        .map(|f| {
            let v = Value::from_yaml(key(m, &f.name).with_context(|| format!("{what}.{} is missing", f.name))?)?;
            ensure!(f.contains(&v), "{what}.{}={} is not in type", f.name, v.repr());
            Ok(v)
        })
        .collect()
}

fn parse_doc(doc: &Yaml) -> Result<Spec> {
    let Yaml::Mapping(doc) = doc else { bail!("spec must be a mapping") };
    ensure!(key(doc, "nerve").and_then(Yaml::as_i64) == Some(1), "spec must declare `nerve: 1`");
    let name = key(doc, "name").map(as_str).transpose()?.unwrap_or_else(|| "unnamed".into());
    let fields = entries(key(doc, "state"))?.iter().map(|(k, v)| parse_field(k, v)).collect::<Result<Vec<_>>>()?;
    ensure!(!fields.is_empty(), "spec has no state");
    let field_idx = |k: &str| fields.iter().position(|f| f.name == k);

    let mut actions = Vec::new();
    for (aname, ad) in entries(key(doc, "actions"))? {
        let params = match &ad {
            Yaml::Mapping(m) => entries(key(m, "params"))?.iter().map(|(k, v)| parse_field(k, v)).collect::<Result<_>>()?,
            _ => Vec::new(),
        };
        actions.push(Action { name: aname, params });
    }
    let known_action = |a: &str| actions.iter().any(|x| x.name == a);

    let mut transitions: Vec<Transition> = Vec::new();
    for (aname, td) in entries(key(doc, "transitions"))? {
        ensure!(known_action(&aname), "transition for unknown action {aname:?}");
        let mut when = None;
        let mut updates = Vec::new();
        for (k, v) in entries(Some(&td))? {
            if k == "when" {
                when = Some(Expr::compile(&expr_src(&v)?)?);
                continue;
            }
            let idx = field_idx(&k).with_context(|| format!("transition {aname} updates unknown field {k:?}"))?;
            updates.push((idx, Expr::compile(&expr_src(&v)?)?));
        }
        transitions.push(Transition { action: aname, when, updates });
    }

    let mut holes = Vec::new();
    for hd in key(doc, "learn").and_then(Yaml::as_sequence).cloned().unwrap_or_default() {
        let Yaml::Mapping(hm) = &hd else { bail!("learn entries must be mappings") };
        let hname = as_str(key(hm, "name").context("learn hole needs a name")?)?;
        let haction = as_str(key(hm, "action").context("learn hole needs an action")?)?;
        let action = actions.iter().find(|a| a.name == haction);
        let action = action.with_context(|| format!("learn hole {hname} refers to unknown action {haction:?}"))?;
        let mut examples = Vec::new();
        for (i, ex) in key(hm, "examples").and_then(Yaml::as_sequence).cloned().unwrap_or_default().iter().enumerate() {
            let Yaml::Mapping(em) = ex else { bail!("learn hole {hname} example {i} must be a mapping") };
            let (Some(st), Some(nx)) = (key(em, "state"), key(em, "next")) else {
                bail!("learn hole {hname} example {i} needs `state` and `next`")
            };
            let mut params = Vec::new();
            if !action.params.is_empty() {
                let pm = key(em, "params").and_then(Yaml::as_mapping);
                for p in &action.params {
                    let v = pm.and_then(|m| key(m, &p.name)).with_context(|| {
                        format!("learn hole {hname} example {i}: action {haction:?} takes params; give them as `params: {{...}}`")
                    })?;
                    let v = Value::from_yaml(v)?;
                    ensure!(p.contains(&v), "learn hole {hname} example {i}: param {}={} is not in the action's type", p.name, v.repr());
                    params.push((p.name.clone(), v));
                }
            }
            examples.push(HoleExample {
                state: yaml_state(&fields, st, &format!("learn hole {hname} example {i}: state"))?,
                params,
                next: yaml_state(&fields, nx, &format!("learn hole {hname} example {i}: next"))?,
            });
        }
        if !transitions.iter().any(|t| t.action == haction) {
            // The fallback is the spec-visible transition; the hole is learned on top.
            let updates = entries(key(hm, "fallback"))?
                .iter()
                .map(|(k, v)| Ok((field_idx(k).with_context(|| format!("fallback updates unknown field {k:?}"))?, Expr::compile(&expr_src(v)?)?)))
                .collect::<Result<_>>()?;
            transitions.push(Transition { action: haction.clone(), when: None, updates });
        }
        holes.push(Hole { name: hname, action: haction, examples });
    }
    for a in &actions {
        if !transitions.iter().any(|t| t.action == a.name) {
            transitions.push(Transition { action: a.name.clone(), when: None, updates: Vec::new() });
        }
    }

    let ensures = key(doc, "ensures")
        .and_then(Yaml::as_sequence)
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|e| {
            let Yaml::Mapping(em) = e else { bail!("ensures entries must be mappings") };
            Ok(Ensures {
                after: key(em, "after").filter(|v| !v.is_null()).map(as_str).transpose()?,
                holds: Expr::compile(&expr_src(key(em, "holds").context("ensures needs `holds`")?)?)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut futures = Vec::new();
    for (fname, fd) in entries(key(doc, "futures"))? {
        let Yaml::Mapping(fm) = &fd else { bail!("future {fname} must be a mapping") };
        futures.push(parse_future(fname, fm, &field_idx)?);
    }

    let mut render = Vec::new();
    for (ename, ed) in entries(key(doc, "render"))? {
        let Yaml::Mapping(em) = &ed else { bail!("render.{ename} needs x and y expressions") };
        let (Some(x), Some(y)) = (key(em, "x"), key(em, "y")) else { bail!("render.{ename} needs x and y expressions") };
        render.push(RenderElement { name: ename, x: Expr::compile(&expr_src(x)?)?, y: Expr::compile(&expr_src(y)?)? });
    }

    let init_map = key(doc, "initial").and_then(Yaml::as_mapping).cloned().unwrap_or_default();
    let initial = fields
        .iter()
        .map(|f| {
            let v = match key(&init_map, &f.name) {
                Some(v) => Value::from_yaml(v)?,
                None if f.finite() || f.step.is_some() => f.domain()?.remove(0),
                None => f.lo.map_or(Value::Int(0), Value::Float),
            };
            ensure!(f.contains(&v), "initial value for {} is out of type: {}", f.name, v.repr());
            Ok(v)
        })
        .collect::<Result<State>>()?;

    let invariants = key(doc, "invariants")
        .and_then(Yaml::as_sequence)
        .cloned()
        .unwrap_or_default()
        .iter()
        .map(|i| Expr::compile(&expr_src(i)?))
        .collect::<Result<Vec<_>>>()?;

    Ok(Spec { name, fields, actions, transitions, invariants, ensures, futures, holes, initial, render })
}

#[cfg(test)]
mod tests {
    use super::*;

    const COUNTER: &str = include_str!("../../../specs/counter.nerve");
    const SLIDER: &str = include_str!("../../../specs/slider.nerve");
    const NEGOTIATE: &str = include_str!("../../../specs/negotiate.nerve");

    #[test]
    fn parses_the_shipped_specs() {
        let c = Spec::parse(COUNTER).unwrap();
        assert_eq!(c.fields.len(), 2);
        assert_eq!(c.state_space().unwrap().len(), 20);
        assert_eq!(c.action_instances().unwrap().len(), 14);
        assert_eq!(c.holes[0].examples.len(), 5);
        assert_eq!(c.future("drag_dial").unwrap().stickiness, 4.0);
        let n = Spec::parse(NEGOTIATE).unwrap();
        assert_eq!(n.state_space().unwrap().len(), 216);
        assert_eq!(n.action_instances().unwrap().len(), 11);
    }

    #[test]
    fn keys_match_python_repr() {
        let n = Spec::parse(NEGOTIATE).unwrap();
        assert_eq!(n.state_key(&n.initial), "('a', 0, 0, 'open', 0)");
        let acts = n.action_instances().unwrap();
        assert_eq!(n.action_key(&acts[1]), "('propose', (('b', 0), ('h', 1)))");
        assert_eq!(n.action_key(&acts[9]), "('accept', ())");
        let c = Spec::parse(COUNTER).unwrap();
        let set3 = c.action_instances().unwrap().into_iter().find(|a| a.name == "set" && a.param("to") == Some(&Value::Int(3)));
        assert_eq!(c.action_key(&set3.unwrap()), "('set', (('to', 3),))");
    }

    #[test]
    fn each_nests_a_vary_under_closest() {
        let tracks = include_str!("../../../specs/tracks.nerve");
        let t = Spec::parse(tracks).unwrap();
        let f = t.future("drag_block").unwrap();
        assert!(f.floating);
        let inner = f.each.as_deref().unwrap();
        assert_eq!((inner.kind.clone(), t.fields[inner.field].name.as_str()), (FutureKind::Vary, "pos"));
        assert_eq!((inner.action.as_deref(), inner.param.as_deref()), (Some("move"), Some("to")));
        assert!(!Spec::parse(SLIDER).unwrap().future("drag_handle").unwrap().floating);
        let not_closest = tracks.replace("    kind: closest\n", "    kind: between\n");
        assert!(Spec::parse(&not_closest).unwrap_err().to_string().contains("each"));
        let not_vary = tracks.replace("each: { kind: vary,", "each: { kind: closest,");
        assert!(Spec::parse(&not_vary).unwrap_err().to_string().contains("vary"));
        let inner_far = tracks.replace("each: { kind: vary,", "each: { far: 10, kind: vary,");
        assert!(Spec::parse(&inner_far).unwrap_err().to_string().contains("outer future"));
    }

    #[test]
    fn integer_grids_hold_ints() {
        let s = Spec::parse(SLIDER).unwrap();
        let dom = s.fields[0].domain().unwrap();
        assert_eq!(dom.len(), 25);
        assert!(matches!(dom[3], Value::Int(30)));
        assert!(matches!(s.fields[0].canon(&Value::Float(137.2)), Value::Int(140)));
        assert_eq!(s.state_key(&s.initial), "(0, False)");
    }
}
