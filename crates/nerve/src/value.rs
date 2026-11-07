//! Dynamically typed values with Python's semantics, so spec expressions and
//! hold-out hashes behave exactly like the Python reference implementation.

use anyhow::{bail, Result};
use std::cmp::Ordering;
use std::fmt;

#[derive(Debug, Clone)]
pub enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Value>),
}

impl Value {
    pub fn str(s: &str) -> Self {
        Self::Str(s.to_string())
    }

    pub fn truthy(&self) -> bool {
        match self {
            Self::None => false,
            Self::Bool(b) => *b,
            Self::Int(i) => *i != 0,
            Self::Float(f) => *f != 0.0,
            Self::Str(s) => !s.is_empty(),
            Self::List(l) => !l.is_empty(),
        }
    }

    /// Numeric view: bools are ints, as in Python.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Bool(b) => Some(f64::from(u8::from(*b))),
            Self::Int(i) => Some(*i as f64),
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            Self::Bool(b) => Some(i64::from(*b)),
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Self::Bool(_) | Self::Int(_) | Self::Float(_))
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::None => "NoneType",
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Str(_) => "str",
            Self::List(_) => "list",
        }
    }

    /// Python `str(v)`, used when an enum update yields a non-string.
    pub fn py_str(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            other => other.repr(),
        }
    }

    /// Python `repr(v)`; hold-out hashes are computed over these strings.
    pub fn repr(&self) -> String {
        match self {
            Self::None => "None".into(),
            Self::Bool(true) => "True".into(),
            Self::Bool(false) => "False".into(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => float_repr(*f),
            Self::Str(s) => str_repr(s),
            Self::List(l) => format!("[{}]", l.iter().map(Value::repr).collect::<Vec<_>>().join(", ")),
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::Value as J;
        match self {
            Self::None => J::Null,
            Self::Bool(b) => J::Bool(*b),
            Self::Int(i) => J::from(*i),
            Self::Float(f) => serde_json::Number::from_f64(*f).map_or(J::Null, J::Number),
            Self::Str(s) => J::String(s.clone()),
            Self::List(l) => J::Array(l.iter().map(Value::to_json).collect()),
        }
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        use serde_json::Value as J;
        Ok(match v {
            J::Null => Self::None,
            J::Bool(b) => Self::Bool(*b),
            J::Number(n) => match n.as_i64() {
                Some(i) => Self::Int(i),
                None => Self::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            J::String(s) => Self::Str(s.clone()),
            J::Array(a) => Self::List(a.iter().map(Self::from_json).collect::<Result<_>>()?),
            J::Object(_) => bail!("objects are not values"),
        })
    }

    pub fn from_yaml(v: &serde_yaml::Value) -> Result<Self> {
        use serde_yaml::Value as Y;
        Ok(match v {
            Y::Null => Self::None,
            Y::Bool(b) => Self::Bool(*b),
            Y::Number(n) => match n.as_i64() {
                Some(i) => Self::Int(i),
                None => Self::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            Y::String(s) => Self::Str(s.clone()),
            Y::Sequence(s) => Self::List(s.iter().map(Self::from_yaml).collect::<Result<_>>()?),
            other => bail!("unsupported YAML value {other:?}"),
        })
    }

    /// Python ordering for `<`, `<=`, ... (numbers with numbers, strings with strings).
    pub fn py_cmp(&self, other: &Self) -> Result<Ordering> {
        match (self, other) {
            (Self::Str(a), Self::Str(b)) => Ok(a.cmp(b)),
            (Self::List(a), Self::List(b)) => {
                for (x, y) in a.iter().zip(b) {
                    let o = x.py_cmp(y)?;
                    if o != Ordering::Equal {
                        return Ok(o);
                    }
                }
                Ok(a.len().cmp(&b.len()))
            }
            (a, b) => match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => x.partial_cmp(&y).ok_or_else(|| anyhow::anyhow!("NaN comparison")),
                _ => bail!("'<' not supported between {} and {}", a.type_name(), b.type_name()),
            },
        }
    }
}

/// Python `==`: numeric types compare by value; other types must match.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None) => true,
            (Self::Str(a), Self::Str(b)) => a == b,
            (Self::List(a), Self::List(b)) => a == b,
            (a, b) => match (a.as_f64(), b.as_f64()) {
                (Some(x), Some(y)) => x == y,
                _ => false,
            },
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.py_str())
    }
}

/// Python's shortest round-trip float repr (`0.6`, `10.0`, `1e-05`).
pub fn float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let abs = f.abs();
    if abs != 0.0 && !(1e-4..1e16).contains(&abs) {
        // Python switches to exponent notation outside [1e-4, 1e16).
        let s = format!("{f:e}");
        let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
        let exp: i32 = exp.parse().unwrap_or(0);
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{mant}e{sign}{:02}", exp.abs());
    }
    let s = format!("{f}");
    if s.contains('.') { s } else { format!("{s}.0") }
}

/// Python `repr` of a str: single quotes unless the text contains one.
pub fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python 3 `round(x)`: ties to even.
pub fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// Python `round(x, ndigits)`, close enough for canonicalising grid points.
pub fn py_round_digits(x: f64, ndigits: i32) -> f64 {
    let m = 10f64.powi(ndigits);
    let r = (x * m).round_ties_even() / m;
    if r.is_finite() { r } else { x }
}

// ---- arithmetic -----------------------------------------------------------------------------------

fn num_op(a: &Value, b: &Value, op: &str) -> Result<(f64, f64, bool)> {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => Ok((x, y, a.as_int().is_some() && b.as_int().is_some())),
        _ => bail!("unsupported operand types for {op}: {} and {}", a.type_name(), b.type_name()),
    }
}

pub fn add(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Str(x), Value::Str(y)) => return Ok(Value::Str(format!("{x}{y}"))),
        (Value::List(x), Value::List(y)) => return Ok(Value::List(x.iter().chain(y).cloned().collect())),
        _ => {}
    }
    let (x, y, ints) = num_op(a, b, "+")?;
    Ok(if ints { Value::Int(a.as_int().unwrap_or(0) + b.as_int().unwrap_or(0)) } else { Value::Float(x + y) })
}

pub fn sub(a: &Value, b: &Value) -> Result<Value> {
    let (x, y, ints) = num_op(a, b, "-")?;
    Ok(if ints { Value::Int(a.as_int().unwrap_or(0) - b.as_int().unwrap_or(0)) } else { Value::Float(x - y) })
}

pub fn mul(a: &Value, b: &Value) -> Result<Value> {
    match (a, b) {
        (Value::Str(s), n) | (n, Value::Str(s)) if n.as_int().is_some() => {
            return Ok(Value::Str(s.repeat(n.as_int().unwrap_or(0).max(0) as usize)));
        }
        _ => {}
    }
    let (x, y, ints) = num_op(a, b, "*")?;
    Ok(if ints { Value::Int(a.as_int().unwrap_or(0) * b.as_int().unwrap_or(0)) } else { Value::Float(x * y) })
}

pub fn div(a: &Value, b: &Value) -> Result<Value> {
    let (x, y, _) = num_op(a, b, "/")?;
    if y == 0.0 {
        bail!("division by zero");
    }
    Ok(Value::Float(x / y))
}

pub fn floordiv(a: &Value, b: &Value) -> Result<Value> {
    let (x, y, ints) = num_op(a, b, "//")?;
    if y == 0.0 {
        bail!("integer division or modulo by zero");
    }
    if ints {
        let (p, q) = (a.as_int().unwrap_or(0), b.as_int().unwrap_or(1));
        return Ok(Value::Int(p.div_euclid(q) - i64::from(q < 0 && p.rem_euclid(q) != 0)));
    }
    Ok(Value::Float((x / y).floor()))
}

pub fn modulo(a: &Value, b: &Value) -> Result<Value> {
    let (x, y, ints) = num_op(a, b, "%")?;
    if y == 0.0 {
        bail!("integer division or modulo by zero");
    }
    if ints {
        let (p, q) = (a.as_int().unwrap_or(0), b.as_int().unwrap_or(1));
        // Python's result takes the sign of the divisor.
        let r = p % q;
        return Ok(Value::Int(if r != 0 && (r < 0) != (q < 0) { r + q } else { r }));
    }
    let r = x % y;
    Ok(Value::Float(if r != 0.0 && (r < 0.0) != (y < 0.0) { r + y } else { r }))
}

pub fn pow(a: &Value, b: &Value) -> Result<Value> {
    let (x, y, ints) = num_op(a, b, "**")?;
    if ints {
        let e = b.as_int().unwrap_or(0);
        if e >= 0 {
            let base = a.as_int().unwrap_or(0);
            return base.checked_pow(e as u32).map(Value::Int).ok_or_else(|| anyhow::anyhow!("integer overflow"));
        }
    }
    Ok(Value::Float(x.powf(y)))
}

pub fn neg(a: &Value) -> Result<Value> {
    match a {
        Value::Bool(_) | Value::Int(_) => Ok(Value::Int(-a.as_int().unwrap_or(0))),
        Value::Float(f) => Ok(Value::Float(-f)),
        other => bail!("bad operand type for unary -: {}", other.type_name()),
    }
}

pub fn contains(container: &Value, item: &Value) -> Result<bool> {
    match container {
        Value::List(l) => Ok(l.contains(item)),
        Value::Str(s) => match item {
            Value::Str(sub) => Ok(s.contains(sub.as_str())),
            other => bail!("'in <string>' requires string as left operand, not {}", other.type_name()),
        },
        other => bail!("argument of type {} is not iterable", other.type_name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repr_matches_python() {
        assert_eq!(Value::Float(0.6).repr(), "0.6");
        assert_eq!(Value::Float(10.0).repr(), "10.0");
        assert_eq!(Value::Float(1e-5).repr(), "1e-05");
        assert_eq!(Value::Float(1e16).repr(), "1e+16");
        assert_eq!(Value::str("on").repr(), "'on'");
        assert_eq!(Value::str("it's").repr(), "\"it's\"");
        assert_eq!(Value::Bool(true).repr(), "True");
    }

    #[test]
    fn arithmetic_matches_python() {
        let i = Value::Int;
        assert_eq!(floordiv(&i(-7), &i(2)).unwrap(), i(-4));
        assert_eq!(modulo(&i(-7), &i(2)).unwrap(), i(1));
        assert_eq!(modulo(&i(7), &i(-2)).unwrap(), i(-1));
        assert_eq!(div(&i(1), &i(2)).unwrap(), Value::Float(0.5));
        assert_eq!(add(&Value::Bool(true), &i(1)).unwrap(), i(2));
        assert_eq!(Value::Int(1), Value::Float(1.0));
        assert_ne!(Value::Int(1), Value::str("1"));
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(3.5), 4.0);
        assert_eq!(py_round_digits(0.6000000000000001, 10), 0.6);
    }
}
