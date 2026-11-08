//! The spec expression language: a safe subset of Python expressions.
//!
//! Supported, as in the Python reference: literals, names, arithmetic,
//! comparisons (chained, `in`, `not in`), `and`/`or`/`not`, conditional
//! expressions, `old.<field>`, list/tuple literals, subscripts and slices,
//! and the builtins `min max abs len int bool clamp`.

use crate::value::{self, Value};
use anyhow::{bail, ensure, Context, Result};

const BUILTINS: [&str; 7] = ["min", "max", "abs", "len", "int", "bool", "clamp"];

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(Value),
    Str(String),
    Name(String),
    Op(&'static str),
}

const OPS: [&str; 22] = [
    "**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "<", ">", "(", ")", "[", "]", ",", ".", ":", "{",
    "}",
];

fn tokenize(src: &str) -> Result<Vec<Tok>> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '.' || chars[i] == '_') {
                // Allow exponent signs: 1e-5
                if (chars[i] == 'e' || chars[i] == 'E') && matches!(chars.get(i + 1), Some('-' | '+')) {
                    i += 1;
                }
                i += 1;
            }
            let text: String = chars[start..i].iter().filter(|c| **c != '_').collect();
            let num = if text.contains(['.', 'e', 'E']) {
                Value::Float(text.parse().with_context(|| format!("bad number {text:?}"))?)
            } else {
                Value::Int(text.parse().with_context(|| format!("bad number {text:?}"))?)
            };
            out.push(Tok::Num(num));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            out.push(Tok::Name(chars[start..i].iter().collect()));
        } else if c == '\'' || c == '"' {
            let mut s = String::new();
            i += 1;
            loop {
                let ch = *chars.get(i).with_context(|| format!("unterminated string in {src:?}"))?;
                i += 1;
                match ch {
                    _ if ch == c => break,
                    '\\' => {
                        let esc = *chars.get(i).context("dangling escape")?;
                        i += 1;
                        s.push(match esc {
                            'n' => '\n',
                            't' => '\t',
                            other => other,
                        });
                    }
                    other => s.push(other),
                }
            }
            out.push(Tok::Str(s));
        } else {
            let rest: String = chars[i..chars.len().min(i + 2)].iter().collect();
            let op = OPS
                .iter()
                .find(|op| rest.starts_with(*op))
                .with_context(|| format!("unexpected character {c:?} in {src:?}"))?;
            ensure!(!matches!(*op, "{" | "}"), "disallowed syntax Dict/Set in {src:?}");
            out.push(Tok::Op(op));
            i += op.len();
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
}

#[derive(Debug, Clone)]
enum Node {
    Const(Value),
    Name(String),
    Old(String),
    List(Vec<Node>),
    Neg(Box<Node>),
    Pos(Box<Node>),
    Not(Box<Node>),
    Bin(BinOp, Box<Node>, Box<Node>),
    And(Vec<Node>),
    Or(Vec<Node>),
    Cmp(Box<Node>, Vec<(CmpOp, Node)>),
    IfExp { test: Box<Node>, body: Box<Node>, orelse: Box<Node> },
    Call(String, Vec<Node>),
    Index(Box<Node>, Box<Node>),
    Slice(Box<Node>, [Option<Box<Node>>; 3]),
}

/// A compiled expression, safe to evaluate repeatedly.
#[derive(Debug, Clone)]
pub struct Expr {
    src: String,
    root: Node,
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
    src: &'a str,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn is_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(o)) if *o == op)
    }

    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Some(Tok::Name(n)) if n == kw)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        let hit = self.is_op(op);
        self.pos += usize::from(hit);
        hit
    }

    fn eat_kw(&mut self, kw: &str) -> bool {
        let hit = self.is_kw(kw);
        self.pos += usize::from(hit);
        hit
    }

    fn expect_op(&mut self, op: &str) -> Result<()> {
        ensure!(self.eat_op(op), "expected {op:?} in {:?}", self.src);
        Ok(())
    }

    fn ternary(&mut self) -> Result<Node> {
        let body = self.or()?;
        if self.eat_kw("if") {
            let test = self.or()?;
            ensure!(self.eat_kw("else"), "expected 'else' in {:?}", self.src);
            let orelse = self.ternary()?;
            return Ok(Node::IfExp { test: Box::new(test), body: Box::new(body), orelse: Box::new(orelse) });
        }
        Ok(body)
    }

    fn or(&mut self) -> Result<Node> {
        let mut parts = vec![self.and()?];
        while self.eat_kw("or") {
            parts.push(self.and()?);
        }
        Ok(if parts.len() == 1 { parts.remove(0) } else { Node::Or(parts) })
    }

    fn and(&mut self) -> Result<Node> {
        let mut parts = vec![self.not()?];
        while self.eat_kw("and") {
            parts.push(self.not()?);
        }
        Ok(if parts.len() == 1 { parts.remove(0) } else { Node::And(parts) })
    }

    fn not(&mut self) -> Result<Node> {
        if self.eat_kw("not") {
            return Ok(Node::Not(Box::new(self.not()?)));
        }
        self.comparison()
    }

    fn cmp_op(&mut self) -> Option<CmpOp> {
        let op = match self.peek()? {
            Tok::Op("==") => CmpOp::Eq,
            Tok::Op("!=") => CmpOp::Ne,
            Tok::Op("<") => CmpOp::Lt,
            Tok::Op("<=") => CmpOp::Le,
            Tok::Op(">") => CmpOp::Gt,
            Tok::Op(">=") => CmpOp::Ge,
            Tok::Name(n) if n == "in" => CmpOp::In,
            Tok::Name(n) if n == "not" && matches!(self.toks.get(self.pos + 1), Some(Tok::Name(m)) if m == "in") => {
                self.pos += 1;
                CmpOp::NotIn
            }
            _ => return None,
        };
        self.pos += 1;
        Some(op)
    }

    fn comparison(&mut self) -> Result<Node> {
        let left = self.arith()?;
        let mut rest = Vec::new();
        while let Some(op) = self.cmp_op() {
            rest.push((op, self.arith()?));
        }
        Ok(if rest.is_empty() { left } else { Node::Cmp(Box::new(left), rest) })
    }

    fn arith(&mut self) -> Result<Node> {
        let mut left = self.term()?;
        loop {
            let op = if self.eat_op("+") {
                BinOp::Add
            } else if self.eat_op("-") {
                BinOp::Sub
            } else {
                return Ok(left);
            };
            left = Node::Bin(op, Box::new(left), Box::new(self.term()?));
        }
    }

    fn term(&mut self) -> Result<Node> {
        let mut left = self.factor()?;
        loop {
            let op = if self.eat_op("*") {
                BinOp::Mul
            } else if self.eat_op("//") {
                BinOp::FloorDiv
            } else if self.eat_op("/") {
                BinOp::Div
            } else if self.eat_op("%") {
                BinOp::Mod
            } else {
                return Ok(left);
            };
            left = Node::Bin(op, Box::new(left), Box::new(self.factor()?));
        }
    }

    fn factor(&mut self) -> Result<Node> {
        if self.eat_op("-") {
            return Ok(Node::Neg(Box::new(self.factor()?)));
        }
        if self.eat_op("+") {
            return Ok(Node::Pos(Box::new(self.factor()?)));
        }
        self.power()
    }

    fn power(&mut self) -> Result<Node> {
        let base = self.postfix()?;
        if self.eat_op("**") {
            // Right-associative and binds tighter than unary minus on its left.
            return Ok(Node::Bin(BinOp::Pow, Box::new(base), Box::new(self.factor()?)));
        }
        Ok(base)
    }

    fn postfix(&mut self) -> Result<Node> {
        let mut node = self.atom()?;
        loop {
            if self.eat_op("(") {
                let Node::Name(name) = &node else { bail!("only builtins {BUILTINS:?} may be called in {:?}", self.src) };
                ensure!(BUILTINS.contains(&name.as_str()), "only builtins {BUILTINS:?} may be called in {:?}", self.src);
                let mut args = Vec::new();
                if !self.eat_op(")") {
                    loop {
                        args.push(self.ternary()?);
                        if self.eat_op(")") {
                            break;
                        }
                        self.expect_op(",")?;
                    }
                }
                node = Node::Call(name.clone(), args);
            } else if self.eat_op("[") {
                node = self.subscript(node)?;
            } else if self.eat_op(".") {
                let Node::Name(base) = &node else { bail!("attribute access only on 'old' in {:?}", self.src) };
                ensure!(base == "old", "attribute access only on 'old' in {:?}", self.src);
                let Some(Tok::Name(attr)) = self.peek().cloned() else { bail!("expected attribute in {:?}", self.src) };
                self.pos += 1;
                node = Node::Old(attr);
            } else {
                return Ok(node);
            }
        }
    }

    fn subscript(&mut self, target: Node) -> Result<Node> {
        let mut parts: [Option<Box<Node>>; 3] = [None, None, None];
        let mut slot = 0;
        let mut is_slice = false;
        loop {
            if self.eat_op("]") {
                break;
            }
            if self.eat_op(":") {
                is_slice = true;
                slot += 1;
                ensure!(slot < 3, "bad slice in {:?}", self.src);
                continue;
            }
            parts[slot] = Some(Box::new(self.ternary()?));
        }
        if is_slice {
            return Ok(Node::Slice(Box::new(target), parts));
        }
        let idx = parts[0].take().with_context(|| format!("empty subscript in {:?}", self.src))?;
        Ok(Node::Index(Box::new(target), idx))
    }

    fn atom(&mut self) -> Result<Node> {
        let tok = self.peek().cloned().with_context(|| format!("unexpected end of {:?}", self.src))?;
        self.pos += 1;
        match tok {
            Tok::Num(v) => Ok(Node::Const(v)),
            Tok::Str(s) => Ok(Node::Const(Value::Str(s))),
            Tok::Name(n) => Ok(match n.as_str() {
                "True" => Node::Const(Value::Bool(true)),
                "False" => Node::Const(Value::Bool(false)),
                "None" => Node::Const(Value::None),
                "lambda" | "import" | "for" | "yield" | "await" => bail!("disallowed syntax {n} in {:?}", self.src),
                _ => Node::Name(n),
            }),
            Tok::Op("(") => {
                if self.eat_op(")") {
                    return Ok(Node::List(Vec::new()));
                }
                let first = self.ternary()?;
                if self.eat_op(")") {
                    return Ok(first);
                }
                let mut items = vec![first];
                while self.eat_op(",") {
                    if self.is_op(")") {
                        break;
                    }
                    items.push(self.ternary()?);
                }
                self.expect_op(")")?;
                Ok(Node::List(items))
            }
            Tok::Op("[") => {
                let mut items = Vec::new();
                while !self.eat_op("]") {
                    items.push(self.ternary()?);
                    if !self.eat_op(",") {
                        self.expect_op("]")?;
                        break;
                    }
                }
                Ok(Node::List(items))
            }
            other => bail!("unexpected {other:?} in {:?}", self.src),
        }
    }
}

/// Variable lookup for evaluation: the current scope and the previous state (`old`).
pub trait Scope {
    fn get(&self, name: &str) -> Option<Value>;
    fn old(&self, name: &str) -> Option<Value> {
        self.get(name)
    }
}

impl Expr {
    pub fn compile(src: &str) -> Result<Self> {
        let toks = tokenize(src)?;
        let mut p = Parser { toks: &toks, pos: 0, src };
        let root = p.ternary()?;
        ensure!(p.pos == toks.len(), "bad expression {src:?}: trailing tokens");
        Ok(Self { src: src.to_string(), root })
    }

    pub fn source(&self) -> &str {
        &self.src
    }

    pub fn eval(&self, scope: &dyn Scope) -> Result<Value> {
        eval(&self.root, scope).with_context(|| format!("evaluating {:?}", self.src))
    }
}

fn eval(node: &Node, scope: &dyn Scope) -> Result<Value> {
    Ok(match node {
        Node::Const(v) => v.clone(),
        Node::Name(n) => scope.get(n).with_context(|| format!("unknown name {n:?}"))?,
        Node::Old(n) => scope.old(n).with_context(|| format!("unknown field {n:?}"))?,
        Node::List(items) => Value::List(items.iter().map(|i| eval(i, scope)).collect::<Result<_>>()?),
        Node::Neg(x) => value::neg(&eval(x, scope)?)?,
        Node::Pos(x) => {
            let v = eval(x, scope)?;
            ensure!(v.is_number(), "bad operand type for unary +: {}", v.type_name());
            v
        }
        Node::Not(x) => Value::Bool(!eval(x, scope)?.truthy()),
        Node::Bin(op, a, b) => {
            let (a, b) = (eval(a, scope)?, eval(b, scope)?);
            match op {
                BinOp::Add => value::add(&a, &b)?,
                BinOp::Sub => value::sub(&a, &b)?,
                BinOp::Mul => value::mul(&a, &b)?,
                BinOp::Div => value::div(&a, &b)?,
                BinOp::FloorDiv => value::floordiv(&a, &b)?,
                BinOp::Mod => value::modulo(&a, &b)?,
                BinOp::Pow => value::pow(&a, &b)?,
            }
        }
        // Like the reference evaluator, `and`/`or` return booleans, not operands.
        Node::And(parts) => {
            for p in parts {
                if !eval(p, scope)?.truthy() {
                    return Ok(Value::Bool(false));
                }
            }
            Value::Bool(true)
        }
        Node::Or(parts) => {
            for p in parts {
                if eval(p, scope)?.truthy() {
                    return Ok(Value::Bool(true));
                }
            }
            Value::Bool(false)
        }
        Node::Cmp(first, rest) => {
            let mut left = eval(first, scope)?;
            for (op, node) in rest {
                let right = eval(node, scope)?;
                let ok = match op {
                    CmpOp::Eq => left == right,
                    CmpOp::Ne => left != right,
                    CmpOp::Lt => left.py_cmp(&right)?.is_lt(),
                    CmpOp::Le => left.py_cmp(&right)?.is_le(),
                    CmpOp::Gt => left.py_cmp(&right)?.is_gt(),
                    CmpOp::Ge => left.py_cmp(&right)?.is_ge(),
                    CmpOp::In => value::contains(&right, &left)?,
                    CmpOp::NotIn => !value::contains(&right, &left)?,
                };
                if !ok {
                    return Ok(Value::Bool(false));
                }
                left = right;
            }
            Value::Bool(true)
        }
        Node::IfExp { test, body, orelse } => {
            if eval(test, scope)?.truthy() {
                eval(body, scope)?
            } else {
                eval(orelse, scope)?
            }
        }
        Node::Call(name, args) => call(name, &args.iter().map(|a| eval(a, scope)).collect::<Result<Vec<_>>>()?)?,
        Node::Index(target, idx) => index(&eval(target, scope)?, &eval(idx, scope)?)?,
        Node::Slice(target, parts) => {
            let target = eval(target, scope)?;
            let bound = |p: &Option<Box<Node>>| -> Result<Option<i64>> {
                p.as_ref()
                    .map(|n| match eval(n, scope)? {
                        Value::Int(i) => Ok(i),
                        Value::Bool(b) => Ok(i64::from(b)),
                        other => bail!("slice indices must be integers, not {}", other.type_name()),
                    })
                    .transpose()
            };
            slice(&target, bound(&parts[0])?, bound(&parts[1])?, bound(&parts[2])?)?
        }
    })
}

fn extremum(args: &[Value], want: std::cmp::Ordering, name: &str) -> Result<Value> {
    let items: &[Value] = match args {
        [Value::List(l)] => l,
        _ => args,
    };
    let mut best = items.first().with_context(|| format!("{name}() arg is an empty sequence"))?;
    for v in &items[1..] {
        if v.py_cmp(best)? == want {
            best = v;
        }
    }
    Ok(best.clone())
}

fn call(name: &str, args: &[Value]) -> Result<Value> {
    Ok(match (name, args) {
        ("min", _) => extremum(args, std::cmp::Ordering::Less, name)?,
        ("max", _) => extremum(args, std::cmp::Ordering::Greater, name)?,
        ("abs", [v]) => match v {
            Value::Float(f) => Value::Float(f.abs()),
            other => Value::Int(other.as_f64().with_context(|| format!("bad operand for abs(): {}", other.type_name()))?.abs() as i64),
        },
        ("len", [Value::List(l)]) => Value::Int(l.len() as i64),
        ("len", [Value::Str(s)]) => Value::Int(s.chars().count() as i64),
        ("int", [v]) => match v {
            Value::Str(s) => Value::Int(s.trim().parse().with_context(|| format!("invalid literal for int(): {s:?}"))?),
            other => Value::Int(other.as_f64().with_context(|| format!("int() of {}", other.type_name()))?.trunc() as i64),
        },
        ("bool", [v]) => Value::Bool(v.truthy()),
        ("clamp", [x, lo, hi]) => {
            // max(lo, min(hi, x)), keeping Python's choice of operand.
            let inner = if hi.py_cmp(x)?.is_lt() { hi } else { x };
            if inner.py_cmp(lo)?.is_lt() { lo.clone() } else { inner.clone() }
        }
        _ => bail!("bad call {name}() with {} arguments", args.len()),
    })
}

fn norm_index(i: i64, len: usize) -> Option<usize> {
    let j = if i < 0 { i + len as i64 } else { i };
    (0..len as i64).contains(&j).then_some(j as usize)
}

fn index(target: &Value, idx: &Value) -> Result<Value> {
    let i = match idx {
        Value::Int(i) => *i,
        Value::Bool(b) => i64::from(*b),
        other => bail!("indices must be integers, not {}", other.type_name()),
    };
    match target {
        Value::List(l) => norm_index(i, l.len()).map(|j| l[j].clone()).context("list index out of range"),
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            norm_index(i, chars.len()).map(|j| Value::Str(chars[j].to_string())).context("string index out of range")
        }
        other => bail!("{} is not subscriptable", other.type_name()),
    }
}

fn slice(target: &Value, lo: Option<i64>, hi: Option<i64>, step: Option<i64>) -> Result<Value> {
    let step = step.unwrap_or(1);
    ensure!(step != 0, "slice step cannot be zero");
    let pick = |len: usize| -> Vec<usize> {
        let n = len as i64;
        let clampi = |v: i64, lo_: i64, hi_: i64| v.max(lo_).min(hi_);
        let resolve = |v: Option<i64>, default: i64, lo_: i64, hi_: i64| {
            v.map_or(default, |x| clampi(if x < 0 { x + n } else { x }, lo_, hi_))
        };
        if step > 0 {
            let (a, b) = (resolve(lo, 0, 0, n), resolve(hi, n, 0, n));
            (a..b).step_by(step as usize).map(|i| i as usize).collect()
        } else {
            let (a, b) = (resolve(lo, n - 1, -1, n - 1), resolve(hi, -1, -1, n - 1));
            let mut out = Vec::new();
            let mut i = a;
            while i > b {
                out.push(i as usize);
                i += step;
            }
            out
        }
    };
    match target {
        Value::List(l) => Ok(Value::List(pick(l.len()).into_iter().map(|i| l[i].clone()).collect())),
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            Ok(Value::Str(pick(chars.len()).into_iter().map(|i| chars[i]).collect()))
        }
        other => bail!("{} is not subscriptable", other.type_name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Env(HashMap<&'static str, Value>, HashMap<&'static str, Value>);

    impl Scope for Env {
        fn get(&self, n: &str) -> Option<Value> {
            self.0.get(n).cloned()
        }
        fn old(&self, n: &str) -> Option<Value> {
            self.1.get(n).or_else(|| self.0.get(n)).cloned()
        }
    }

    fn run(src: &str) -> Value {
        let env = Env(
            HashMap::from([
                ("count", Value::Int(8)),
                ("status", Value::str("off")),
                ("armed", Value::Bool(true)),
                ("items", Value::List(vec![Value::Int(3), Value::Int(1), Value::Int(2)])),
                ("value", Value::Float(137.2)),
            ]),
            HashMap::from([("count", Value::Int(7))]),
        );
        Expr::compile(src).unwrap().eval(&env).unwrap()
    }

    #[test]
    fn evaluates_spec_expressions() {
        assert_eq!(run("min(count + 1, 9)"), Value::Int(9));
        assert_eq!(run("'on' if status == 'off' else 'off'"), Value::str("on"));
        assert_eq!(run("count >= old.count and armed"), Value::Bool(true));
        assert_eq!(run("0 <= count <= 9"), Value::Bool(true));
        assert_eq!(run("0 <= count < 8"), Value::Bool(false));
        assert_eq!(run("clamp(value, 0, 100)"), Value::Int(100));
        assert_eq!(run("not armed or count == 8"), Value::Bool(true));
        assert_eq!(run("90 if status == 'on' else 30"), Value::Int(30));
        assert_eq!(run("20 + 20 * count"), Value::Int(180));
        assert_eq!(run("-2 ** 2"), Value::Int(-4));
        assert_eq!(run("2 ** 3 ** 2"), Value::Int(512));
        assert_eq!(run("7 // 2 + 7 % 3"), Value::Int(4));
        assert_eq!(run("count / 2"), Value::Float(4.0));
        assert_eq!(run("1 in items and 5 not in items"), Value::Bool(true));
        assert_eq!(run("items[-1] + len(items[1:])"), Value::Int(4));
        assert_eq!(run("max(items)"), Value::Int(3));
        assert_eq!(run("(count, 1)"), Value::List(vec![Value::Int(8), Value::Int(1)]));
        assert_eq!(run("30 + 60 * (status == 'on')"), Value::Int(30));
    }

    #[test]
    fn rejects_unsafe_syntax() {
        for bad in ["__import__('os')", "foo(1)", "x.y", "{1: 2}", "lambda: 1", "count +"] {
            assert!(Expr::compile(bad).is_err(), "{bad} should not compile");
        }
    }
}
