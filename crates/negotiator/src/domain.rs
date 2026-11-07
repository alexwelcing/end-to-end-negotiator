//! The object-division negotiation domain: each agent sees item counts and
//! its own private values, and the pair must agree on how to split items.

use anyhow::{bail, ensure, Context, Result};
use regex::Regex;
use std::sync::OnceLock;

pub const NO_AGREEMENT: &str = "<no_agreement>";
pub const DISCONNECT: &str = "<disconnect>";

fn item_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^item([0-9])=([0-9\-]+)$").expect("valid regex"))
}

/// Returns true for item selection tokens such as `item0=2`.
pub fn is_item_token(word: &str) -> bool {
    item_pattern().is_match(word)
}

#[derive(Debug, Clone, Default)]
pub struct ObjectDivision;

impl ObjectDivision {
    /// Number of tokens in a full selection: our half followed by the partner's.
    pub const fn selection_length(&self) -> usize {
        6
    }

    /// Number of item types.
    pub const fn input_length(&self) -> usize {
        3
    }

    /// Splits a context `[cnt0, val0, cnt1, val1, ...]` into counts and values.
    pub fn parse_context(&self, ctx: &[String]) -> Result<(Vec<i64>, Vec<i64>)> {
        ensure!(ctx.len().is_multiple_of(2), "context must have an even length: {ctx:?}");
        let nums = ctx
            .iter()
            .map(|t| t.parse::<i64>().with_context(|| format!("bad context token {t:?}")))
            .collect::<Result<Vec<_>>>()?;
        let cnts = nums.iter().step_by(2).copied().collect();
        let vals = nums.iter().skip(1).step_by(2).copied().collect();
        Ok((cnts, vals))
    }

    /// Enumerates every way to split the items, plus the two failure outcomes.
    /// Each choice lists our items followed by the partner's.
    pub fn generate_choices(&self, ctx: &[String]) -> Result<Vec<Vec<String>>> {
        let (cnts, _) = self.parse_context(ctx)?;
        let mut choices = Vec::new();
        let mut picked = Vec::with_capacity(cnts.len());
        enumerate_splits(&cnts, &mut picked, &mut choices);
        for token in [NO_AGREEMENT, DISCONNECT] {
            choices.push(vec![token.to_string(); self.selection_length()]);
        }
        Ok(choices)
    }

    /// Parses `itemN=K` into `(N, K)`.
    pub fn parse_choice(&self, choice: &str) -> Result<(usize, i64)> {
        let caps = item_pattern()
            .captures(choice)
            .with_context(|| format!("invalid choice token {choice:?}"))?;
        Ok((caps[1].parse()?, caps[2].parse()?))
    }

    /// Scores our half of `choice` under our own values.
    pub fn score(&self, ctx: &[String], choice: &[String]) -> Result<i64> {
        ensure!(choice.len() == self.selection_length(), "bad choice length {choice:?}");
        let ours = &choice[..choice.len() / 2];
        if ours[0] == NO_AGREEMENT || ours[0] == DISCONNECT {
            return Ok(0);
        }
        let (_, vals) = self.parse_context(ctx)?;
        ours.iter().zip(vals).enumerate().try_fold(0, |acc, (i, (c, v))| {
            let (idx, cnt) = self.parse_choice(c)?;
            ensure!(idx == i, "choice {c:?} out of order");
            Ok(acc + cnt * v)
        })
    }

    /// Parses a human's space-separated item counts, e.g. `"1 0 2"`.
    pub fn parse_human_choice(&self, ctx: &[String], input: &str) -> Result<Vec<String>> {
        let (cnts, _) = self.parse_context(ctx)?;
        let picks = input
            .split_whitespace()
            .map(|t| t.parse::<i64>().with_context(|| format!("not a number: {t:?}")))
            .collect::<Result<Vec<_>>>()?;
        if picks.len() != cnts.len() {
            bail!("expected {} numbers, got {}", cnts.len(), picks.len());
        }
        picks
            .iter()
            .zip(&cnts)
            .enumerate()
            .map(|(i, (&x, &n))| {
                ensure!((0..=n).contains(&x), "item{i}: {x} is outside 0..={n}");
                Ok(format!("item{i}={x}"))
            })
            .collect()
    }

    /// Scores both agents' choices; agreement means the items add up exactly.
    pub fn score_choices(&self, choices: &[Vec<String>], ctxs: &[Vec<String>]) -> Result<(bool, Vec<i64>)> {
        ensure!(choices.len() == ctxs.len(), "one choice per agent expected");
        let (cnts, _) = self.parse_context(&ctxs[0])?;
        let mut scores = vec![0; ctxs.len()];
        let mut agree = true;
        for (i, &n) in cnts.iter().enumerate() {
            let mut left = n;
            for (agent, (choice, ctx)) in choices.iter().zip(ctxs).enumerate() {
                let taken = taken_count(&choice[i]);
                let value: i64 = ctx[2 * i + 1].parse()?;
                left -= taken;
                scores[agent] += value * taken;
            }
            agree &= left == 0;
        }
        Ok((agree, scores))
    }
}

/// Reads the trailing digit of an `itemN=K` token; non-item tokens take nothing.
fn taken_count(token: &str) -> i64 {
    token
        .chars()
        .last()
        .and_then(|c| c.to_digit(10))
        .map_or(0, i64::from)
}

fn enumerate_splits(cnts: &[i64], picked: &mut Vec<i64>, out: &mut Vec<Vec<String>>) {
    let idx = picked.len();
    if idx == cnts.len() {
        let ours = picked.iter().enumerate().map(|(i, c)| format!("item{i}={c}"));
        let theirs = picked
            .iter()
            .zip(cnts)
            .enumerate()
            .map(|(i, (c, n))| format!("item{i}={}", n - c));
        out.push(ours.chain(theirs).collect());
        return;
    }
    for c in 0..=cnts[idx] {
        picked.push(c);
        enumerate_splits(cnts, picked, out);
        picked.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn generates_all_splits_plus_failures() {
        let d = ObjectDivision;
        let choices = d.generate_choices(&ctx("1 4 4 1 1 2")).unwrap();
        assert_eq!(choices.len(), 2 * 5 * 2 + 2);
        assert_eq!(choices[0], ctx("item0=0 item1=0 item2=0 item0=1 item1=4 item2=1"));
        assert_eq!(choices.last().unwrap()[0], DISCONNECT);
    }

    #[test]
    fn scores_own_half() {
        let d = ObjectDivision;
        let c = ctx("item0=1 item1=0 item2=1 item0=0 item1=4 item2=0");
        assert_eq!(d.score(&ctx("1 4 4 1 1 2"), &c).unwrap(), 6);
        assert_eq!(d.score(&ctx("1 4 4 1 1 2"), &vec![NO_AGREEMENT.into(); 6]).unwrap(), 0);
    }

    #[test]
    fn score_choices_detects_agreement() {
        let d = ObjectDivision;
        let ctxs = vec![ctx("1 4 4 1 1 2"), ctx("1 0 4 2 1 2")];
        let agree = vec![ctx("item0=1 item1=0 item2=1"), ctx("item0=0 item1=4 item2=0")];
        assert_eq!(d.score_choices(&agree, &ctxs).unwrap(), (true, vec![6, 8]));
        let greedy = vec![ctx("item0=1 item1=4 item2=1"), ctx("item0=0 item1=4 item2=0")];
        assert!(!d.score_choices(&greedy, &ctxs).unwrap().0);
    }

    #[test]
    fn parses_human_choice() {
        let d = ObjectDivision;
        let c = ctx("1 4 4 1 1 2");
        assert_eq!(d.parse_human_choice(&c, " 1 2 0 ").unwrap(), ctx("item0=1 item1=2 item2=0"));
        assert!(d.parse_human_choice(&c, "2 0 0").is_err());
        assert!(d.parse_human_choice(&c, "1 0").is_err());
    }
}
