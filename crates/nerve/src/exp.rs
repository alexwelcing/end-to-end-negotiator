//! The experiment log (`experiments/round2/results.jsonl`): `nerve record` scores a trained model and
//! appends its rows, `nerve serve-set` picks the served network per spec from the log and copies it to
//! `models/served`, and `nerve summarize` prints the log as markdown tables. Lines are only ever
//! appended, never edited, so every served network is traceable to the run that produced it.

use crate::spec::Spec;
use crate::train::{evaluate, Candidates, HoldOut, Model, TrainConfig};
use anyhow::{ensure, Context, Result};
use candle_core::Device;
use serde_json::{json, Value as J};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

/// A row of `results.jsonl` as `nerve eval --json` prints it: the model rescored on the hold-out it was
/// trained with, answering from `candidates` (the spec's valid states, or every state).
pub fn eval_line(spec: &Spec, model_dir: &Path, candidates: Candidates) -> Result<J> {
    let m = Model::load_as(spec.clone(), model_dir, &Device::Cpu, Some(candidates))?;
    let t: TrainConfig = serde_json::from_value(m.meta().context("checkpoint has no model.json")?["train"].clone())?;
    let r = evaluate(&m, &HoldOut::new(spec, &t)?)?;
    Ok(json!({
        "spec": spec.name,
        "model": model_dir.display().to_string().replace('\\', "/"),
        "candidates": m.candidates,
        "candidate_count": m.candidate_count(),
        "seed": t.seed,
        "holdout": t.holdout,
        "salt": t.holdout_salt,
        "rule": t.holdout_rule,
        "epochs": t.epochs,
        "fact_weight": t.fact_weight,
        "heldout": r["heldout"],
        "seen": r["seen"],
        "all": r["all"],
        "hole_examples": r["hole_examples"],
    }))
}

/// Scores one trained model and appends its rows to `log`: a finite model twice (answering from the
/// spec's valid states, and from every state), a continuous one once from its own checkpoint.
pub fn record(spec_path: &Path, model_dir: &Path, tag: &str, log: &Path) -> Result<Vec<J>> {
    let spec = Spec::load(spec_path)?;
    let meta: J = serde_json::from_str(&std::fs::read_to_string(model_dir.join("model.json")).with_context(|| format!("reading {}", model_dir.join("model.json").display()))?)?;
    let cost = json!({"params": meta["eval"]["params"], "seconds": meta["eval"]["seconds"]});
    let mut rows = Vec::new();
    if meta["kind"] == "continuous" {
        let (e, t) = (&meta["eval"], &meta["train"]);
        rows.push(json!({
            "spec": spec.name, "model": model_dir.display().to_string().replace('\\', "/"), "mode": "network",
            "seed": t["seed"], "holdout": J::Null, "salt": J::Null, "rule": J::Null, "epochs": t["epochs"],
            "heldout": e["all"], "all": e["all"], "continuous": true, "head": e["head"],
            "params": cost["params"], "seconds": cost["seconds"],
        }));
    } else {
        let trained = meta.get("candidates").cloned().unwrap_or(json!("all"));
        for (mode, cands) in [("spec-assisted", Candidates::Valid), ("network-only", Candidates::All)] {
            let mut row = eval_line(&spec, model_dir, cands)?;
            row["mode"] = json!(mode);
            row["trained_candidates"] = trained.clone();
            row["params"] = cost["params"].clone();
            row["seconds"] = cost["seconds"].clone();
            rows.push(row);
        }
    }
    let mut blob = String::new();
    for r in &rows {
        // The checkpoint's own fingerprint names the rules it was trained on (checkpoints written before
        // fingerprints have none), so a later retraining of a changed spec outranks it when the served set
        // is picked.
        let mut line = json!({"round": 2, "tag": tag, "spec_fingerprint": meta.get("spec_fingerprint").cloned().unwrap_or(J::Null)});
        for (k, v) in r.as_object().into_iter().flatten() {
            line[k] = v.clone();
        }
        blob.push_str(&line.to_string());
        blob.push('\n');
    }
    // One appending write per model, so concurrent sweeps cannot interleave lines.
    let mut f = std::fs::OpenOptions::new().append(true).create(true).open(log).with_context(|| format!("opening {}", log.display()))?;
    f.write_all(blob.as_bytes())?;
    for r in &rows {
        let h = &r["heldout"];
        println!("{:<16} {tag:<22} {:<14} held-out {:.4} of {}", spec.name, r["mode"].as_str().unwrap_or(""), h["exact"].as_f64().unwrap_or(0.0), h["n"]);
    }
    Ok(rows)
}

fn read_log(log: &Path) -> Result<Vec<J>> {
    let text = std::fs::read_to_string(log).with_context(|| format!("reading {}", log.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| Ok(serde_json::from_str(l)?)).collect()
}

/// A rerun of the same (spec, tag, seed, mode) replaces the earlier row instead of counting twice.
fn latest(rows: Vec<J>) -> BTreeMap<String, J> {
    let mut out = BTreeMap::new();
    for r in rows {
        let key = format!("{}|{}|{}|{}", r["spec"], r["tag"], r["seed"], r["mode"]);
        out.insert(key, r);
    }
    out
}

/// Builds `models/served` from the log: for each spec, the full-spec run with the highest score
/// (a run recorded against the spec as it is now, by its fingerprint, over one recorded before the spec
/// changed; then spec-assisted exact for finite specs, within-tolerance for continuous ones; fewest
/// violations; the lowest seed on ties) whose checkpoint still loads against the spec is copied
/// unchanged. A spec whose rules changed is retrained; its older checkpoints either no longer load or
/// rank below the run that knows the new rules. The choice is written to `served.json`.
/// A run's rank in `serve_set`: recorded against the current spec, exactness, fewer violations, a step head, the lower seed.
type Rank = (f64, f64, f64, f64, f64);
/// Per spec: the spec as it is now, and its full-spec runs best first.
type Ranked = BTreeMap<String, (Spec, Vec<(J, Rank)>)>;

/// Every full-spec run per spec in the log, best first, against the specs as they are now: recorded
/// against the current spec, then exactness, fewer violations, a step head, the lower seed.
fn ranked_runs(log: &Path, root: &Path) -> Result<Ranked> {
    let rows = latest(read_log(log)?);
    let mut candidates: BTreeMap<String, Vec<(J, Rank)>> = BTreeMap::new();
    let mut specs: BTreeMap<String, Spec> = BTreeMap::new();
    for r in rows.values() {
        if r["tag"] != "full" || !(r["mode"] == "spec-assisted" || r["mode"] == "network") {
            continue;
        }
        let name = r["spec"].as_str().unwrap_or("").to_string();
        if !specs.contains_key(&name) {
            specs.insert(name.clone(), Spec::load(&root.join("specs").join(format!("{name}.nerve")))?);
        }
        let current = f64::from(r["spec_fingerprint"].as_str() == Some(specs[&name].fingerprint().as_str()));
        // A continuous network with a step head answers in one pass instead of descent: at equal
        // exactness it is the one to serve.
        let head = f64::from(r["head"].is_object());
        let score = (current, r["all"]["exact"].as_f64().unwrap_or(0.0), -r["all"]["violation_rate"].as_f64().unwrap_or(1.0), head, -r["seed"].as_f64().unwrap_or(0.0));
        candidates.entry(name).or_default().push((r.clone(), score));
    }
    ensure!(!candidates.is_empty(), "no full-spec runs in {}", log.display());
    Ok(candidates
        .into_iter()
        .map(|(spec, mut cs)| {
            cs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let sp = specs.remove(&spec).expect("a spec per candidate");
            (spec, (sp, cs))
        })
        .collect())
}

pub fn serve_set(log: &Path, root: &Path, served: &Path, manifest: &Path) -> Result<()> {
    let mut best: BTreeMap<String, (J, Rank)> = BTreeMap::new();
    for (spec, (sp, cs)) in ranked_runs(log, root)? {
        let mut chosen = None;
        for (r, s) in cs {
            let dir = root.join(r["model"].as_str().unwrap_or("").replace('\\', "/"));
            match crate::train::load_learned(sp.clone(), &dir, &Device::Cpu) {
                Ok(_) => {
                    chosen = Some((r, s));
                    break;
                }
                Err(e) => println!("{spec:<18} skipping seed {} ({}): {e:#}", r["seed"], dir.display()),
            }
        }
        match chosen {
            Some(c) => {
                best.insert(spec, c);
            }
            None => println!("{spec:<18} no checkpoint loads against the current spec"),
        }
    }
    ensure!(!best.is_empty(), "no full-spec runs in {}", log.display());
    if served.exists() {
        std::fs::remove_dir_all(served)?;
    }
    std::fs::create_dir_all(served)?;
    let mut out = serde_json::Map::new();
    for (spec, (r, _)) in &best {
        // Older rows were written on Windows with backslashes.
        let src = root.join(r["model"].as_str().context("row has no model path")?.replace('\\', "/"));
        let dst = served.join(spec);
        std::fs::create_dir_all(&dst)?;
        for f in ["model.json", "model.safetensors"] {
            std::fs::copy(src.join(f), dst.join(f)).with_context(|| format!("copying {}", src.join(f).display()))?;
        }
        out.insert(spec.clone(), json!({"model": r["model"], "seed": r["seed"], "exact": r["all"]["exact"], "violation_rate": r["all"]["violation_rate"], "cells": r["all"]["n"]}));
        println!("{spec:<18} seed {}  exact {:.4}  violations {:.4}  <- {}", r["seed"], r["all"]["exact"].as_f64().unwrap_or(0.0), r["all"]["violation_rate"].as_f64().unwrap_or(0.0), r["model"].as_str().unwrap_or(""));
    }
    std::fs::write(manifest, format!("{}\n", serde_json::to_string_pretty(&J::Object(out))?))?;
    Ok(())
}

fn split_name(r: &J) -> String {
    if r["continuous"].as_bool().unwrap_or(false) {
        return "random pairs".into();
    }
    if let Some(rule) = r["rule"].as_str() {
        return format!("rule `{rule}`");
    }
    let holdout = r["holdout"].as_f64().unwrap_or(0.0);
    if holdout == 0.0 {
        return "full spec".into();
    }
    let salt = match r["salt"].as_str() {
        None | Some("nerve") => String::new(),
        Some(s) => format!(", salt {s}"),
    };
    format!("{:.0}% held out{salt}", holdout * 100.0)
}

fn stat(xs: &[Option<f64>]) -> String {
    let xs: Vec<f64> = xs.iter().flatten().copied().collect();
    if xs.is_empty() {
        return "—".into();
    }
    let m = xs.iter().sum::<f64>() / xs.len() as f64;
    if xs.len() == 1 {
        return format!("{:.1}%", m * 100.0);
    }
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    format!("{:.1} ± {:.1}%", m * 100.0, var.sqrt() * 100.0)
}

/// The log as markdown: mean ± sd over seeds per spec, split and mode (runs trained differently are
/// never pooled).
pub fn summarize(log: &Path, wanted: &[String]) -> Result<String> {
    let rows = latest(read_log(log)?);
    let mut groups: BTreeMap<(String, String, String, String, String), Vec<J>> = BTreeMap::new();
    for r in rows.values() {
        let spec = r["spec"].as_str().unwrap_or("").to_string();
        if !wanted.is_empty() && !wanted.contains(&spec) {
            continue;
        }
        let key = (spec, split_name(r), r["mode"].as_str().unwrap_or("").to_string(), r["epochs"].to_string(), r["fact_weight"].to_string());
        groups.entry(key).or_default().push(r.clone());
    }
    let mut out = String::from("| spec | split | mode | seeds | held-out cells | held-out exact | violations | hole, unseen exact | hole, unseen violations |\n|---|---|---|---|---|---|---|---|---|\n");
    for ((spec, split, mode, epochs, fw), rs) in &groups {
        let cells = rs[0]["heldout"]["n"].as_u64().unwrap_or(0);
        let all = rs[0]["all"]["n"].as_u64().unwrap_or(0);
        let continuous = rs[0]["continuous"].as_bool().unwrap_or(false);
        let cells_txt = if cells == all && !continuous { format!("{cells} (all)") } else { cells.to_string() };
        let mut split = split.clone();
        if !(epochs == "null" || epochs == "300") || !(fw == "null" || fw == "10.0") {
            split = format!("{split} ({epochs} epochs, fact {fw})");
        }
        let seeds: std::collections::BTreeSet<String> = rs.iter().map(|r| r["seed"].to_string()).collect();
        let held: Vec<Option<f64>> = rs.iter().map(|r| r["heldout"]["exact"].as_f64()).collect();
        let viol: Vec<Option<f64>> = rs.iter().map(|r| r["all"]["violation_rate"].as_f64()).collect();
        let hole_e: Vec<Option<f64>> = rs.iter().map(|r| r["hole_examples"]["heldout"]["exact"].as_f64()).collect();
        let hole_v: Vec<Option<f64>> = rs.iter().map(|r| r["hole_examples"]["heldout"]["violation_rate"].as_f64()).collect();
        out.push_str(&format!("| {spec} | {split} | {mode} | {} | {cells_txt} | {} | {} | {} | {} |\n", seeds.len(), stat(&held), stat(&viol), stat(&hole_e), stat(&hole_v)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_served_set_is_what_the_log_says() {
        // serve_set's choice, replayed over the committed log, must match the committed manifest.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let log = root.join("experiments/round2/results.jsonl");
        let manifest: J = serde_json::from_str(&std::fs::read_to_string(root.join("experiments/round2/served.json")).unwrap()).unwrap();
        // The committed served set is the committed log's own best run per spec, by `serve_set`'s ranking
        // against the specs as they are now. `serve_set` also skips a checkpoint that does not load, which
        // the committed set never needs (its first-ranked run loads for every spec); the checkpoints are not
        // in the repository, so this ranks without loading them.
        let ranked = ranked_runs(&log, &root).unwrap();
        for (spec, entry) in manifest.as_object().unwrap() {
            let top = &ranked[spec].1[0].0;
            assert_eq!(top["model"].as_str().unwrap().replace('\\', "/"), entry["model"].as_str().unwrap().replace('\\', "/"), "{spec}");
        }
        assert!(summarize(&log, &["toggle".to_string()]).unwrap().contains("| toggle | full spec | spec-assisted |"));
        assert_eq!(stat(&[Some(1.0), Some(0.5)]), "75.0 ± 35.4%");
        assert_eq!(split_name(&json!({"holdout": 0.15})), "15% held out");
    }
}
