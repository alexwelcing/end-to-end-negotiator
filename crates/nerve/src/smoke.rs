//! `nerve smoke <url>`: asks a running lab what CI needs to know. The served networks load and
//! answer, the machine room ticks the chain by its networks in agreement with the composed oracle,
//! and the render networks draw. A small HTTP/1.1 client over `std::net` keeps it dependency-free.

use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Value as J};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

const TICKS: usize = 20;

/// One request to `base` (`http://host:port`), returning the JSON body.
fn http(base: &str, method: &str, path: &str, body: Option<&J>) -> Result<J> {
    let hostport = base.strip_prefix("http://").context("the smoke test speaks plain http")?.trim_end_matches('/');
    let host = hostport.split(':').next().unwrap_or(hostport);
    let mut stream = TcpStream::connect(hostport).with_context(|| format!("connecting to {hostport}"))?;
    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{payload}",
        payload.len()
    );
    stream.write_all(req.as_bytes())?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let text = String::from_utf8_lossy(&raw);
    let (head, body) = text.split_once("\r\n\r\n").context("no HTTP header terminator")?;
    let status: u16 = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).context("no HTTP status")?;
    let j: J = serde_json::from_str(body).with_context(|| format!("{method} {path}: {status}: {}", &body[..body.len().min(200)]))?;
    if let Some(err) = j.get("error").and_then(J::as_str) {
        bail!("{method} {path}: {err}");
    }
    ensure!((200..300).contains(&status), "{method} {path}: HTTP {status}");
    Ok(j)
}

pub fn run(base: &str) -> Result<()> {
    let specs = http(base, "GET", "/api/specs", None)?;
    let specs = specs.as_array().context("/api/specs is a list")?;
    let served: Vec<&str> = specs.iter().filter(|s| !s["model"].is_null()).filter_map(|s| s["name"].as_str()).collect();
    println!("{} interfaces, {} with a network: {}", specs.len(), served.len(), served.join(", "));
    ensure!(!served.is_empty(), "no model loaded");

    let r = http(base, "POST", "/api/hanoi/step", Some(&json!({"state": {"d1": 0, "d2": 0, "d3": 0, "d4": 0}, "action": "move", "params": {"src": 0, "dst": 2}})))?;
    println!("hanoi move(0, 2): {} {}", r["verdict"], r["network"]);
    ensure!(r["verdict"] == "agrees", "hanoi step: {r}");

    // Every interface with a scene draws; the render network's error against the spec's drawing is reported.
    let mut drawn = 0;
    let mut worst = (0.0f64, String::new());
    for s in specs {
        if s["scene"].is_null() {
            continue;
        }
        let name = s["name"].as_str().unwrap_or_default();
        let p = http(base, "POST", &format!("/api/{name}/picture"), Some(&json!({"state": s["initial"]})))?;
        ensure!(p["oracle"].is_string() && p["w"].as_u64().is_some(), "{name}: no picture");
        if let Some(mae) = p["mae"].as_f64() {
            drawn += 1;
            if mae > worst.0 {
                worst = (mae, name.to_string());
            }
        }
    }
    println!("{drawn} render networks draw their initial state; worst pixel error {:.3} ({})", worst.0, worst.1);

    let machines = http(base, "GET", "/api/machines", None)?;
    let chain = machines.as_array().and_then(|ms| ms.iter().find(|m| m["name"] == "chain")).context("no chain machine")?;
    let blocks: Vec<String> = chain["blocks"].as_array().unwrap_or(&vec![]).iter().map(|b| format!("{}:{}{}", b["instance"].as_str().unwrap_or(""), b["type"].as_str().unwrap_or(""), if b["network"].as_bool().unwrap_or(false) { "" } else { " (spec only)" })).collect();
    println!("{}", blocks.join(", "));
    ensure!(chain["blocks"].as_array().unwrap_or(&vec![]).iter().all(|b| b["network"].as_bool().unwrap_or(false)), "a chain block has no network");
    let mut state = chain["initial"].clone();
    let advance = json!([{"block": "clock", "action": "advance"}]);
    let mut fired = std::collections::BTreeSet::new();
    for i in 0..TICKS {
        let t = http(base, "POST", "/api/machine/chain/tick", Some(&json!({"state": state, "external": advance})))?;
        ensure!(t["agrees"].as_bool() == Some(true), "tick {}: the networks disagree with the composed oracle: {t}", i + 1);
        ensure!(t["violations"].as_array().is_some_and(Vec::is_empty), "tick {}: {}", i + 1, t["violations"]);
        for s in t["steps"].as_array().unwrap_or(&vec![]) {
            if !s["via"].is_null() {
                fired.insert(format!("{}.{}", s["via"]["block"].as_str().unwrap_or(""), s["via"]["event"].as_str().unwrap_or("")));
            }
        }
        if i == 0 {
            let names: Vec<String> = t["steps"].as_array().unwrap_or(&vec![]).iter().map(|s| format!("{}.{}", s["block"].as_str().unwrap_or(""), s["action"].as_str().unwrap_or(""))).collect();
            println!("tick 1: {} agrees, {:.2} ms", names.join(" -> "), t["ms"].as_f64().unwrap_or(0.0));
        }
        state = t["state"].clone();
    }
    println!("{TICKS} ticks agree with the composed oracle; wires fired: {}", fired.iter().cloned().collect::<Vec<_>>().join(", "));
    ensure!(!fired.is_empty(), "no wire fired");
    println!("smoke ok");
    Ok(())
}
