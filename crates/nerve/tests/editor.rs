//! docs/rube-goldberg.md, M8: the machine-document editor. A document posted to the runtime is re-parsed
//! against the specs the lab serves and replaces the machine of that name; a run edited to start one tick
//! from the bell rings it on the next tick. Blocks without a served network step by their oracles here, so
//! this runs without any trained model.

use anyhow::{Context, Result};
use nerve::machine::MachineSpec;
use nerve::machine_api::{MachineEntry, TickReq};
use nerve::serve::{App, Entry};
use nerve::spec::Spec;
use serde_json::{json, Value as J};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The lab as `nerve serve` builds it, without networks: every spec, and the light machine.
fn lab() -> Result<(App, String)> {
    let mut entries = Vec::new();
    for e in std::fs::read_dir(root().join("specs"))? {
        let p = e?.path();
        if p.extension().is_some_and(|x| x == "nerve") {
            let source = std::fs::read_to_string(&p)?;
            entries.push(Entry { spec: Spec::parse(&source)?, model: None, picture: None, source });
        }
    }
    let app = App::new(entries, 1.0);
    let text = std::fs::read_to_string(root().join("machines/light.nerve"))?;
    let spec = MachineSpec::load(&root().join("machines/light.nerve"))?;
    app.add_machine(MachineEntry { spec, source: text.clone() });
    Ok((app, text))
}

fn listing(app: &App, name: &str) -> Result<J> {
    app.machines_json().as_array().context("a list")?.iter().find(|m| m["name"] == name).cloned().context("the machine is listed")
}

#[test]
fn a_document_edited_to_skip_to_the_bell_rings_it_on_the_next_tick() -> Result<()> {
    let (app, text) = lab()?;
    // The edit: the bucket starts at eight, armed, with its relay armed, so the next tick fills it, flips
    // the light and rings the bell.
    let edited = text.replace("  bucket: {count: 0, armed: false}", "  bucket: {count: 8, armed: true}\n  r_bucket: {armed: true}");
    assert_ne!(edited, text, "the edit found the line it changes");
    let name = app.add_machine_text(&edited)?;
    assert_eq!(name, "light");
    let m = listing(&app, "light")?;
    assert_eq!(m["initial"]["bucket"]["count"], json!(8), "the listing shows the edited document's initial state");
    let req: TickReq = serde_json::from_value(json!({"state": m["initial"], "external": [{"block": "clock", "action": "advance", "params": {}}]}))?;
    let tick = app.machine_tick("light", &req)?;
    assert!(tick["violations"].as_array().is_some_and(Vec::is_empty), "{tick}");
    assert_eq!(tick["derived"]["rang"], json!(1), "the bell rang on the next tick: {}", tick["derived"]);
    assert_eq!(tick["derived"]["lit"], json!(true));
    Ok(())
}

#[test]
fn a_document_that_does_not_parse_is_refused_and_the_old_machine_stays() -> Result<()> {
    let (app, text) = lab()?;
    let broken = text.replace("do: bell.ring", "do: bell.gong");
    let err = app.add_machine_text(&broken).err().context("a wire to a missing action is refused")?;
    assert!(format!("{err:#}").contains("gong"), "{err:#}");
    assert_eq!(listing(&app, "light")?["source"], json!(text), "the machine the lab serves is unchanged");
    Ok(())
}

#[test]
fn keys_resolve_against_the_machine_state() -> Result<()> {
    let (app, _) = lab()?;
    let m = listing(&app, "light")?;
    let keys: Vec<&str> = m["bindings"].as_array().context("bindings listed")?.iter().filter_map(|b| b["key"].as_str()).collect();
    assert!(keys.contains(&"Space") && keys.contains(&"g") && keys.contains(&"b"), "{keys:?}");
    let ask = |key: &str| -> Result<J> { app.dispatch("/api/machine/light/binding", &json!({"state": m["initial"], "key": key})) };
    assert_eq!(ask("Space")?, json!({"block": "marble", "action": "drop", "params": {}}));
    // The gate: from the top track the key sends the block to the middle one, at its current position
    // clamped into that track's reach (60..180), so from the start it lands at 60 (keys are
    // case-insensitive and payloads are evaluated over the state).
    let gate = ask("G")?;
    assert_eq!((gate["block"].as_str(), gate["action"].as_str(), gate["params"]["t"].as_i64()), (Some("track"), Some("move"), Some(1)), "{gate}");
    assert_eq!(gate["params"]["to"].as_f64(), Some(60.0), "{gate}");
    assert_eq!(ask("b")?, json!({"verb": "break"}));
    assert_eq!(ask("x")?, J::Null);
    Ok(())
}
