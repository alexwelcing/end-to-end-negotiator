//! `nerve site`: the static lab for any static host. The same two pages the server serves (the drag
//! lab and the machine room), answered by the WebAssembly build of this crate instead of HTTP:
//! `site/nerve-local.js` loads the module, the specs, the machine documents, the served networks and
//! the render networks, and the pages route their `api()` calls through it. Nothing runs on a server;
//! a tick, a drag frame or a picture is a function call in the visitor's browser.

use anyhow::{bail, ensure, Context, Result};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Command;

/// What `nerve site` needs to know about the checkout.
pub struct SiteBuild {
    pub root: PathBuf,
    pub dist: PathBuf,
    pub skip_wasm: bool,
}

const INJECT: &str = "<script>window.nerveReady = new Promise(r => { window.nerveResolve = r; });</script>\n<script type=\"module\" src=\"./nerve-local.js\"></script>\n";

fn run(root: &Path, program: &str, args: &[&str]) -> Result<()> {
    println!("+ {program} {}", args.join(" "));
    let status = Command::new(program).args(args).current_dir(root).status().with_context(|| format!("running {program}"))?;
    ensure!(status.success(), "{program} failed with {status}");
    Ok(())
}

fn copy(from: &Path, to: &Path) -> Result<()> {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::copy(from, to).with_context(|| format!("copying {} to {}", from.display(), to.display()))?;
    Ok(())
}

/// A page for the static site: the shim injected into `<head>`, and routes turned into file names.
fn page(src: &Path, dst: &Path) -> Result<()> {
    let html = std::fs::read_to_string(src)?;
    ensure!(html.contains("<head>"), "{} has no <head>", src.display());
    let html = html.replacen("<head>", &format!("<head>\n{INJECT}"), 1).replace("href=\"/machine\"", "href=\"./machine.html\"").replace("href=\"/\"", "href=\"./index.html\"");
    std::fs::write(dst, html)?;
    Ok(())
}

fn stems(dir: &Path, ext: &str) -> Result<Vec<String>> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| dir.display().to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == ext))
        .filter_map(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect();
    v.sort();
    Ok(v)
}

fn dirs_with(dir: &Path, file: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.join(file).exists()).filter_map(|p| p.file_name().map(|s| s.to_string_lossy().into_owned())).collect())
        .unwrap_or_default();
    v.sort();
    v
}

impl SiteBuild {
    pub fn build(&self) -> Result<()> {
        let (root, dist) = (&self.root, &self.dist);
        if dist.exists() {
            for child in std::fs::read_dir(dist)? {
                let child = child?.path();
                if self.skip_wasm && child.file_name().is_some_and(|n| n == "pkg") {
                    continue;
                }
                if child.is_dir() { std::fs::remove_dir_all(&child)? } else { std::fs::remove_file(&child)? }
            }
        }
        std::fs::create_dir_all(dist)?;
        if !self.skip_wasm {
            // The `wasm` profile (workspace Cargo.toml): opt-level s, LTO, no debuginfo, for a smaller download.
            run(root, "cargo", &["rustc", "-p", "nerve", "--lib", "--crate-type", "cdylib", "--profile", "wasm", "--target", "wasm32-unknown-unknown", "--no-default-features", "--features", "wasm"])?;
            let wasm = root.join("target/wasm32-unknown-unknown/wasm/nerve.wasm");
            run(root, "wasm-bindgen", &["--target", "web", "--out-dir", &dist.join("pkg").to_string_lossy(), &wasm.to_string_lossy()])?;
        }
        ensure!(dist.join("pkg/nerve.js").exists(), "no wasm package in {}; run without --skip-wasm", dist.display());

        let web = root.join("crates/nerve/web");
        page(&web.join("index.html"), &dist.join("index.html"))?;
        page(&web.join("machine.html"), &dist.join("machine.html"))?;
        copy(&root.join("site/nerve-local.js"), &dist.join("nerve-local.js"))?;

        let specs = stems(&root.join("specs"), "nerve")?;
        let machines = stems(&root.join("machines"), "nerve")?;
        let models = dirs_with(&root.join("models/served"), "model.json");
        let pictures = dirs_with(&root.join("models/pictures"), crate::picture::META_FILE);
        for n in &specs {
            copy(&root.join("specs").join(format!("{n}.nerve")), &dist.join("specs").join(format!("{n}.nerve")))?;
        }
        for n in &machines {
            copy(&root.join("machines").join(format!("{n}.nerve")), &dist.join("machines").join(format!("{n}.nerve")))?;
        }
        for n in &models {
            for f in ["model.json", "model.safetensors"] {
                copy(&root.join("models/served").join(n).join(f), &dist.join("models").join(n).join(f))?;
            }
        }
        for n in &pictures {
            for f in [crate::picture::META_FILE, crate::picture::WEIGHTS_FILE] {
                copy(&root.join("models/pictures").join(n).join(f), &dist.join("models").join(n).join(f))?;
            }
        }
        let sha = Command::new("git").args(["rev-parse", "--short", "HEAD"]).current_dir(root).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
        let manifest = json!({
            "specs": specs, "machines": machines, "models": models, "pictures": pictures,
            "built": {"commit": sha, "at": chrono_free_now()},
        });
        std::fs::write(dist.join("manifest.json"), format!("{}\n", serde_json::to_string_pretty(&manifest)?))?;
        // Caching for the big assets; the pages and the manifest stay fresh.
        // The pages fetch the module and the networks under URLs that carry the build (site/nerve-local.js),
        // so those can be cached for good; the manifest, pages, specs and machines are revalidated each visit.
        std::fs::write(dist.join("_headers"), "/pkg/*\n  Cache-Control: public, max-age=31536000, immutable\n/models/*\n  Cache-Control: public, max-age=31536000, immutable\n")?;
        let total: u64 = walk(dist)?.iter().map(|p| p.metadata().map(|m| m.len()).unwrap_or(0)).sum();
        let wasm_size = dist.join("pkg/nerve_bg.wasm").metadata().map(|m| m.len()).unwrap_or(0);
        println!(
            "{}: {} specs, {} machines, {} networks, {} pictures, wasm {:.1} MB, total {:.1} MB",
            dist.display(),
            specs.len(),
            machines.len(),
            models.len(),
            pictures.len(),
            wasm_size as f64 / 1e6,
            total as f64 / 1e6
        );
        if pictures.is_empty() {
            bail!("no render networks in models/pictures: the site would draw the specs' own pictures; run `nerve train-picture`");
        }
        Ok(())
    }
}

fn walk(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() { out.extend(walk(&p)?) } else { out.push(p) }
    }
    Ok(out)
}

/// An ISO-8601 UTC timestamp without a date crate: seconds since the epoch, civil-from-days.
fn chrono_free_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (hh, mm, ss) = (rem / 3600, rem % 3600 / 60, rem % 60);
    // Howard Hinnant's civil_from_days.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_timestamp_is_iso_8601() {
        let t = chrono_free_now();
        assert_eq!(t.len(), 20, "{t}");
        assert!(t.ends_with('Z') && &t[4..5] == "-" && &t[10..11] == "T");
    }
}
