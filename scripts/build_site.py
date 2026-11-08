"""Build the static lab for Cloudflare Pages (or any static host) into site/dist.

usage: python scripts/build_site.py [--skip-wasm]

The site is the same two pages the server serves (the drag lab and the machine room), answered by the
WebAssembly build of crates/nerve instead of HTTP: `site/nerve-local.js` loads the module, the specs, the
machine documents and the served networks, and the pages route their `api()` calls through it. Nothing
runs on a server; a tick or a drag frame is a function call in the visitor's browser.

Deploy with `wrangler pages deploy site/dist --project-name <project>`.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "site" / "dist"
WASM = ROOT / "target" / "wasm32-unknown-unknown" / "wasm" / "nerve.wasm"
INJECT = (
    '<script>window.nerveReady = new Promise(r => { window.nerveResolve = r; });</script>\n'
    '<script type="module" src="./nerve-local.js"></script>\n'
)


def run(*cmd: str) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_wasm() -> None:
    # The `wasm` profile (workspace Cargo.toml): opt-level s, LTO, no debuginfo, for a smaller download.
    run("cargo", "rustc", "-p", "nerve", "--lib", "--crate-type", "cdylib", "--profile", "wasm",
        "--target", "wasm32-unknown-unknown", "--no-default-features", "--features", "wasm")
    run("wasm-bindgen", "--target", "web", "--out-dir", str(DIST / "pkg"), str(WASM))


def page(src: Path, dst: Path) -> None:
    html = src.read_text(encoding="utf-8")
    assert "<head>" in html, src
    html = html.replace("<head>", "<head>\n" + INJECT, 1)
    # Static hosts serve files, not routes: point the pages at each other by file name.
    html = html.replace('href="/machine"', 'href="./machine.html"').replace('href="/"', 'href="./index.html"')
    dst.write_text(html, encoding="utf-8")


def main() -> None:
    skip_wasm = "--skip-wasm" in sys.argv
    if DIST.exists():
        for child in DIST.iterdir():
            if child.name == "pkg" and skip_wasm:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    DIST.mkdir(parents=True, exist_ok=True)
    if not skip_wasm:
        build_wasm()
    assert (DIST / "pkg" / "nerve.js").exists(), "no wasm package; run without --skip-wasm"

    web = ROOT / "crates" / "nerve" / "web"
    page(web / "index.html", DIST / "index.html")
    page(web / "machine.html", DIST / "machine.html")
    shutil.copy(ROOT / "site" / "nerve-local.js", DIST / "nerve-local.js")

    specs = sorted(p.stem for p in (ROOT / "specs").glob("*.nerve"))
    machines = sorted(p.stem for p in (ROOT / "machines").glob("*.nerve"))
    models = sorted(p.parent.name for p in (ROOT / "models" / "served").glob("*/model.json"))
    (DIST / "specs").mkdir()
    (DIST / "machines").mkdir()
    for name in specs:
        shutil.copy(ROOT / "specs" / f"{name}.nerve", DIST / "specs" / f"{name}.nerve")
    for name in machines:
        shutil.copy(ROOT / "machines" / f"{name}.nerve", DIST / "machines" / f"{name}.nerve")
    for name in models:
        (DIST / "models" / name).mkdir(parents=True)
        for f in ("model.json", "model.safetensors"):
            shutil.copy(ROOT / "models" / "served" / name / f, DIST / "models" / name / f)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {"specs": specs, "machines": machines, "models": models, "built": {"commit": sha, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    (DIST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    # Long-lived caching for the hashed-by-content-in-practice assets; the pages and manifest stay fresh.
    (DIST / "_headers").write_text("/pkg/*\n  Cache-Control: public, max-age=3600\n/models/*\n  Cache-Control: public, max-age=86400\n", encoding="utf-8")
    total = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    wasm_size = (DIST / "pkg" / "nerve_bg.wasm").stat().st_size
    print(f"site/dist: {len(specs)} specs, {len(machines)} machines, {len(models)} networks, wasm {wasm_size / 1e6:.1f} MB, total {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
