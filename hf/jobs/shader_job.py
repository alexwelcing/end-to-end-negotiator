# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "safetensors", "huggingface_hub"]
# ///
"""Shader training as a Hugging Face Job. Inputs under runs/<run>/in/: meta.json, rasters.f32, x.f32 (from
`nerve rasters`) and config.json; outputs under runs/<run>/out/: shader.json + shader.safetensors, which the
Rust engine loads (`nerve serve --shaders`, `nerve eval-shader`)."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--branch", default="nerve"); ap.add_argument("--git", default="https://github.com/alexwelcing/end-to-end-negotiator.git")
    ap.add_argument("--local-src"); ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp()); src = Path(a.local_src) if a.local_src else work / "src"
    if not a.local_src:
        subprocess.run(["git", "clone", "--depth", "1", "--branch", a.branch, a.git, str(src)], check=True)
    sys.path.insert(0, str(src))
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    inp = Path(snapshot_download(a.repo, repo_type="dataset", allow_patterns=[f"runs/{a.run}/in/*"], token=os.environ.get("HF_TOKEN"))) / "runs" / a.run / "in"
    cfg = json.loads((inp / "config.json").read_text())
    from nerve.render.shader import Rasters, ShaderTrain, train_shader
    import torch
    rs = Rasters(inp)
    arch = {k: cfg.pop(k) for k in ("kind", "bases", "channels", "cond", "hidden", "layers", "octaves") if k in cfg}
    if "bases" in arch:
        arch["bases"] = tuple(arch["bases"])
    spec_name = cfg.pop("spec_name")
    print(f"[job] shader spec={spec_name} rasters={rs.images.shape} cuda={torch.cuda.is_available()} arch={arch} train={cfg}", flush=True)
    out = work / "out"
    r = train_shader(spec_name, rs, rs.config(**arch), ShaderTrain(**cfg), out)
    print("[job] eval:", json.dumps(r), flush=True)
    if not a.no_upload:
        api.upload_folder(folder_path=str(out), repo_id=a.repo, repo_type="dataset", path_in_repo=f"runs/{a.run}/out",
                          commit_message=f"shader {spec_name}: mae {r['mae']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
