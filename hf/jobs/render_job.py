# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "pyyaml", "zstandard", "huggingface_hub", "pillow"]
# ///
"""Neural field renderer training as a Hugging Face Job. Inputs under runs/<run>/in/: spec.nerve, images.npy,
meta.json, config.json. Outputs under runs/<run>/out/: field.pt, eval.json, history.jsonl."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--branch", default="nerve"); ap.add_argument("--git", default="https://github.com/alexwelcing/nerve.git")
    ap.add_argument("--local-src"); ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp()); src = Path(a.local_src) if a.local_src else work / "src"
    if not a.local_src:
        if len(a.branch) == 40 and all(c in "0123456789abcdef" for c in a.branch):
            # an exact commit: the launcher pins the code the job runs to the code it pushed
            src.mkdir(parents=True)
            for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", a.git], ["git", "fetch", "-q", "--depth", "1", "origin", a.branch], ["git", "checkout", "-q", "FETCH_HEAD"]):
                subprocess.run(cmd, cwd=src, check=True)
        else:
            subprocess.run(["git", "clone", "--depth", "1", "--branch", a.branch, a.git, str(src)], check=True)
    sys.path.insert(0, str(src))
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    inp = Path(snapshot_download(a.repo, repo_type="dataset", allow_patterns=[f"runs/{a.run}/in/*"], token=os.environ.get("HF_TOKEN"))) / "runs" / a.run / "in"
    cfg = json.loads((inp / "config.json").read_text())
    from nerve.spec import load_spec
    from nerve.render import RenderSet
    from nerve.render.train import train_render, RenderTrainConfig
    import torch
    spec = load_spec(inp / "spec.nerve"); rs = RenderSet(inp)
    print(f"[job] render spec={spec.name} images={rs.images.shape} cuda={torch.cuda.is_available()} cfg={cfg}", flush=True)
    out = work / "out"
    r = train_render(spec, rs, out, RenderTrainConfig(**cfg))
    print("[job] eval:", json.dumps(r), flush=True)
    if not a.no_upload:
        api.upload_folder(folder_path=str(out), repo_id=a.repo, repo_type="dataset", path_in_repo=f"runs/{a.run}/out",
                          commit_message=f"nerve render run {a.run}: held-out PSNR {r['heldout']['psnr_mean']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
