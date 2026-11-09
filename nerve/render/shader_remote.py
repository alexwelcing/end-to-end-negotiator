"""Train a shader on a Hugging Face GPU Job from a `nerve rasters` dump; fetch shader.json + shader.safetensors."""
from __future__ import annotations

import json, os, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..train.remote import DEFAULT_REPO, upload_inputs
from .remote import _wait
from .shader import ShaderTrain

JOB_SCRIPT_PATH = "hf/jobs/shader_job.py"


def job_script_url() -> str:
    """The job script at the exact commit that is pushed: raw.githubusercontent.com caches the branch URL for
    minutes, which once ran a stale script against a new config."""
    import subprocess
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=__import__("pathlib").Path(__file__).resolve().parents[2]).stdout.strip()
    return f"https://raw.githubusercontent.com/alexwelcing/nerve/{sha}/{JOB_SCRIPT_PATH}"


def shader_remote(spec_name: str, rasters_dir: str, out_dir: str, train: ShaderTrain, arch: dict[str, Any] | None = None,
                  flavor: str = "a10g-small", repo: str = DEFAULT_REPO) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"error": "HF_TOKEN is not set"}
    from huggingface_hub import HfApi, run_uv_job
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    run = f"shader-{spec_name}/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}"
    c = asdict(train); c["device"] = "auto"; c["spec_name"] = spec_name; c.update(arch or {})
    files = {f: f"{rasters_dir}/{f}" for f in ("meta.json", "rasters.f32", "x.f32")}
    files["config.json"] = json.dumps(c).encode()
    upload_inputs(api, repo, run, files)
    job = run_uv_job(job_script_url(), script_args=["--repo", repo, "--run", run, "--branch", __import__("subprocess").run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()], flavor=flavor, secrets={"HF_TOKEN": token}, token=token)
    print(f"[remote] shader job {job.id} on {flavor} (run {run})", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "job.json").write_text(json.dumps({"job": job.id, "run": run, "flavor": flavor, "repo": repo}))
    r = _wait(api, job.id, run, out_dir, repo, token)
    return r
