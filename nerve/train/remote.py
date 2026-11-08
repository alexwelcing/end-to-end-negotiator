"""`nerve train --remote hf`: upload inputs, run hf/jobs/train_job.py as a Hugging Face Job on a GPU, download outputs.

Needs HF_TOKEN in the environment (write access to the runs dataset repo, and the `jobs` scope). Nothing here is
mocked: without a token this returns an error that says so.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .loop import TrainConfig

JOB_SCRIPT = "https://raw.githubusercontent.com/alexwelcing/nerve/nerve/hf/jobs/train_job.py"
DEFAULT_REPO = "AlexWelcing/nerve-runs"


def wait_and_fetch(job_id: str, run: str, out_dir: str, repo: str = DEFAULT_REPO, timeout_s: int = 3 * 3600,
                   flavor: str = "", t0: float | None = None) -> dict[str, Any]:
    """Wait for a Job to finish and download runs/<run>/out into out_dir. Also used to attach to a running job."""
    token = os.environ.get("HF_TOKEN")
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)
    t0 = t0 or time.time()
    stage = ""
    while time.time() - t0 < timeout_s:
        st = api.inspect_job(job_id=job_id)
        stage = getattr(getattr(st, "status", None), "stage", None) or str(getattr(st, "status", ""))
        if stage in ("COMPLETED", "ERROR", "CANCELED"):
            break
        time.sleep(15)
    else:
        return {"error": "timed out waiting for job", "job": job_id}
    if stage != "COMPLETED":
        logs = ""
        try:
            logs = "\n".join(list(api.fetch_job_logs(job_id=job_id))[-30:])
        except Exception:  # noqa: BLE001
            pass
        return {"error": f"job ended with {stage}", "job": job_id, "logs": logs[-3000:]}
    local = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[f"runs/{run}/out/*"], token=token)) / "runs" / run / "out"
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for f in local.iterdir():
        (out / f.name).write_bytes(f.read_bytes())
    ev = json.loads((out / "eval.json").read_text())
    return {"job": job_id, "run": run, "flavor": flavor, "score": ev["score"], "seconds": ev["seconds"], "out": str(out),
            "heldout": ev["heldout"]["exact"], "learned_only": ev["learned_only"]["exact"], "minutes_wall": round((time.time() - t0) / 60, 1)}


def train_remote(spec_path: str, log_path: str, out_dir: str, cfg: TrainConfig, flavor: str = "a10g-small",
                 repo: str = DEFAULT_REPO, timeout_s: int = 3 * 3600) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"error": "HF_TOKEN is not set in this environment; add it as a network secret / env var and start a new session"}
    try:
        from huggingface_hub import HfApi, run_uv_job, snapshot_download
    except ImportError:
        return {"error": "pip install huggingface_hub"}
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    from ..spec import load_spec
    name = load_spec(spec_path).name
    run = f"{name}/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}"
    c = asdict(cfg); c.pop("init", None); c["device"] = "auto"
    mb = Path(log_path).stat().st_size / 1e6
    if mb > 300:
        return {"error": f"episode log is {mb:.0f} MB; refusing to upload (cap 300 MB). Regenerate with sane rollout knobs."}
    for p, dest in ((spec_path, "spec.nerve"), (log_path, "episodes.v1.jsonl")):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"runs/{run}/in/{dest}", repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=json.dumps(c).encode(), path_in_repo=f"runs/{run}/in/config.json", repo_id=repo, repo_type="dataset")
    job = run_uv_job(JOB_SCRIPT, script_args=["--repo", repo, "--run", run], flavor=flavor, secrets={"HF_TOKEN": token}, token=token)
    print(f"[remote] job {job.id} on {flavor}: {getattr(job, 'url', '')}  (run {run})", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "job.json").write_text(json.dumps({"job": job.id, "run": run, "flavor": flavor, "repo": repo}))
    return wait_and_fetch(job.id, run, out_dir, repo, timeout_s, flavor, t0=time.time())
