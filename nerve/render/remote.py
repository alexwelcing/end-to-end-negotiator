"""`nerve render-train --remote hf`: upload the render set + spec, run hf/jobs/render_job.py on a GPU, fetch outputs."""
from __future__ import annotations

import json, os, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..train.remote import DEFAULT_REPO
from .train import RenderTrainConfig

JOB_SCRIPT = "https://raw.githubusercontent.com/alexwelcing/end-to-end-negotiator/nerve/hf/jobs/render_job.py"


def render_remote(spec_path: str, render_dir: str, out_dir: str, cfg: RenderTrainConfig, flavor: str = "a10g-small",
                  repo: str = DEFAULT_REPO) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"error": "HF_TOKEN is not set"}
    from huggingface_hub import HfApi, run_uv_job
    from ..spec import load_spec
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    name = load_spec(spec_path).name
    run = f"render-{name}/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}"
    c = asdict(cfg); c["device"] = "auto"
    for p, dest in ((spec_path, "spec.nerve"), (f"{render_dir}/images.npy", "images.npy"), (f"{render_dir}/meta.json", "meta.json")):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"runs/{run}/in/{dest}", repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=json.dumps(c).encode(), path_in_repo=f"runs/{run}/in/config.json", repo_id=repo, repo_type="dataset")
    job = run_uv_job(JOB_SCRIPT, script_args=["--repo", repo, "--run", run], flavor=flavor, secrets={"HF_TOKEN": token}, token=token)
    print(f"[remote] render job {job.id} on {flavor} (run {run})", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "job.json").write_text(json.dumps({"job": job.id, "run": run, "flavor": flavor, "repo": repo}))
    return _wait(api, job.id, run, out_dir, repo, token)


def _wait(api, job_id, run, out_dir, repo, token, timeout_s=3 * 3600):
    from huggingface_hub import snapshot_download
    t0 = time.time(); stage = ""
    while time.time() - t0 < timeout_s:
        st = api.inspect_job(job_id=job_id); stage = getattr(getattr(st, "status", None), "stage", "")
        if stage in ("COMPLETED", "ERROR", "CANCELED"):
            break
        time.sleep(15)
    if stage != "COMPLETED":
        logs = ""
        try:
            logs = "\n".join(list(api.fetch_job_logs(job_id=job_id))[-30:])
        except Exception:  # noqa: BLE001
            pass
        return {"error": f"job ended with {stage}", "job": job_id, "logs": logs[-3000:]}
    local = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[f"runs/{run}/out/*"], token=token)) / "runs" / run / "out"
    out = Path(out_dir)
    for f in local.iterdir():
        (out / f.name).write_bytes(f.read_bytes())
    if (out / "eval.json").exists():
        ev = json.loads((out / "eval.json").read_text())
        return {"job": job_id, "run": run, "out": str(out), **{k: ev[k] for k in ("train", "heldout", "seconds", "params")}, "minutes_wall": round((time.time() - t0) / 60, 1)}
    meta = json.loads((out / "shader.json").read_text())
    return {"job": job_id, "run": run, "out": str(out), "eval": meta.get("eval"), "minutes_wall": round((time.time() - t0) / 60, 1)}
