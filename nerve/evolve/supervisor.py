"""The EAT loop:

    while budget remains:
        supervisor: REFLECT on last team's output -> EVOLVE roles/prompts/knobs -> emit TEAM_CONFIG
        harness:    run phases -> official eval -> record cost and score

Two supervisors are provided. ``offline`` is a deterministic rule-based
reflector so the loop runs with no model at all (and in CI). ``claude`` asks a
Claude Code process to do REFLECT/EVOLVE and parse its TEAM_CONFIG, the way
EAT's supervisor is itself a coding agent. The harness side is identical.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..data import EpisodeLog, rollout_dataset
from ..spec import Oracle, load_spec
from ..train import TrainConfig, train
from .team_config import Phase, Role, TeamConfig, initial_team

SUPERVISOR_PROMPT = """You are the supervisor of an Evolving Agent Team (EAT). You never write the solution.
You read the previous team's results and emit exactly one TEAM_CONFIG as JSON, nothing else.

Goal: maximise the official score of a Nerve network trained to behave like the spec at {spec_path}.
Score = 0.5*heldout_exact + 0.3*all_exact + 0.2*(1-violation_rate), blended 80/20 with hole-example accuracy.

Previous iterations (newest last):
{history}

Shared learnings:
{learnings}

Role library (you may reuse, rewrite, or invent roles; a role's `owns` is a subset of
[spec, rollout, adversary, train, eval] and decides which pipeline verbs it may use):
{roles}

Each role runs as its own Claude Code process with the spec, the episode log, eval.json and the pipeline verbs
`python -m nerve check|rollout|train|eval|step`. Roles that own `spec` edit a copy of the spec (never the author's
behaviour, only missing guards/invariants/ensures). Roles that own `train` write knobs.json (TrainConfig overrides);
the harness trains. Roles that own `eval` write to learnings.md.

REFLECT in one paragraph (put it in "rationale"), then EVOLVE. You may: add/remove/rename roles, rewrite prompts,
change knobs (rollout: episodes, horizon, negatives; train: epochs, lr, d, hidden, hole_weight), reorder phases.
Create a role when a blocker has no owner. Output JSON only, with keys:
rationale, roles[{{name,prompt,owns,knobs}}], phases[{{name,roles,gate}}], train{{}}, rollout{{}}.
"""

ROLE_PROMPT = """You are the role `{name}` on an Evolving Agent Team building a Nerve network for the spec below.
Your charter: {prompt}
You own these pipeline stages: {owns}. Knobs you may set: {knobs}.

Repository: {repo} (run commands from here; PYTHONPATH is set). Iteration directory: {it_dir}
- Spec (original, read-only): {spec_path}
- Spec working copy you may edit if you own `spec`: {spec_copy}
- Episode log (append-only; add data with `python -m nerve rollout {spec_copy} --log {log} ...`): {log}
- Previous eval (if any): {prev_eval}
- Shared learnings: {learnings}
- Knobs file you may write if you own `train` (JSON of TrainConfig overrides: epochs, lr, d, hidden, hole_weight): {knobs_file}

Official score = 0.4*heldout + 0.2*all + 0.2*(1-violations) + 0.2*learned_only (blended 80/20 with hole-example
accuracy when the spec has `learn` holes). Do only your charter, then finish with a 3-line report:
WHAT I CHANGED / WHY / WHAT I EXPECT. Never modify files outside the iteration directory except the episode log.
Never disable the compiled circuit. Never change the author's behaviour in the spec.

Spec:
```yaml
{spec_text}
```
Last eval.json:
{eval_text}
"""


REPO = Path(__file__).resolve().parents[2]
OWNABLE = {"spec", "rollout", "adversary", "train", "eval"}


class Supervisor:
    def __init__(self, spec_path: str, workdir: str, budget_usd: float = 5.0, iterations: int = 4, mode: str = "offline",
                 model: str = "claude-fable-5-1", role_model: str = "claude-sonnet-5-5"):
        self.spec_path = str(Path(spec_path).resolve())
        self.role_model = role_model
        self._pending_cost = 0.0
        self.spec = load_spec(spec_path)
        self.work = Path(workdir); self.work.mkdir(parents=True, exist_ok=True)
        self.budget = budget_usd
        self.iterations = iterations
        self.mode = mode
        self.model = model
        self.history_path = self.work / "evolution" / "history.jsonl"
        self.learnings_path = self.work / "evolution" / "learnings.md"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.learnings_path.exists():
            self.learnings_path.write_text("# Learnings\n\n")

    # ---- history ------------------------------------------------------
    def history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return [json.loads(l) for l in self.history_path.read_text().splitlines() if l.strip()]

    def _record(self, rec: dict[str, Any]) -> None:
        with open(self.history_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def _learn(self, text: str) -> None:
        with open(self.learnings_path, "a") as fh:
            fh.write(f"- {text}\n")

    # ---- REFLECT / EVOLVE ---------------------------------------------
    def reflect_evolve(self, it: int) -> TeamConfig:
        hist = self.history()
        if not hist:
            tc = initial_team(); tc.iteration = it
            return tc
        if self.mode == "claude" and shutil.which("claude"):
            try:
                return self._claude_supervisor(it, hist)
            except Exception as e:  # noqa: BLE001
                self._learn(f"iter {it}: claude supervisor failed ({e}); fell back to offline reflector")
        return self._offline_supervisor(it, hist)

    def _offline_supervisor(self, it: int, hist: list[dict[str, Any]]) -> TeamConfig:
        """Rule-based REFLECT/EVOLVE. Each rule is a learning EAT's supervisors wrote down in prose;
        we encode them so the loop is reproducible without a model."""
        last = hist[-1]; ev = last["eval"]; prev = TeamConfig.load(self.work / f"iter_{last['iteration']:03d}" / "team_config.json")
        train = dict(prev.train); roll = dict(prev.rollout); roles = list(prev.roles); why = []
        held, alle, lo = ev["heldout"]["exact"], ev["all"]["exact"], ev["learned_only"]["exact"]
        holes = ev.get("hole_examples", {}).get("exact", 1.0)
        if lo < 0.9:
            roll["episodes"] = int(roll.get("episodes", 50) * 1.5); train["epochs"] = int(train.get("epochs", 60) * 1.3)
            why.append(f"learned-only conformance {lo:.2f} < 0.9: more coverage and longer training")
        if held < alle - 0.02:
            train["d"] = min(int(train.get("d", 64) * 1.5), 256); roll["negatives"] = roll.get("negatives", 3) + 2
            why.append(f"held-out {held:.2f} trails seen {alle:.2f}: wider representation, more adversarial negatives")
        if holes < 1.0:
            train["epochs"] = int(train.get("epochs", 60) * 1.2); train["lr"] = train.get("lr", 3e-3) * 0.7
            if not any(r.name == "hole_tutor" for r in roles):
                roles.append(Role("hole_tutor", "Study the `learn` holes. Ensure their examples are oversampled and that "
                                  "the fallback is never confused with the examples.", owns=["train"], knobs={"hole_weight": 40}))
            why.append(f"hole examples {holes:.2f} < 1: added hole_tutor role, slower lr, longer")
        if ev["all"]["violation_rate"] > 0:
            roll["negatives"] = roll.get("negatives", 3) + 3
            why.append("violations > 0: push harder on spec-violating futures")
        if not why and lo < 0.99:
            train["epochs"] = int(train.get("epochs", 60) * 1.3); roll["episodes"] = int(roll.get("episodes", 50) * 1.3)
            why.append(f"circuit is exact but learned-only is {lo:.2f}: the network must learn what the compiler knows")
        if not why:
            train["lr"] = train.get("lr", 3e-3) * 0.5
            why.append("plateau: anneal lr to sharpen margins")
        tc = TeamConfig(it, "; ".join(why), roles, list(prev.phases), train, roll)
        return tc

    def _role_library(self) -> str:
        d = REPO / "harness" / "eat" / "roles"
        return "\n".join(f"### {p.stem}\n{p.read_text().strip()}" for p in sorted(d.glob("*.md"))) if d.exists() else ""

    def _claude(self, prompt: str, model: str, cwd: Path, allowed: list[str], timeout: int = 1500) -> tuple[str, float]:
        """One Claude Code process on the user's subscription. Returns (result text, reported cost)."""
        cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json"]
        if allowed:
            cmd += ["--allowedTools", *allowed]
        env = dict(os.environ, PYTHONPATH=str(REPO))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(cwd), env=env)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout)[:800])
        j = json.loads(r.stdout)
        return j.get("result", ""), float(j.get("total_cost_usd", 0) or 0)

    def _claude_supervisor(self, it: int, hist: list[dict[str, Any]]) -> TeamConfig:
        hist_txt = "\n".join(json.dumps({k: v for k, v in h.items() if k not in ("team", "roles_report")}) for h in hist[-6:])
        prompt = SUPERVISOR_PROMPT.format(spec_path=self.spec_path, history=hist_txt,
                                          learnings=self.learnings_path.read_text()[-6000:], roles=self._role_library())
        text, cost = self._claude(prompt, self.model, self.work, allowed=[], timeout=900)
        self._pending_cost += cost
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group(0) if m else text)
        roles = []
        for x in d["roles"]:
            owns = [o for o in x.get("owns", []) if o in OWNABLE]
            roles.append(Role(x["name"], x.get("prompt", ""), owns, x.get("knobs", {}) or {}))
        return TeamConfig(it, d.get("rationale", ""), roles, [Phase(x["name"], x.get("roles", []), x.get("gate")) for x in d["phases"]],
                          d.get("train", {}) or {}, d.get("rollout", {}) or {})

    # ---- roles as Claude Code processes (EAT: the team is coding agents) ------
    def _run_role(self, role: Role, tc: TeamConfig, it_dir: Path, log_path: Path) -> dict[str, Any]:
        prev_eval = self.work / f"iter_{tc.iteration - 1:03d}" / "eval.json"
        spec_copy = it_dir / "spec.nerve"
        prompt = ROLE_PROMPT.format(
            name=role.name, prompt=role.prompt, owns=role.owns, knobs=role.knobs or "{}", repo=REPO, it_dir=it_dir,
            spec_path=Path(self.spec_path).resolve(), spec_copy=spec_copy, log=log_path,
            prev_eval=prev_eval if prev_eval.exists() else "none", learnings=self.learnings_path,
            knobs_file=it_dir / f"knobs.{role.name}.json", spec_text=spec_copy.read_text(),
            eval_text=prev_eval.read_text()[:4000] if prev_eval.exists() else "none yet")
        allowed = ["Read", "Bash(python -m nerve:*)", "Bash(cat:*)", "Bash(ls:*)", "Bash(wc:*)", "Bash(head:*)", "Bash(tail:*)"]
        if "spec" in role.owns or "train" in role.owns or "eval" in role.owns:
            allowed += ["Edit", "Write"]
        t0 = time.time()
        try:
            text, cost = self._claude(prompt, self.role_model, REPO, allowed)
            ok = True
        except Exception as e:  # noqa: BLE001
            text, cost, ok = f"ROLE FAILED: {e}", 0.0, False
        (it_dir / f"role.{role.name}.md").write_text(text)
        return {"role": role.name, "ok": ok, "cost_usd": round(cost, 4), "seconds": round(time.time() - t0), "report": text[-600:]}

    # ---- harness: run phases, official eval -----------------------------
    def run_team(self, tc: TeamConfig, it_dir: Path) -> dict[str, Any]:
        t0 = time.time()
        log_path = it_dir / "episodes.v1.jsonl"
        log = EpisodeLog(log_path)
        spec_copy = it_dir / "spec.nerve"
        if not spec_copy.exists():
            shutil.copy(self.spec_path, spec_copy)
        spec = load_spec(spec_copy)
        cost = self._pending_cost; self._pending_cost = 0.0
        reports: list[dict[str, Any]] = []
        claude = self.mode == "claude" and shutil.which("claude")
        for ph in tc.phases:
            roles = [r for r in tc.roles if r.name in ph.roles]
            owned = {o for r in roles for o in r.owns}
            if claude and roles:
                # EAT: every role is a coding agent; a phase's roles run in parallel on the shared file system
                with ThreadPoolExecutor(max_workers=max(1, len(roles))) as ex:
                    for rep in ex.map(lambda r: self._run_role(r, tc, it_dir, log_path), roles):
                        reports.append(rep); cost += rep["cost_usd"]
                        print(f"    role {rep['role']:18s} ok={rep['ok']} ${rep['cost_usd']:.3f} {rep['seconds']}s")
                # a spec role may have edited the working copy: it must still pass check, else revert
                try:
                    cand = load_spec(spec_copy); probs = Oracle(cand).self_check()
                    if probs:
                        raise ValueError(probs[:5])
                    spec = cand
                except Exception as e:  # noqa: BLE001
                    self._learn(f"iter {tc.iteration}: spec edit rejected ({str(e)[:200]}); reverted to author's spec")
                    shutil.copy(self.spec_path, spec_copy); spec = load_spec(spec_copy)
                subprocess.run(["git", "checkout", "--", "specs"], cwd=str(REPO), capture_output=True)   # author's specs are sacred
            if "spec" in owned:
                probs = Oracle(spec).self_check()
                (it_dir / "spec_audit.json").write_text(json.dumps({"problems": probs}, indent=1))
                if ph.gate == "spec.problems == 0" and probs:
                    return {"error": "spec failed audit", "problems": probs[:10], "cost_usd": cost}
            if ("rollout" in owned or "adversary" in owned) and not (claude and any(log)):
                # offline mode (or roles that added nothing): the harness fills the log with the knobs
                for pol in ("coverage", "random", "adversary"):
                    rollout_dataset(spec, log, episodes=tc.rollout.get("episodes", 50), horizon=tc.rollout.get("horizon", 20),
                                    policy=pol, negatives=tc.rollout.get("negatives", 3), seed=tc.iteration * 7 + hash(pol) % 100)
            if "train" in owned:
                knobs = {k: v for k, v in tc.train.items() if k in TrainConfig.__dataclass_fields__}
                for r in roles:
                    knobs.update({k: v for k, v in r.knobs.items() if k in TrainConfig.__dataclass_fields__})
                    kf = it_dir / f"knobs.{r.name}.json"
                    if kf.exists():
                        try:
                            knobs.update({k: v for k, v in json.loads(kf.read_text()).items() if k in TrainConfig.__dataclass_fields__})
                        except json.JSONDecodeError:
                            pass
                if not any(log):
                    for pol in ("coverage", "random", "adversary"):
                        rollout_dataset(spec, log, episodes=tc.rollout.get("episodes", 50), policy=pol, seed=tc.iteration)
                knobs.pop("init", None); knobs["log_every"] = 10 ** 6
                ev = train(spec, log, it_dir / "run", TrainConfig(**knobs, seed=tc.iteration))
                (it_dir / "eval.json").write_text(json.dumps(ev, indent=1))
                (it_dir / "knobs.effective.json").write_text(json.dumps(knobs, indent=1))
        if not claude:
            cost += (time.time() - t0) / 3600 * 0.5     # nominal $0.50/h CPU; claude mode uses the CLI's reported cost
        if not (it_dir / "eval.json").exists():
            return {"error": "no train phase ran", "cost_usd": round(cost, 4)}
        ev = json.loads((it_dir / "eval.json").read_text())
        return {"eval": ev, "cost_usd": round(cost, 4), "seconds": round(time.time() - t0, 1), "roles_report": reports}

    # ---- the loop -------------------------------------------------------
    def run(self) -> dict[str, Any]:
        spent = sum(h.get("cost_usd", 0) for h in self.history())
        start = len(self.history())
        best = max((h["eval"]["score"] for h in self.history() if "eval" in h), default=0.0)
        for it in range(start, start + self.iterations):
            if spent >= self.budget:
                self._learn(f"budget exhausted at iteration {it}")
                break
            tc = self.reflect_evolve(it)
            it_dir = self.work / f"iter_{it:03d}"; it_dir.mkdir(parents=True, exist_ok=True)
            tc.dump(it_dir / "team_config.json")
            print(f"[iter {it}] TEAM_CONFIG roles={[r.name for r in tc.roles]} train={tc.train} rollout={tc.rollout}")
            print(f"[iter {it}] rationale: {tc.rationale}")
            res = self.run_team(tc, it_dir)
            rec = {"iteration": it, "team": [r.name for r in tc.roles], "rationale": tc.rationale, **res}
            if "roles_report" in rec:
                rec["roles_report"] = [{k: v for k, v in r.items() if k != "report"} for r in rec["roles_report"]]
            self._record(rec)
            spent += res.get("cost_usd", 0)
            if "eval" in res:
                sc = res["eval"]["score"]
                print(f"[iter {it}] score={sc:.4f} heldout={res['eval']['heldout']['exact']:.3f} "
                      f"learned-only={res['eval']['learned_only']['exact']:.3f} holes={res['eval'].get('hole_examples',{}).get('exact','-')} "
                      f"cost=${spent:.4f}")
                if sc > best:
                    best = sc
                    shutil.copy(it_dir / "run" / "model.pt", self.work / "best.pt")
                    self._learn(f"iter {it}: new best {sc:.4f} with {tc.rationale}")
        return {"best_score": best, "iterations": len(self.history()), "spent_usd": round(spent, 4),
                "best_model": str(self.work / "best.pt"), "history": str(self.history_path)}
