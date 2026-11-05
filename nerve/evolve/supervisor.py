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
import re
import shutil
import subprocess
import time
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

REFLECT in one paragraph (put it in "rationale"), then EVOLVE. You may: add/remove/rename roles, rewrite prompts,
change knobs (rollout: episodes, horizon, negatives; train: epochs, lr, d, hidden), reorder phases.
Output JSON with keys: rationale, roles[{{name,prompt,owns,knobs}}], phases[{{name,roles,gate}}], train{{}}, rollout{{}}.
"""


class Supervisor:
    def __init__(self, spec_path: str, workdir: str, budget_usd: float = 5.0, iterations: int = 4, mode: str = "offline",
                 model: str = "claude-fable-5-1"):
        self.spec_path = spec_path
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

    def _claude_supervisor(self, it: int, hist: list[dict[str, Any]]) -> TeamConfig:
        hist_txt = "\n".join(json.dumps({k: v for k, v in h.items() if k != "team"}) for h in hist[-5:])
        prompt = SUPERVISOR_PROMPT.format(spec_path=self.spec_path, history=hist_txt, learnings=self.learnings_path.read_text()[-4000:])
        r = subprocess.run(["claude", "-p", prompt, "--model", self.model, "--output-format", "json"],
                           capture_output=True, text=True, timeout=600, cwd=str(self.work))
        if r.returncode != 0:
            raise RuntimeError(r.stderr[:500])
        text = json.loads(r.stdout).get("result", r.stdout)
        m = re.search(r"\{.*\}", text, re.S)
        d = json.loads(m.group(0) if m else text)
        return TeamConfig(it, d.get("rationale", ""), [Role(**x) for x in d["roles"]], [Phase(**x) for x in d["phases"]],
                          d.get("train", {}), d.get("rollout", {}))

    # ---- harness: run phases, official eval -----------------------------
    def run_team(self, tc: TeamConfig, it_dir: Path) -> dict[str, Any]:
        t0 = time.time()
        log = EpisodeLog(it_dir / "episodes.v1.jsonl")
        cost = 0.0
        for ph in tc.phases:
            owned = {o for rn in ph.roles for r in tc.roles if r.name == rn for o in r.owns}
            if "spec" in owned:
                probs = Oracle(self.spec).self_check()
                (it_dir / "spec_audit.json").write_text(json.dumps({"problems": probs}, indent=1))
                if ph.gate == "spec.problems == 0" and probs:
                    return {"error": "spec failed audit", "problems": probs[:10], "cost_usd": cost}
            if "rollout" in owned or "adversary" in owned:
                for pol in ("coverage", "random", "adversary"):
                    rollout_dataset(self.spec, log, episodes=tc.rollout.get("episodes", 50), horizon=tc.rollout.get("horizon", 20),
                                    policy=pol, negatives=tc.rollout.get("negatives", 3), seed=tc.iteration * 7 + hash(pol) % 100)
            if "train" in owned:
                knobs = {k: v for k, v in tc.train.items() if k in TrainConfig.__dataclass_fields__}
                for r in tc.roles:
                    if r.name in ph.roles:
                        knobs.update({k: v for k, v in r.knobs.items() if k in TrainConfig.__dataclass_fields__})
                ev = train(self.spec, log, it_dir / "run", TrainConfig(**knobs, seed=tc.iteration))
                (it_dir / "eval.json").write_text(json.dumps(ev, indent=1))
            # eval roles read eval.json; in claude mode they are real agents, here the numbers speak
        # cost: compute time only in offline mode; claude mode adds token cost reported by the CLI
        cost += (time.time() - t0) / 3600 * 0.5     # nominal $0.50/h CPU
        ev = json.loads((it_dir / "eval.json").read_text())
        return {"eval": ev, "cost_usd": round(cost, 4), "seconds": round(time.time() - t0, 1)}

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
