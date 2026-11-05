"""EAT-style TEAM_CONFIG: the *only* thing the supervisor emits.

From Evolving Agent Teams: the supervisor "never writes the solution"; each
iteration it REFLECTs on the last team's output and EVOLVEs roles, prompts,
phases and ownership. Here ownership is over nerve pipeline stages and
hyper-parameters rather than files and GPUs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Role:
    name: str
    prompt: str                                  # what this agent is for, in its own words
    owns: list[str]                              # pipeline stages: spec | rollout | train | adversary | eval
    knobs: dict[str, Any] = field(default_factory=dict)   # the parameters it may change


@dataclass
class Phase:
    name: str
    roles: list[str]                             # run these roles (in parallel) in this phase
    gate: str | None = None                      # condition to continue, e.g. "spec.problems == 0"


@dataclass
class TeamConfig:
    iteration: int
    rationale: str                               # the REFLECT output, kept for the shared learnings
    roles: list[Role]
    phases: list[Phase]
    train: dict[str, Any] = field(default_factory=dict)      # TrainConfig overrides
    rollout: dict[str, Any] = field(default_factory=dict)    # rollout overrides

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "TeamConfig":
        d = json.loads(Path(path).read_text())
        return cls(iteration=d["iteration"], rationale=d["rationale"],
                   roles=[Role(**r) for r in d["roles"]], phases=[Phase(**p) for p in d["phases"]],
                   train=d.get("train", {}), rollout=d.get("rollout", {}))


DEFAULT_ROLES = [
    Role("spec_auditor", "Read the spec. Find invariants and ensures the author forgot, actions whose guards "
         "are never tested, and unreachable states. Propose spec patches; never change behaviour the author wrote.",
         owns=["spec"]),
    Role("coverage_hunter", "Drive the oracle so every (state, action) pair appears in the log, with more episodes "
         "where the previous eval missed.", owns=["rollout"], knobs={"episodes": 50, "horizon": 20}),
    Role("adversary", "Generate hard negatives: near-miss successor states that break exactly one invariant or "
         "one ensures clause. Pair every negative with its positive.", owns=["adversary"], knobs={"negatives": 3}),
    Role("trainer", "Train the JEPA energy from the log. Report held-out conformance, learned-only conformance, "
         "and whether holes are learned.", owns=["train"], knobs={"epochs": 60, "lr": 3e-3, "d": 64}),
    Role("energy_auditor", "Inspect the energy landscape: margin between the true successor and the runner-up on "
         "held-out pairs, calibration on violating futures, collapse symptoms in the encoder.", owns=["eval"]),
]

DEFAULT_PHASES = [
    Phase("audit", ["spec_auditor"], gate="spec.problems == 0"),
    Phase("data", ["coverage_hunter", "adversary"]),
    Phase("learn", ["trainer"]),
    Phase("judge", ["energy_auditor"]),
]


def initial_team() -> TeamConfig:
    return TeamConfig(0, "Initial team: audit the spec, cover the space, train, judge.",
                      list(DEFAULT_ROLES), list(DEFAULT_PHASES),
                      train={"epochs": 60, "lr": 3e-3, "d": 64, "hidden": 128},
                      rollout={"episodes": 50, "horizon": 20, "negatives": 3})
