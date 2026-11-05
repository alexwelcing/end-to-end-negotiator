"""Append-only JSONL episode log.

Follows the DeepSeek-harness rule "model-visible means logged": every
transition the network is ever trained on is reconstructable from this file.
Negative samples (spec-violating futures) are logged too, so a training run
is a pure function of the log.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

LOG_VERSION = 1


@dataclass
class Step:
    state: dict[str, Any]
    action: list                       # [name, params]
    next: dict[str, Any]
    ok: bool = True                    # True: oracle transition. False: negative sample.
    violations: list[str] = field(default_factory=list)
    source: str = "oracle"             # oracle | agent | hole-example | adversary
    episode: int = 0
    t: float = field(default_factory=time.time)


class EpisodeLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w") as fh:
                fh.write(json.dumps({"kind": "header", "version": LOG_VERSION, "t": time.time()}) + "\n")

    def append(self, step: Step) -> None:
        with open(self.path, "a") as fh:
            d = asdict(step)
            d["kind"] = "step"
            fh.write(json.dumps(d, separators=(",", ":")) + "\n")

    def extend(self, steps: list[Step]) -> None:
        with open(self.path, "a") as fh:
            for s in steps:
                d = asdict(s)
                d["kind"] = "step"
                fh.write(json.dumps(d, separators=(",", ":")) + "\n")

    def __iter__(self) -> Iterator[Step]:
        with open(self.path) as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("kind") != "step":
                    continue
                d.pop("kind")
                yield Step(**d)

    def steps(self) -> list[Step]:
        return list(iter(self))
