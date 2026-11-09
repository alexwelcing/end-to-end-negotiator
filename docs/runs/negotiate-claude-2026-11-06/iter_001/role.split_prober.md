I couldn't score the probe set. The rollout ran, but the shell refused to run the scoring script and asked for approval each time, so there are no probe numbers.

**WHAT I CHANGED:** I rolled out a fresh probe log of 8000 positives. It used seed 98765, 400 random episodes and horizon 20, and I wrote it to `probe.jsonl` in my scratchpad. I also wrote `probe_eval.py` in the same folder. I did not touch `episodes.v1.jsonl`, `spec.nerve` or `learnings.md`, and I wrote no knobs file.

**WHY:** `python -m nerve eval` can't score the probe log directly. Its `--holdout` flag is a float fraction, and it scores the whole finite state space rather than a log. So I wrote `probe_eval.py` to load `iter_001/run/model.pt` and compare it with the oracle on the unique (state, action) pairs from the probe log. That is the step the shell blocked. The `model.pt` it loads may be left over from the harness's last training run, since I can't tell whether it matches the current iteration.

**WHAT I EXPECT:** The official numbers from `iter_000` were 1.0 on every split, so I expect the probe to score at or above 0.98. That is an expectation, not a measurement. To get the real numbers, approve this command (from the repo root, `/home/user/end-to-end-negotiator`):

`python /tmp/claude-0/-home-user-end-to-end-negotiator/15e6ecdd-669e-4270-ae43-277cd3ec7a35/scratchpad/probe_eval.py /tmp/claude-0/-home-user-end-to-end-negotiator/15e6ecdd-669e-4270-ae43-277cd3ec7a35/scratchpad/probe.jsonl`

It prints exact, violation_rate per action and any failing pairs. If it scores below 0.98, I'll list the failing pairs in `evolution/learnings.md`.