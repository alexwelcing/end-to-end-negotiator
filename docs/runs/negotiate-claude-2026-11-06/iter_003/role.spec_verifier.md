I didn't change the spec, and the check came back clean.

- **WHAT I CHANGED:** I added one line to `runs/negotiate-claude/evolution/learnings.md`. `python -m nerve check specs/negotiate.nerve` reports 0 problems, 216 states, 11 actions and no holes. The working copy `iter_003/spec.nerve` is byte-identical to `specs/negotiate.nerve`, with sha256 `eead25d351a4ffcfce1d311ef022f7a61578612f98d81f4cab92f5d71961f04f`.
- **WHY:** That is the verification in my charter. It confirms the spec is exactly the author's version.
- **WHAT I EXPECT:** The score should stay at about 0.999, since the spec is unchanged. Iteration 002 scored 1.0 on both seen and held-out. The only shortfall is the learned-only `accept` accuracy of 0.968, and I can't change that from the spec side.