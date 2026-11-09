# Learnings

- iter 0: new best 1.0000 with Initial team: audit the spec, cover the space, train, judge.
- iter 1 spec_verifier: `nerve check` = 0 problems (216 states, 11 actions, no holes). Spec NOT edited.
  * DISCREPANCY: the invariants exclude NO states. Both hold on all 216 states (field ranges already enforce them). coverage()'s own filter, run locally, gives 0 invariant-violating states, so valid=216, not 164. iter_000 eval.json's 164/216 does not reproduce with the current code and spec.
  * Real reachability (BFS from initial over all 11 actions): 83 states reachable, 133 unreachable.
    - rounds=0: only (a,0,0,open) and its walk successor (a,0,0,no_deal) are reachable. accept needs rounds>0, and propose always increments rounds. The other 52 rounds=0 states are unreachable (17 open + 17 no_deal + 18 deal). These are likely the "52" the charter means.
    - rounds=1..3: propose flips turn every round, so turn is fixed by parity (b at odd rounds, a at even rounds). The 81 wrong-parity states are unreachable.
  * Coverage gap for coverage_hunter: none among reachable states (all 83 appear; eval fraction 1.0 on 1804 pairs). Unreachable grid states are never logged as positives. This is harmless for score unless heldout samples them.
- iter_003 spec_verifier: nerve check OK (0 problems, 216 states, 11 actions, no holes); spec sha256 eead25d351a4ffcfce1d311ef022f7a61578612f98d81f4cab92f5d71961f04f identical to working copy; unchanged.
- iter_004 spec_verifier: nerve check OK (0 problems, 216 states, 11 actions, no holes); sha256 eead25d351a4ffcfce1d311ef022f7a61578612f98d81f4cab92f5d71961f04f confirmed; spec unchanged.
- iter_005 spec_verifier: nerve check OK (0 problems, 216 states, 11 actions, no holes); sha256 eead25d351a4ffcfce1d311ef022f7a61578612f98d81f4cab92f5d71961f04f confirmed; spec unchanged.
- iter_006 spec_verifier: nerve check OK (0 problems, 216 states, 11 actions, no holes); sha256 eead25d351a4ffcfce1d311ef022f7a61578612f98d81f4cab92f5d71961f04f confirmed; spec unchanged.
- iter_007 spec_verifier: nerve check OK (0 problems, 216 states, 11 actions, no holes); sha256 eead25d351a4ffcfce1d311ef022f7a61578612f98d81f4cab92f5d71961f04f confirmed; spec unchanged.
- iter_006 margin_checker: UNAVAILABLE - python margin script (NerveEnergy.load(spec, runs/negotiate-claude/iter_006/run/model.pt); jepa.energy_all per pair) -> run denied by sandbox permission layer, no output produced; recommend permitting that python run (the API exists: NerveEnergy.jepa.energy_all returns learned-only energies over all 216 candidate successors)
- iter_007 margin_checker: UNAVAILABLE via nerve step because it runs one pair per process (~1.7s), prints only the combined circuit+JEPA energy of the chosen successor plus 2 nearest alternatives (no learned-only flag, no full candidate set), and launching a sampled sweep script was denied by the sandbox; fallback script at runs/negotiate-claude/iter_007/run/margin_check.py, needs permission for: python runs/negotiate-claude/iter_007/run/margin_check.py runs/negotiate-claude/iter_006/run/model.pt
- iter_008 spec_verifier: nerve check 0 problems, 216 states, 11 actions, holes 0; sha256 eead25d351a4ffcfce1d311ef022f7a61578612f98d81f4cab92f5d71961f04f; spec unchanged
- iter_009 spec_verifier: nerve check 0 problems, 216 states, 11 actions, holes 0; sha256 eead25d351a4ffcfce1d311ef022f7a61578612f98d81f4cab92f5d71961f04f; spec unchanged
