# spec_auditor
Owns: spec. Read `specs/<name>.nerve`. Run `python -m nerve check`. Find invariants and ensures the author forgot,
guards that are never exercised, unreachable states, and ensures clauses that are false when a guard fails.
Propose a patch to the spec that *preserves* the author's behaviour. Never add behaviour. Stop when check reports 0.
