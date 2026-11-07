# What the paper exposed, and what to send back to the `nerve` branch

Written 2026-11-07 while turning the branch into a nine-page paper. Everything below was found by reading the
code, re-running it on a two-core CPU, and trying to write down what it guarantees. The first list is already
done on the `paper-fixes` branch (a patch is attached); the second is what still needs a human or a GPU; the
third is the next step.

## 1. Done on `paper-fixes` (apply the patch, run `pytest`, push)

| change | why |
|---|---|
| `nerve/model/energy.py`: `circuit_weight()` raises the circuit's weight per query to `n_fields * spread(learned) + 1`, floor 1000. `edge/src/nn.ts` does the same. | The README said the circuit "wins where it speaks". It did not, in general: the circuit's energy gap is `1/n` (Hamming distance over `n` fields), so it outvotes the learned term only if `1000/n` exceeds the learned term's spread over candidates. On the trained negotiate network that spread reaches 1,665 against a bound of 200. The trained models happened to agree with the oracle, so `all = 1.0` held empirically, but a corrupt patch or a half-trained checkpoint could have moved a compiled transition. Now the guarantee is unconditional (Theorem 1 / Corollary 1 in the paper). |
| `tests/test_circuit_dominates.py` | Random and deliberately hostile learned weights (×50) plus the circuit must still match the oracle on every compiled cell of counter, slider and negotiate; hole actions must still be left to the learned term. Passes; `tests/test_workerd.py` parity still passes with the TS change. |
| `.gitignore`: `runs/` → `/runs/` | The old pattern matched *any* directory named `runs`, so `docs/runs/negotiate-claude-2026-11-06/` — the whole record of the $14.23 evolution run that README and ROADMAP cite — was never committed. Commit that directory now. |
| Score formula in `SUPERVISOR_PROMPT` and `harness/eat/SKILL.md` | They said `0.5·heldout + 0.3·all + 0.2·(1−viol)`. The eval and the role prompt say `0.4/0.2/0.2/0.2(learned-only)`. The supervisor was optimising a formula the harness does not compute. |
| `TrainConfig.holdout_salt`, `--holdout-salt`, `evaluate(..., salt=)` | The held-out set was fixed by a hard-coded salt, so three seeds measured init/shuffle variance only, never split variance. Vary the salt with the seed in the next sweep. |
| `--log-every` on `nerve train` | The every-10-epochs eval dominated CPU time: slider took 774 s with it and 57 s without. Raise it for sweeps; keep 10 when you want curves. |
| `nerve/sync/pulse.py` | Removed a dead `if False else` expression in `apply_patch`. |
| `tests/test_workerd.py` | The session-persistence assertion used a fixed session id, so the second run on a machine failed (`edge/.do-storage` had kept `t1` at count 2 — the feature working against the test). Now a fresh id per run. |
| `nerve/spec/drag.py`, `nerve drag`, `tests/test_drag.py`, `render:` blocks and `grab:` in the four hand-written specs, `docs/dragology.md` | `futures` was decoration (`Encoder.candidates()` returns the whole space). Dragology's operators are argmin / convex relaxation / descent on a *gap* that is an energy, so a drag is one minimisation of `E_spec + gap` over futures the spec derives. The prototype does `closest` with stickiness, `between`, `vary` through the spec's own clamps, and `far` with hysteresis, against the oracle, with a trained network's energies as optional tie-breaker. The note has the full design and five research questions (equivariance by type is the one to pursue). |
| README: PULSE claim, "bit for bit", results caveats, Paper section | See §2 for the facts behind each edit. The CPU re-run table in the README is filled from `docs/paper/results.jsonl`. |

## 2. Facts the paper had to state differently from the README

1. **Counter is not 1.000 on every seed.** Three CPU seeds: 33/34, 33/34, 34/34 held-out cells. The misses are
   `inc` at `count = 8` (predicted no change) and `arm` at `count = 9` — both next to the top of the range. Slider:
   213/213, 213/213, 212/213. Negotiate: 367/367 on all three seeds. The GPU table is one seed per row and should
   say so; it now does.
2. **PULSE ratio.** "~100× smaller" is the PULSE paper's number for 7B models. On the 140k-parameter negotiate
   network one epoch at lr 3e-6 leaves 92.2% of BF16 values unchanged and the patch is 30.6 kB against a 280 kB
   checkpoint, 9.2×. The property we actually rely on is bit-identity, and that holds. The README now says the
   measured number; `docs/paper/pulse.jsonl` has the sweep over learning rates.
3. **"The Worker computes the same energy bit for bit."** The *weights* are bit-identical (digest). The energies are
   not: `nn.ts` uses the tanh approximation of GELU and PyTorch's `nn.GELU()` is exact, which is why
   `test_workerd.py` compares energies to 0.05. Decide one of: (a) `nn.GELU(approximate="tanh")` in Python — same
   function family on both sides, but every existing checkpoint was trained and scored with exact GELU, so re-score
   them; or (b) an erf-based GELU in TS. Either way, tighten the parity tolerance to ~1e-4 afterwards.
4. **The three screenshot specs are not in the repo.** `docs/play/*.json` has their exported networks, and
   `intake/samples/` has the screenshots, but `specs/` has no `push_notifications_toggle.nerve`,
   `guest_stepper.nerve` or `volume_slider.nerve`, and the intake transcripts (rounds, cost) are not recorded. The
   paper marks these rows as "repository record" and cannot show their held-out counts. Commit the specs (e.g.
   `specs/intake/`), the `nerve intake` JSON output per spec, and add them to `tests/test_spec.py::SPECS`.
5. **The GPU rows have no committed configs.** `AlexWelcing/nerve-runs` is private. Commit each row's
   `eval.json`, `train_config.json` and `history.jsonl` under `docs/runs/results-2026-11-06/<spec>/`, and have
   `hf/jobs/train_job.py` write the git commit hash into `eval.json` (today a job clones the branch head, so a run's
   code version is not pinned — roadmap item 33).
6. **Toggle's "held-out 1.000"** is all two cells; specs under 20 cells get no hold-out (`effective_holdout`).
   Marked in both README and paper.
7. **Continuous fields use the JEPA term only.** `_continuous()` descends `energy_grad` = JEPA error; the factored
   head is not part of continuous inference. Fine, but say so (the paper does).
8. **NeurIPS 2026 is over.** Decisions are out (accepted papers are already on arXiv with that line). The nine-page
   NeurIPS format is right for NeurIPS 2027 (~May 2027). The nearer targets are ICML 2027 (abstract ~Jan 16,
   full paper ~Jan 22, 2027 AoE; 8 pages — cut §7's detail into the appendix) and workshops in between. ICLR 2027's
   deadline (late Sept 2026) has passed.

8b. **The continuous slider row does not reproduce from committed inputs.** GPU record 0.998 within tolerance;
   CPU re-run with the same `TrainConfig` (200 epochs, d=128, wd 0.05) but the default 40-episode rollout log:
   0.915, with 8.1% of predictions violating the clamp/guard. The record's log (how many episodes, which seed) is
   not committed. Commit it with the row, or regenerate the row from the default log so the README number is the
   reproducible one.
9. **The evolving team tunes on the test cells.** The supervisor maximises the official score, and the score's
   held-out term is computed on the same hash split the paper reports. In an evolution run those cells are a
   validation set, and the reported held-out number is optimistic. Use `--holdout-salt` to give the harness a
   validation salt and keep a sealed salt for reporting (the paper says this in its limitations).
10. **Counter's late regression.** Seed 0 was exact on all 34 held-out cells from epoch 10 to 100 and lost `inc` at
   8 at epoch 110; the final checkpoint is what is reported. A validation split (item 9) also gives you checkpoint
   selection for free.

## 3. The next step, in order (each with its acceptance test, ROADMAP style)

1. **Commit the evidence.** `docs/runs/negotiate-claude-2026-11-06/`, the three screenshot specs with their
   intake transcripts, the GPU rows' `eval.json`/`train_config.json`, and `docs/paper/`. *Done when:* a fresh clone
   can regenerate every number in the paper's Table 1 bottom half from committed files, and a reader can open the
   evolution record the README cites.
2. **Seeds and salts in CI.** A GitHub Action that runs toggle/counter/slider at three (seed, salt) pairs and
   negotiate at one, learned-only, `--log-every 1000` (≈10 min on two cores), and fails if any run drops more than
   two held-out cells below the committed `results.jsonl`. *Done when:* green on the branch and the README table is
   generated from the CI artifact.
3. **The two boundary cells.** Mine hard negatives from the model's own near-misses each epoch (ROADMAP 23) and
   add the boundary-adjacent cells to the adversary policy's preference. *Done when:* counter is 34/34 on all
   three seeds and all three salts.
4. **Exact Worker parity.** Pick GELU (item 2.3), tighten `test_workerd.py` to 1e-4, and extend it to negotiate.
5. **Cross-spec generalisation — the experiment that would change the paper's claim.** Today "learned-only 1.0"
   is generalisation *within* one interface (85% of its cells to the rest). Train one JEPA + factored head on
   {toggle, counter, slider, stepper, volume slider} with the spec text as a conditioning input (ROADMAP 69, 21),
   and report learned-only on negotiate and the push-notifications toggle with *no* training cells from them. Even
   a partial result (say, the ordinal rules transfer and the enum rules do not) is a finding; the current paper is a
   systems note with one modelling idea. *Done when:* a held-out-spec row exists in Table 1, whatever its value.
6. **Negotiation with language (ROADMAP 70).** Two LLM negotiators over the 2017 dataset's items, with the
   negotiate network as referee: every proposal is stepped through the network and scored for violations. Cheap,
   and it closes the loop with the repository's origin. *Done when:* a table of violation rate per negotiator model
   and a transcript where the referee refuses an illegal move.
7. **Hard refusal at the edge (ROADMAP 50)** now has its theory: with `circuit_weight()` the compiled cells are
   safe; refusal is for the learned cells. Return the state unchanged with alternatives when the learned margin is
   below τ. *Done when:* injected bad weights never move a session into a violating state (the test exists in
   spirit in `test_circuit_dominates.py`; extend it through the Worker).

## 4. For the submission itself

- The manuscript is `docs/paper/main.tex` (NeurIPS 2025 style file, `preprint` option; drop the option for the
  anonymous version — it is the same file the 2026 CfP used apart from the year strings, and the 2027 file will
  differ only there too). Compile with `latexmk -pdf main.tex`; `make_tables.py` regenerates every number from
  `results.jsonl`, `pulse.jsonl` and `margins_negotiate_s0.json`, so re-running the sweep updates the paper.
- Authorship: the LLM-usage paragraph at the end of the paper says that Claude wrote the branch and the manuscript
  as an agent under your direction and that the method itself uses Claude Code (intake, the evolving team). NeurIPS
  and ICML both require that disclosure and neither allows a model as an author. Reword to taste, do not remove.
- Before submitting, re-run `sweep2.py` with `holdout_salt` varied (`--holdout-salt s1`, `s2`) so Table 1 reports
  split variance too, and add one GPU seed sweep for the three screenshot specs once their specs are committed.
