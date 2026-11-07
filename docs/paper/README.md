# The paper

`main.tex` (+ `appendix.tex`, `refs.bib`, `neurips_2025.sty`) is the manuscript: nine pages of main text in
NeurIPS format plus references and appendix. `latexmk -pdf main.tex` builds `main.pdf`.

Every number in it is generated:

* `sweep.py` / `sweep2.py` — the CPU runs behind Table 1 (bottom), Table 2 and Figure 2: three seeds of the four
  hand-written finite specs, ablations (no factored head, JEPA only, weight decay), hold-out fractions, the
  continuous slider, and two circuit-on runs. Output: `results.jsonl` (one line per run) and `runs/` (not committed).
* `pulse.py` — patch size against learning rate on the negotiate network. Output: `pulse.jsonl`.
* `margins_negotiate_s0.json` — energy margins and spread on the negotiate seed-0 model (computed inline; the
  snippet is in `GUIDANCE.md`'s history, the numbers are in the paper's Section 6).
* `make_tables.py` — reads the three files above and writes `numbers.tex`, the table rows and the pgfplots
  figures that `main.tex` inputs. Re-run it after any sweep; the paper updates itself.

`GUIDANCE.md` is what writing the paper exposed in the branch and what to do next.

Reproduce one row from the repository root:

```
python -m nerve rollout specs/negotiate.nerve --log /tmp/neg.jsonl --episodes 40
python -m nerve train specs/negotiate.nerve --log /tmp/neg.jsonl --out /tmp/neg --epochs 300 --weight-decay 0.05 --no-circuit --seed 0 --log-every 1000
```
