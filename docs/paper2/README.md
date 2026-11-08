# Paper 2: Machines Without Code

`main.tex` + `sections/*.tex` + `refs.bib` + `neurips_2025.sty` is the manuscript, Part II of `docs/paper/`.
`OUTLINE.md` is the outline it was written from. `latexmk -pdf main.tex` (or `make`) builds `main.pdf`.

Every number in Section 2 is generated: `make_tables.py` reads `experiments/round2/results.jsonl` (the
append-only log the round-2 scripts write) and emits `numbers.tex`, `table_round2_rows.tex` and `fig_cost.tex`.
Re-run it after any sweep and the paper updates itself. Sections 3–7 are a plan; their predictions are numbered
so the next paper can report against them.
