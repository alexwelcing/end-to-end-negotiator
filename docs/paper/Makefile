all: main.pdf
numbers.tex table_main_rows.tex table_ablate_rows.tex table_all_rows.tex table_pulse_rows.tex fig_curves.tex fig_holdout.tex: make_tables.py results.jsonl pulse.jsonl margins_negotiate_s0.json
	python3 make_tables.py
main.pdf: main.tex appendix.tex refs.bib numbers.tex table_main_rows.tex table_ablate_rows.tex table_all_rows.tex table_pulse_rows.tex fig_curves.tex fig_holdout.tex
	latexmk -pdf -interaction=nonstopmode main.tex
clean:
	latexmk -C; rm -f numbers.tex table_*_rows.tex fig_*.tex
