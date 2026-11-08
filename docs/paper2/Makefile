# `make` builds main.pdf with latexmk when present, else with tectonic (https://tectonic-typesetting.github.io;
# on this machine: C:\Users\alexw\tools\tectonic\tectonic.exe, which `make TECTONIC=<path>` can override).
TECTONIC ?= tectonic
GENERATED = numbers.tex table_round2_rows.tex fig_cost.tex

all: main.pdf

$(GENERATED): make_tables.py ../../experiments/round2/results.jsonl
	python3 make_tables.py

main.pdf: main.tex sections/*.tex refs.bib $(GENERATED)
	@if command -v latexmk >/dev/null 2>&1; then latexmk -pdf -interaction=nonstopmode main.tex; \
	else $(TECTONIC) --keep-logs main.tex; fi

clean:
	latexmk -C 2>/dev/null || true; rm -f $(GENERATED) main.pdf main.log main.xdv main.aux main.bbl main.blg
