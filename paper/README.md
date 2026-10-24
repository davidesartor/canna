# Paper draft (ICML 2026 template)

For now the paper is its appendices: `main.tex` inputs `appendix/*.tex` and has no main text.

## Numbers and figures

Never type a number by hand. `scripts/numbers_<appendix>.py` writes `numbers/<appendix>.tex`,
one macro per number (`\ChartMcMedian`-style, prefixed by the appendix), all of which
`main.tex` inputs in its preamble, so the abstract and any appendix can use them; `scripts/fig_<name>.py` writes `figures/<name>.pdf`. The
scripts read only `reports/*/data` and the lisa configs (`canna.lisa.config.load`).

`scripts/make_figures.sh [--pdf] [script ...]` runs every script (or the named ones) and
recompiles `figures/src/architecture.tex` (xelatex, Inter from `fonts/`) when it changed. The
scripts import JAX, so run it in a CPU job, from the repo root:

```bash
sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal \
    paper/scripts/make_figures.sbatch
```

Commit a script together with the figures and numbers it changed.

## Building

`figures/` and `numbers/` are committed, so `cd paper && latexmk` needs only TeX; `.latexmkrc`
puts `main.pdf` and every build product in `build/` (ignored).

## Layout

| path | what |
|---|---|
| `main.tex` | the template, packages and `\input`s |
| `macros.tex` | notation, units, acronyms |
| `appendix/` | the drafted appendices |
| `scripts/` | `fig_*.py`, `numbers_*.py`, `style.py` (paths, palette, `save`, `save_numbers`), `make_figures.sh`/`.sbatch` |
| `numbers/`, `figures/` | generated; `figures/src/` holds the TikZ source |
| `references.bib` | checked against arXiv and INSPIRE-HEP |
| `icml2026.*`, `algorithm*.sty`, `fancyhdr.sty`, `siunitx*` | style files: ICML's, and siunitx 3.3.10 vendored |
