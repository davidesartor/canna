# Paper draft (ICML 2026 template)

`main.tex` inputs the main text from `sections/*.tex` (Results still red TODOs) and the appendices from `appendix/*.tex`.

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

Six report data folders the scripts read are gitignored because they regenerate in minutes, so
on a fresh clone run their CPU jobs first (from the repo root, `source slurm/unity.env`):

```bash
for name in frequency chirp-mass amplitude sky polarization; do                        # reports/prior/<name>/data/<name>.json
    sbatch $CANNA_SBATCH_CPU reports/prior/$name/scripts/$name.sbatch
done
sbatch $CANNA_SBATCH_CPU reports/diagnostics/forward-model/scripts/forward-model.sbatch  # reports/diagnostics/forward-model/data/forward-model.json
```

The other `reports/*/data` the scripts read (benchmarks, mcmc, strub-source, test-power) are
committed.

## Building

`figures/` and `numbers/` are committed, so `cd paper && latexmk` needs only TeX; `.latexmkrc`
puts `main.pdf` and every build product in `build/` (ignored).

## Layout

| path | what |
|---|---|
| `main.tex` | the template, packages and `\input`s |
| `macros.tex` | notation, units, acronyms |
| `sections/` | the main text |
| `appendix/` | the appendices |
| `scripts/` | `fig_*.py`, `numbers_*.py`, `style.py` (paths, palette, `save`, `save_numbers`), `make_figures.sh`/`.sbatch` |
| `numbers/`, `figures/` | generated; `figures/src/` holds the TikZ source |
| `references.bib` | checked against arXiv and INSPIRE-HEP |
| `icml2026.*`, `algorithm*.sty`, `fancyhdr.sty`, `siunitx*` | style files: ICML's, and siunitx 3.3.10 vendored |
