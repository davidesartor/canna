# Strub source: does the pipeline reproduce a published LISA posterior?

## Takeaways

- **The pipeline reproduces Strub et al. 2022's faint source** (arXiv:2204.04467, Table VII):
  - optimal SNR 7.9508 against their 7.93, converged in window size and channel set;
  - f₀ 68% interval +1.29/−1.40 nHz against their +1.9/−1.5 nHz;
  - SNR gain from fitting ρ_MLE − ρ_obs = 0.405 ± 0.039 over 48 noise realisations, against
    their single-realisation 0.32 (scatter 0.27);
  - ι = 1.11 in the Fisher box against their 1.33; under the full prior 0.88, with a cos ι tail
    to face-on;
  - the noiseless chain centres 6 of 8 parameters within |pull| 0.25; log A (−1.33) and cos ι
    (+1.17) move towards face-on and low A, prior volume along their degeneracy, while the
    refined maximum likelihood sits at the truth.

Figure: [`strub-source.html`](strub-source.html), interactive and standalone. Every number on
it is computed in the page from the data it inlines.

## What the page shows

The view lives in the URL hash, so a view can be linked.

- **How do its posterior and SNR gain compare with Strub et al.'s?** Tabs for the Strub source:
  f₀ histograms under the full prior and the Fisher box with σ_f₀ over the sweep, cos ι, the SNR
  gain per realisation, and the noiseless pulls; then a table against Strub's values.
- **What this means**: the takeaway above, recomputed from the inlined data.

## Method

[`strub_source.py`](scripts/strub_source.py) builds a one-source `LisaGB` with
T = 63 115 200 s and Strub's parameters (sky rotated from ecliptic to equatorial), and runs
canna's sampler with a target in Strub's priors: f₀ within ±0.5 µHz, log₁₀ ḟ on [−18.5, −16]
as log M<sub>c</sub>, log₁₀ A on [−23.5, −21.5], isotropic sky, cos ι uniform. 32 walkers ×
6 temperatures, 10 000 burn, 30 000 kept, thin 10. Chains: the noiseless injection, noise
realisation 0 under the full prior and in a ±3σ Fisher box (as the original notebook), and 48
more realisations for the SNR gain and σ_f₀. All 51 chains have R̂ < 1.0001.

[`build_page.py`](scripts/build_page.py) writes `strub-source.html`, its page and `data/`
inline.

## How to rebuild

From the repo root, after `source slurm/unity.env` (or `trex.env`). The chains are JAX, so they
run as jobs.

```bash
# Strub chains (50 tasks, ~40 min), then the record and the page
sbatch $CANNA_SBATCH_CPU reports/diagnostics/strub-source/scripts/strub_chains.sbatch
sbatch $CANNA_SBATCH_CPU reports/diagnostics/strub-source/scripts/strub_summary.sbatch
```

Each sbatch header has a smoke-test line. The page alone is numpy and runs on a login node:

```bash
bash -c 'ulimit -v 4000000; uv run --frozen python reports/diagnostics/strub-source/scripts/build_page.py'
```

## Data

- `data/strub_source.json`: every Strub-source number, with the commit and `src/` tree it came
  from.
- `data/seed0.npz`: realisation 0's f₀ and cos ι draws under the full prior and in the box.
