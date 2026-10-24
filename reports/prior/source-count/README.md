# Source count: how many sources, and how many visible ones, a window holds (2026-10-26)

## Takeaways

At N = 9,000,000 sources in the band and full-B's w = 512 bins (4 × R = 128):

- **A window holds hundreds of thousands of sources.** The median is 172,651, and 99% of windows
  hold at most 345,032.
- **Most windows hold no visible source.** 0.70% of the prior's sources reach SNR 7, a mean of 6.9
  per window. Windows are centred where the prior is, mostly below where SNR 7 is reachable, so
  90.8% of windows hold none.
- **The rest can hold hundreds.**
  - 6.3% of windows hold more than the 4 source tokens.
  - 99% of windows hold at most 263 visible sources, 99.9% at most 329 and 99.99% at most 352.
  - The busiest centres, near 1.68 mHz, hold up to 365 in 99% of their windows. 1.3% of windows
    are centred within 0.1 dex of them.
- **The count scales with the window.** At SNR ≥ 7, 99.99% of windows hold at most 30 at w = 32,
  97 at w = 128, 352 at w = 512 and 2649 at w = 4096.

## The question

How many source tokens does a window need? Take N sources in the band, each drawn independently
from the prior. Then a window of w bins centred at f holds each source with probability

p(f, w) = ∫_window prior(f₀) P(SNR ≥ t | f₀) df₀,

so its count is K ~ Bin(N, p(f, w)). This is the prior thinned to the sources at SNR ≥ t,
prior(f₀ | SNR ≥ t), with N·P(SNR ≥ t) of them. t = 0 counts every source. Window centres follow the
prior's f₀ marginal, as training draws them, so the distribution of K over windows is a mixture of
these binomials.

## What the page shows

[source-count.html](source-count.html), hand-written JS like the other prior pages (auto
light/dark, hash state `#snr=…&w=…`). Single problem, `full-B`. The controls set:

- the SNR threshold t, from 0 to 30 in steps of 0.5 (default 7);
- the window width w, as powers of two from 32 to 4096 bins (default 4 × `response_points`).

The cards are:

- **How many sources does a window hold?** The share of windows holding more than k sources,
  P(K > k), on a log scale down to 10⁻⁶, against log k, one curve per N in 1M, 3M, 10M, 30M, 100M.
- **Where are the windows that hold the most?** Per window centre, K's median and its 10–90, 1–99,
  0.1–99.9 and 0.01–99.99% intervals, both bounds in one shade, and its mean N·p, at the catalogue's N = 9M.
- **What this means:** the takeaways at t = 0 and t = 7, computed from the data.

## Method

- `scripts/draw_count.py` runs on CPU under Slurm (`scripts/count.sbatch`).
  - It draws 65536 sources of `config.load("full-B")` from `jr.split(jr.key(0), 65536)`. f₀ is
    log-uniform over the band and weighted back to the catalogue's f₀ marginal. ḟ, sky and
    orientation follow the prior.
  - Each source gets its SNR at A = 1 against a 4096-point jaxgb reference (A, E, T), and the mean of
    log₁₀ A given (f₀, ḟ).
  - These are response-span's draws: same key, same draw function.
- `scripts/build_page.py` builds the page (numpy + scipy + hand-written JS, data inlined).
  - log₁₀ A is Gaussian given (f₀, ḟ), so P(SNR ≥ t) = Φ((μ − log₁₀(t / SNR₁)) / σ_A) per source, in
    closed form. Its f₀-weighted mean in 96 log-f₀ bins gives P(SNR ≥ t | f₀) for t = 0…30.
  - The window integral is a 9-point Simpson rule in log₁₀ f₀ over the prior's f₀ marginal, clipped to
    the band.
  - The centres are 600 log cells weighted by the marginal. As in `window_index_range`, each is
    clamped so the window stays at most half a response's bins past the band edge.
  - The takeaways use scipy's exact binomials. In the page's JS, each centre's binomial is exact by
    recurrence up to a mean of 500 and normal (with continuity correction) past it.

## How to rebuild

From the repo root:

```bash
# redraw and rebuild the page (CPU, ~3 min; an argument sets the source count, a multiple of 32, 32 for a smoke test)
sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/prior/source-count/scripts/count.sbatch
# rebuild the page only (login node is fine)
uv run --frozen python reports/prior/source-count/scripts/build_page.py
```

The current `data/` was copied from response-span's run (job 66137238 at `bc6211f` with `src/`
dirty). Its draws are the same, and its JSON records `response_points` 128 and `n_sources` 4 from
the current full-B config.

## Data

`data/` (gitignored), written by `draw_count.py`:

- `source-count.npz`: per source f₀, its f₀ weight, its SNR at A = 1 and the mean of log₁₀ A;
- `source-count.json`: the commit, the config, the source count, A's conditional std, T, the band,
  `response_points`, `n_sources` and the reference length.
