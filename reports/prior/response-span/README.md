# Response span: the R that keeps x of a source's above-noise power with probability y (2026-10-26)

## Takeaways

- **R = 128 keeps almost all of what the data see, 8× below the hard bound.**
  - It keeps 99% of the above-noise power for 99% of the sources, and every above-noise bin for
    99.9% of them. The hard bound, which autosizing used, is 1024.
  - full-B's `response_points` is now 128, and the sampler no longer cuts ḟ.
  - 99.99% of the sources span at most 166 bins.
- **Most visible sources are narrow.**
  - The median source needs 18 bins for all its above-noise power.
  - R = 32 keeps 99% of it for half the sources and R = 64 for 90%. Per f₀, R = 64 does so for 99%
    of the sources up to 4.6 mHz.
  - Only the top of the band, above 7.4 mHz (0.18% of the visible sources), needs 256.
- **Leakage below the noise is gone, so x = 100% costs no more than 99%.**
  - Past the response band, the sinc sidelobes sit under the noise. R(100%, y) equals R(99%, y) for
    every y on the page.
- **narrow-B's R = 32 loses nothing.** Below its 1.26 mHz top (17% of the visible sources), every
  above-noise bin fits for 99.9% of them.
- **1.18% of the prior's sources have a bin above the noise.**
  - 0.70% reach SNR 7, so 41% of the visible sources stay below it. The faintest visible one has
    SNR 1.6.
- **A higher SNR cut widens the tail.** Brighter sources lift more sidelobes above the noise.
  - Counting only SNR ≥ k, the sources `LisaGB.visible_snr = k` places training windows on (default
    k = 7): R(99%, 99%) stays 128 for every k ≤ 20, and R(100%, 99.9%) is 128 up to k = 7, then 256 from k = 10.
  - The median source's full width grows from 18 bins (off) to 26 at k = 7 and 44 at k = 20.
- **The brute-force draw agrees.**
  - 1,048,576 prior sources, with ḟ cut at 5σ, ran in 21 min (job 66136660), against 3 min here.
  - 1.16% of them are visible, and their width quantiles match to 2 bins up to y = 99%. At 99.9%
    they come 6–14 bins narrower (it cut ḟ), and every R(x, y) on the page is the same.

## The question

Autosizing gave every full-B source R = `response_points` = 1024 bins, the next power of two above
`response_span` of the fastest chirp the sampler then allowed (ḟ cut 5σ above its mean) at the top of
f0. That is a hard bound, and it exists only because of the cut. The draws here do not cut ḟ.
The fuzzy bound R(x, y) is the smallest power of two such that a single source,
drawn from the prior, keeps at least a share x of its **above-noise power** inside its R bins with
probability y.

- A bin's signal power is p = 2|h|²/(S_n·T/2) per A, E, T channel, in units where it sums to SNR².
  The noise there has expected power 2.
- A source's above-noise power is Σ max(p − 2, 0) over bins and channels. Power the noise buries,
  such as far sinc sidelobes, does not count. So x = 100% is finite: the outermost bin above the
  noise.
- Sources with no above-noise power are invisible to the data and do not count: y is over the
  sources that have some.

## What the page shows

[response-span.html](response-span.html), hand-written JS like the other prior pages (light/dark,
hash state `#k=…`). Single problem, `full-B`. One slider sets the SNR cut k ∈ {off, 1, …, 20}; the
tiles above the cards (counted share, R(99%, 99%), R(100%, 99.9%), R in use and the hard bound)
follow it.
- **SNR ≥ k** counts only the sources that also reach SNR k; off counts every source above the noise.
  Both cards follow it. `LisaGB.visible_snr = k` (default 7) centres training windows by about the
  prior's f₀ times P(SNR ≥ k | f₀): the counted sources' curve, which the page weights to. The sources
  in those windows stay the catalogue's, so fainter ones occur too; they are narrower, so at each f₀
  the R shown still holds them. The dashed curve is the catalogue's f₀ (`sample_f`).

- **How much of a source's above-noise power does each R keep?** One line per R on nines axes,
  −log₁₀(1 − x) against −log₁₀(1 − y), with x = 100% as the last point. Only the first R that keeps
  everything is drawn; R in use is bold.
- **How many bins does a source need at each f₀?** The weighted density of the counted sources in
  (log f₀, log full width), with contours holding 50/90/99% of them, against `response_span` at the
  old 5σ ḟ ceiling (the hard bound), the R in use, and narrow-B's R and f₀ top. Widths come in even
  bins, so each is spread uniformly over its step before a one-cell smoothing. The tooltip gives the
  50–99.9% width quantiles per f₀ bin.
  - Below it, on the same f₀ axis, the share of sources per bin: the prior's f₀ marginal in the band
    (the catalogue) and that of the counted sources (where `visible_snr = k` centres training windows)
    (peaks 0.23 vs 1.6 mHz with the cut off).
- **What this means:** the takeaways, computed from the data, and how each quantity is measured.

## Method

- `scripts/draw_span.py` runs on CPU under Slurm (`scripts/span.sbatch`, 3 min; job 66137238 at
  `bc6211f` with `src/` dirty).
  - It draws 65536 sources of `config.load("full-B")` from `jr.split(jr.key(0), 65536)`. Each draw
    has log f0 uniform over the band, weighted back to the catalogue's f0 marginal, so the visible top
    of the band gets as many sources as the bottom. ḟ, sky and orientation follow the prior, with ḟ
    uncut.
  - Each source's response comes from a 4096-point jaxgb reference at A = 1 (4× the hard bound, with
    ι → π/2 − ι as in `source_segments`), A, E and T.
  - The response is linear in A, so A is integrated out. Given (f0, ḟ), log₁₀ A is Gaussian. A source
    has a bin above the noise iff A² · max power > 2, so the prior share of A that makes it seen is
    closed form. 16 stratified draws of A inside that share then carry the rest.
  - jaxgb centres an n-point response on rint(f0·T), so an R-point response holds the bins
    [rint(f0·T) − R/2, rint(f0·T) + R/2). Per A draw the script records the above-noise share each R
    in 4…2048 loses, and the x-width: the narrowest such centred band holding x, for
    x = 50, 90, 99, 99.9, 100%.
- `scripts/build_page.py` builds the page (numpy + hand-written JS, data inlined). A draw's weight is
  its f0 weight × the seen share of A / 16; SNR ≥ k keeps the draws at A · SNR(A = 1) ≥ k, which
  stays unbiased since the A draws are stratified. Quantiles are the weighted inverted CDF. An f0 bin stays
  empty where fewer than 1/(1 − y) effective sources (Kish) lie above the quantile.

## How to rebuild

From the repo root:

```bash
# redraw and rebuild the page (CPU; an argument sets the source count, a multiple of 32, 32 for a smoke test)
sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/prior/response-span/scripts/span.sbatch
# rebuild the page only (login node is fine)
uv run --frozen python reports/prior/response-span/scripts/build_page.py
```

## Data

`data/` (gitignored), written by `draw_span.py`:

- `response-span.npz`: per source f0, its f0 weight, SNR at A = 1, the mean of log₁₀ A and the seen
  share of A; per A draw log₁₀ A, the x-widths and the share each R loses;
- `response-span.json`: the commit, the config, the source and A-draw counts, A's conditional std,
  R, narrow-B's R and f0 top, the x and R lists, the noise power and the hard-bound grid.
