# Source pairing: path length per source in f0 order, against the nearest labelling, loudest first or random labels (2026-10-30)

## Takeaways

Job 66326150 on src commit 8310cb3, 65536 windows per band, 4 sources. Windows come from
`sample_window` at the training default `visible_snr = 7`, the sources in them from the catalogue.
Mean squared path length per source, flow coordinates (± one standard error):

| pairing | narrow-B | full-B | f0 | log ḟ | sky | z± narrow | z± full |
|---|---|---|---|---|---|---|---|
| f0 order (training) | 7.38 ± 0.02 | 14.60 ± 0.03 | 0.26 | 2.01 | 4.03 | 0.54 | 4.15 |
| nearest labelling | 5.58 ± 0.02 | 10.83 ± 0.03 | 0.49 | 1.10 | 3.03 | 0.48 | 2.82 |
| loudest first | 7.33 ± 0.02 | 14.56 ± 0.03 | 0.67 | 1.56 | 4.03 | 0.54 | 4.15 |
| random labels | 7.78 ± 0.02 | 15.00 ± 0.03 | 0.67 | 2.00 | 4.03 | 0.54 | 4.15 |

(f0, log ḟ and sky columns: narrow-B; full-B agrees to 0.01 except the nearest labelling,
f0 0.57, log ḟ 1.37, sky 3.27. z± is per amplitude.)

- **f0 order has the shortest f0 paths**, 0.26 against 0.49 (narrow) and 0.57 (full) for the
  nearest labelling and 0.67 for the labellings that ignore f0; no two paths cross in f0.
- **The nearest labelling's total is 24% (narrow) and 26% (full) below f0 order's**: it trades
  f0 length for log ḟ (−0.9 narrow, −0.6 full), sky (−1.0, −0.8) and, on full, z± (−1.3 each).
  f0 order is the nearest labelling in 9% (narrow) and 7% (full) of windows.
- **f0 order is 5% (narrow) and 3% (full) shorter in total than random labels**, all of it from
  f0; in log ḟ, sky and z± it pairs no better than chance.
- **Loudest first is 0.7% (narrow) and 0.3% (full) shorter in total than f0 order**, not random:
  loudness rank carries some of log ḟ (1.56 against 2.00 for random) but nothing of f0, sky or z±.
  f0 order is loudest first in 4% of windows.
- **The band matters through z± alone**: 0.54 per amplitude on narrow, 4.15 on full. At the
  catalogue's windows (the previous run, `visible_snr = 0`) they were 0.07 and 0.02 and the totals
  6.34 and 6.44; f0, log ḟ and sky did not move.

## What the page shows

[source-pairing.html](source-pairing.html). The top bar picks the band (narrow-B or full-B prior
window).

- **Tiles:** the mean squared path length per source in f0 order (what training uses); the
  nearest labelling and loudest first against it; the share of windows where f0 order is already
  the nearest labelling.
- **Where is the path length?** The length per source split by coordinate group (f0, log ḟ, sky,
  z+, z−), one bar per pairing, with a table and the 10/50/90% quantiles of each window's length
  over f0 order's.
- **Which sources pay for a fixed order?** The length per target source by its loudness rank.
- **What this means.**

## Method

`scripts/pairing.py`, network-free, on `src/` (commit 8310cb3):

- Per window, as `train_sample` draws it: `f` from `sample_window` (`visible_snr = 7`), the target X1 from
  `sample_physical` in flow coordinates, one base draw X0 from `sample_flow`. Both come loudest
  first.
- Pairings of X0's sources with X1's:
  - **f0 order:** both ends sorted by f0 and paired by rank, as `paths` sorts a batch's sources (what training uses; `match_f0` is gone, the pairing is inlined as `f0_rank_pairing`);
  - **nearest labelling:** the least total squared distance over the S! labellings, all
    coordinates (the coupling of equivariant flow matching, exhaustive here);
  - **loudest first:** X0 as drawn;
  - **random:** X0 under a uniform permutation.
- Length: squared distance per (target source, coordinate), summed per group as `LOSS_GROUPS`
  splits them, averaged over sources and then over windows. The standard error is over windows.
- 65536 windows per band (key 0), CPU, chunks of 1024.

## How to rebuild

```
sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/architecture/source-pairing/scripts/pairing.sbatch
```

The job writes `data/pairing.json` (gitignored, regenerates in minutes) and then the page.
`scripts/build_page.py` (stdlib) rebuilds the page alone. A smoke test
(`--export=ALL,PAIRING_WINDOWS=2048`) writes `data/pairing-smoke.json` and no page.

## Data

`data/pairing.json`: `commit`, `groups`, `pairings`, `date`, and per band `sources`, `windows`, `mean`
[pairing → per group], `total` and `total_se` [pairing], `by_rank` [pairing → rank × group],
`ratio_quantiles` [pairing → 10/50/90% of the per-window total over f0 order's],
`f0_is_nearest`, `f0_is_snr_order`.

## History

Replaces `slot-assignment` (2026-10-24), which probed how a trained flow with loudest-first slots
assigned its draws. Research-log F34 measured the same lengths on the code of 2026-10-13 (XS
windows, the chart with Mc): nearest labelling 8.73, SNR order 12.04, f0 order 12.04, random 12.44.
