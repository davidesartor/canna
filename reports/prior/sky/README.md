# Sky prior: the galactic mixture, its frames and its flow coordinates (2026-10-30)

## Takeaways

- **The prior traces the galactic plane, not a point.**
  - 95.1% of the training windows' sources have |b| < 10°.
  - Only 37%, 53%, 63% and 78% lie within 0.1, 0.2, 0.3 and 0.5 rad of the galactic centre.
  - The plane's density falls from 35.5 sr⁻¹ at the centre to 0.189 at l = 90° and 0.00595 at the
    anticentre. That is 9.6× the isotropic floor's 6.2e-4.
- **Each of the four parts does its own job.**
  - The weights are 25.15 / 15.59 / 58.48 / 0.78%, fitted to a 9M-source WD–WD catalogue. The
    drawn shares are 25.14 / 15.58 / 58.51 / 0.77%.
  - The bulge makes the central peak: 93% of it lies within 0.1 rad of the centre.
  - The thin disk (99.6% at |b| < 10°) and the thick disk (74.5%) make the band.
  - The isotropic floor carries 34% of b's second moment.
- **The frame changes the picture.**
  - Near-plane shares (|lat| < 10°): 52.3% of sources in the ecliptic frame and 9.1% in the
    equatorial frame, against 95.1% in the galactic frame.
  - The galactic plane is tilted 60.2° to the ecliptic and 62.9° to the equator.
  - The galactic centre sits at (λ, β) = (266.8°, −5.5°).
- **The frame matters to LISA through the ecliptic latitude β.**
  - β sets LISA's Doppler and antenna modulation. The SNR side of this is on
    [the f₀, ḟ, A page](../f0-fdot-amplitude/f0-fdot-amplitude.html).
  - The prior's median |β| is 9.3°, against 30° for an isotropic sky. 52.4% of sources have
    |β| < 10°, and the mean cos β is 0.931 (isotropic: 0.785).
  - Median |β| per part: bulge 5.5°, thin disk 12.9°, thick disk 21.8°, floor 30.1°.
- **The draws match the density.**
  - χ²/dof of every marginal (3 frames × lat/lon × the 4 parts, their sum and the training windows)
    lies within 1.7σ of 1 for its dof, except the training windows' cells (0.82–1.42), which reach 2.5σ.
  - The KS distance from the centre is 1.5e-3 for the sum.
  - Every component integrates to 1 in the ecliptic and equatorial quadratures.
  - In the galactic frame the disks come out at 0.999987 (thick) and 0.999805 (thin). This is the
    midpoint rule's error on the Laplace cusp at b = 0, which predicts 0.9999873 and 0.9998046. It
    is not a bug.
- **The rotations are exact.**
  - `EQUATORIAL_TO_GALACTIC` is orthonormal: |MMᵀ − I| = 2.2e-16 and det = 1.
  - The round-trip error is ≤ 1.1e-15 rad.
  - The galactic centre is 0.022″ from Hipparcos. The galactic and ecliptic frames are 0.025″ and
    0.042″ from astropy.
  - The retired Gaussian proxy was centred 23.4° from the true centre.
- **l and b reach the flow whitened, and little mass sits near the l wrap.**
  - `GALACTIC_SCALE` = (0.53038, 0.10341) rad, the prior's own RMS, so the flow sees std 0.995 (l) and
    1.032 (b) over the training windows (1.001 and 0.999 over the 2²⁰ draws).
  - Both are flat coordinates nothing clips: 98% of the sources lie in [-3.07, 3.10] (l) and
    [-3.01, 3.16] (b).
  - l wraps at ±5.92 flow units. Only 0.048% of the mass lies within 0.1 rad of l = ±π, against
    3.18% for a uniform l.
  - The two bands draw the identical sky. The four sources of a window are unordered and every one
    sees the same sky (std l 0.98–1.01, b 0.97–1.08), and |r| ≤ 0.014 with the other flow coordinates.

## What the page shows

[sky.html](sky.html). The top bar picks the **frame** (galactic / ecliptic / equatorial) and the
**part** (sum, bulge, thick disk, thin disk, isotropic floor). Together they drive every card.

- **Tiles:** the near-plane share in each frame, the share within 0.5 rad of the centre, the flow std of l and b,
  the mass near the l wrap and the frame round-trip error.
- **Cards:**
  1. **Where on the sky does the prior put sources?** A Mollweide map of the chosen part's density
     in the chosen frame, on a log colour scale whose ramp bottoms out at the floor. It overlays the
     galactic plane, the ecliptic and the equator, and marks the galactic centre (plus, in
     equatorial, the old proxy's centre). Hovering shows each part's share of the density.
  2. **What does each part of the mixture contribute?** Weights, scales, drawn shares and each
     part's near-plane share per frame.
  3. **Do the draws match the density in every frame?** Latitude and longitude marginals against
     the quadrature, plus a χ²/dof table for every frame, part and coordinate. A source seg picks
     2²⁰ sky draws (per part) or the 32,768 training-window sources.
  4. **How far from the galactic centre do sources sit?** The cumulative share within r of the
     centre per part, drawn against quadrature, with KS distances.
  5. **Why does the frame matter to LISA?** The |β| histogram against an isotropic sky, the tilts,
     the centre in ecliptic coordinates, and a link to the f₀, ḟ, A page.
  6. **Are the frame rotations and the mixture exact?** Orthonormality, round trips, Hipparcos and
     astropy checks; each component's integral per frame, with the cusp predictions.
  7. **What do l and b look like in the flow's coordinates?** The flow-unit marginals against the
     l wrap lines; per-part std and variance share; the mass near the
     wrap; per-source-index stds; the correlations of l and b with the other flow coordinates.
  8. **What this means**: the takeaways, short.

The state lives in the URL hash, e.g. `#frame=ecliptic&part=thin&src=sky` (also `band`,
carried to the sibling prior pages).

## Method

[`draw_sky.py`](scripts/draw_sky.py), run by [`sky.sbatch`](scripts/sky.sbatch) on CPU: job 66324816,
54 s at src 8310cb3 (smoke: job 66324743, 45 s).

- **Draws and parts:** 2²⁰ draws of `physics.sample_sky`. Each draw's part is recovered from its key
  with the same split and choice that `sample_sky` makes (`component_of`), so the per-part
  histograms come from the real sampler.
- **Densities:** `sky_log_density` and its per-component terms, evaluated on a 480 × 240 map and a
  2880 × 1440 quadrature grid in each frame. The marginals, near-plane shares and from-centre
  CDFs are integrated from that grid.
- **Frames:** from `EQUATORIAL_TO_GALACTIC` and the J2000 obliquity. They are checked against
  Hipparcos' pole and centre, and against astropy (`Galactic`, `BarycentricMeanEcliptic`) on
  100,000 draws.
- **Training windows:** 8192 windows per band (the sources unordered), built with `sample_f` → `sample_physical` →
  `physical_to_flow` on `config.load("<band>-B")`. They give the flow-unit l and b, per-source-index stds
  and correlations.
- **Bulge clipping:** the untruncated bulge candidates are re-drawn (`bulge_candidate`) to count
  clips at |sin b| > 1 and |l| > π.

[`build_page.py`](scripts/build_page.py) uses only the standard library. It computes the χ²/dof
(over bins expecting ≥ 5 draws, dof = bins − 1), the KS distances, the cusp predictions and the old
proxy's offset, and it prints every number above.

The mixture's catalogue fit itself cannot be redone here: the catalogue is not in the repo.

## How to rebuild

The page runs on a login node:

    uv run --frozen python reports/prior/sky/scripts/build_page.py

To redraw (JAX, a CPU job, ~1 min), submit from the repo root:

    sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/prior/sky/scripts/sky.sbatch

## Data

`data/sky.json`, from job 66324816:

- **`sky`:** frames (`lon0`, `centre`, `integral`, base64 `map`, per-component integrals and
  near-plane shares), `components` (names, weights, scales, catalogue size), `galactic_scale`,
  `quadrature`.
- **`sky_draws`:** per frame and part, the marginals with quadrature reference counts, the
  near-plane shares, and the from-centre CDFs.
- **`checks`:** the rotation and mixture checks.
- **`lisa`:** tilts, the centre in (λ, β), and the |β| statistics per part.
- **`chart`:** `GALACTIC_SCALE`, the wrap, and per-part flow stds and variance shares; `near_wrap`, `plane_density`, `clip`.
- **`bands.{narrow,full}`:** `sky` (`marginals`, `abs_lat_below_10deg`, `within_rad_of_centre`),
  `flow` (l and b), `slots` (per source index), `corr_l` and `corr_b`.

The `prior.json` key names are kept for everything carried over from the old sky page.
