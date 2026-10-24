# The LisaGB prior: polarization (2026-10-30)

## Takeaways

- **The orientation is uniform, as drawn.**
  - sin ι, ψ and φ₀ are uniform on [-1, 1], [0, 2π) and [0, 2π). 8192 windows of 4 sources give 32,768 sources per band.
  - χ²/dof against the exact prior (bins with ≥ 5 expected, dof = bins − 1): sin ι 32/47, ψ 58/47, φ₀ 37/47, ι 39/47 (both bands, same keys).
  - 50.2% of sources have sin ι < 0. Median axis ratio ε = 2|s|/(1 + s²) is 0.798.
- **The flow predicts z±, which are the two circular components of the polarization ellipse.**
  - z± = A(1 ∓ s)²/4 · e^{i(φ₀ ± 2ψ)}, and the ellipse is w = −A e^{2iψ}[(1 + s²)cos θ + 2is sin θ], θ = Ω − φ₀.
  - Its semi-axes are A(1 + s²) and 2A|s|, with the major axis at 2ψ mod π. z₋ dominates for s > 0 (counter-clockwise) in 49.8% of sources.
  - The zW form agrees with jaxgb's own arithmetic to 1.7e-15, the axis ratio to 4.9e-16 and the z₊/z₋ split to 2.5e-16 (full).
  - The Python and JS ports of `physical_to_flow` and `flow_to_physical` match the 12 test vectors to ≤ 6.7e-16 (angles wrapped).
- **The z± coordinates are narrow compared with the other flow coordinates.**
  - The catalogue prior puts A far below one `amplitude_unit`, where asinh is linear, so the warped z± are about the unwarped ones.
  - The four warped Re/Im z± have std 0.143–0.150 (narrow 0.072–0.075), against 0.58–1.03 for f₀, ḟ, l and b. That is 0.16× their mean (narrow 0.08×).
  - The largest |Re z±| or |Im z±| is 4.158 (narrow 1.815). asinh|z±| has rms 0.207 and 0.205 (full), 0.104 and 0.1036 (narrow).
  - The largest off-diagonal correlation of the eight flow coordinates is 0.030 (full, Re z₊ with Im z₊) and 0.014 (narrow).
  - κ = SNR / (A / unit) has median 1.63 over the 1500 scatter sources (both bands).
- **The angles are singular where z± are not.**
  - The Fisher condition number in (ln A, s, ψ, φ₀) is 1.18 at s = 0 and reaches 3.2e20 (full) and 4.7e21 (narrow) at |sin ι| = 0.99999.
  - In z it stays at 1.177, flat to 2.8e-6 (full). In the warped flow coordinates it is 1.18 in both bands, since the warp is linear at these amplitudes.
  - corr(ψ, φ₀) goes to ∓1 at the poles. Where the marginal curve is missing, the matrix is singular in float64.
- **Four angle choices are one signal.**
  - (ψ + kπ/2, φ₀ + (k mod 2)π) agree to ≤ 5.4e-13 in the signal over 256 sources, against 1.414 (median) for a wrong step.
  - `flow_to_physical` lands on the four branches for 8254, 8150, 8197 and 8167 of 32,768 sources (full).
  - The returned ψ is the triangle (2π − 4|c|)/π² on the centred branch (χ²/dof 56/47 full, 45/47 narrow); the returned φ₀ is uniform under the middle reference (78/95 full, 79/95 narrow).
  - At a pole only one phase survives: moving ψ and φ₀ so that it is kept costs ≤ 5.8e-13; moving it by 2 sin 1.4 costs 1.971.
- **The round trip is exact to the float.**
  - physical → flow → physical gives a worst signal mismatch of 5.1e-11 (full) and 4.6e-11 (narrow) in float64, over 256 sources.
  - A float32-rounded flow output fed into the float64 chart gives 2.7e-5 (full) and 6.0e-6 (narrow). This is not a float32 chart.
  - sin ι is recovered to 2.2e-16 for gap ≥ 1e-6 and to 1e-6 (the clip) at the poles.
- **The phase reference sits at the window middle.**
  - corr(f₀, φ), conditional, falls from 0.872 (start) to 0.025 (middle) in full-B, and from 0.871 to 0.014 in narrow-B. The marginal goes 0.533 → −0.241 (full).
  - The start reference's f₀-dependent shift is 2.0e4–1.4e6 rad (full), and float32 would round it by up to 0.12 rad. σ(φ) is 1.18× larger at the start (median).

## What the page shows

[polarization.html](polarization.html). A band picker (narrow-B, full-B). The nav links the other prior pages and carries the band:
- [f₀, ḟ, A](../f0-fdot-amplitude/f0-fdot-amplitude.html);
- polarization;
- [sky](../sky/sky.html);
- [response span](../response-span/response-span.html);
- [source count](../source-count/source-count.html).

**Tiles**: the z± std range against the other coordinates, the four-branch signal mismatch, the round trip in float64 and float32, corr(f₀, φ) start → middle, the worst condition numbers, and the largest |Re z±|, |Im z±|.

1. **Which angles does the prior draw?** Toggle sin ι, ι, ψ, φ₀, each against its exact prior with χ²/dof.
2. **How do the angles make z±?** The formulas with `jaxgb.py` line references, and a table of every check (ellipse identity, axis ratio, split, Fisher variation, linearity, Python and JS ports).
3. **What does the polarization ellipse look like?** A gallery of s ∈ {1, 0.5, 0, −0.5, −1} × ψ ∈ {0, π/8, π/4, 3π/8}. Clicking a cell loads it into the explorer.
4. **Explorer.** Sliders in physical, z± Cartesian or z± polar coordinates. It shows the ellipse with the zW curve, jaxgb's points and the two phasors, the z₊ and z₋ planes against the prior's cloud, and h₊/h× over Ω (with a play button). The readout gives all three systems, the four equivalent (ψ, φ₀) with the returned one marked, ε, V/I, handedness and pole warnings.
5. **Polarization shape.** Toggle axis ratio, V/I, log₁₀|z₊|/|z₋|, |z±| against A, and arg z±.
6. **The prior in three coordinate systems.** Marginals, a correlation table and a scatter for the physical, polar and Cartesian systems.
7. **The flow coordinates.** Marginals of the four Re/Im z± coordinates, std of all eight, and the 8 × 8 correlations.
8. **Conditioning against sin ι.** Condition numbers of the Fisher matrix in angles, z and flow coordinates; corr(ψ, φ₀); a table near the poles.
9. **The four branches.** The variants table, the returned ψ and φ₀ against closed forms, and the pole kept/lost table.
10. **Round trip.** The float64 and float32 tables and the pole table by gap.
11. **The phase reference.** Start against middle: corr(f₀, φ), conditional and marginal, and σ(φ) per source.
12. **What this means.**

The state lives in the URL hash, e.g. `#band=narrow&angle=psi&coord=cart&ex=1,0.5,0.4,1&om=0`.

## Method

[`draw_polarization.py`](scripts/draw_polarization.py) and [`polarization.sbatch`](scripts/polarization.sbatch): job 66324891 on CPU at src 8310cb3 (2 min 36 s; smoke job 66324739).

- **Draws:** 8192 windows per band with `jr.key(0)` on `LisaGB(**config.load("<band>-B").problem)`: `sample_f`, `sample_physical`, then `physical_to_flow`. The orientation histograms have bin widths 2π/96 for ψ and φ₀, and their references come from the closed forms.
- **χ²/dof:** over bins whose reference count is at least 5, dof = bins − 1.
- **Reference counts for the returned angles:** ψ centred c in (−π/2, π/2) has density (2π − 4|c|)/π², and φ₀ is uniform under the middle reference (under the start reference it would be (π − |c|)/π²). The midpoint rule is exact for these piecewise-linear densities.
- **Fisher matrices:** one source per sin ι grid point, in (ln A, s, ψ, φ₀), z and warped flow coordinates, each unit-diagonal. The page notes the source's f₀ and ḟ.
- **Round trip and poles:** 256 sources; the pole table pushes sin ι to ±(1 − gap).
- **Phase reference:** 256 sources, Fisher matrices with the phase referenced at the window start and at the middle.
- **Test vectors:** 12 (A, s, ψ, φ₀) rows with their z and `flow_to_physical` output at full precision.

[`build_page.py`](scripts/build_page.py) uses only the standard library. It reads `data/polarization.json`, adds the closed-form references and every χ²/dof, checks its Python port against the test vectors, and prints every number above. The page re-runs the same port check in JS (tolerances 1e-9 for the port, 1e-12 for the ellipse) and shows a banner if it fails.

## How to rebuild

The page needs only the standard library and runs on a login node:

    uv run --frozen python reports/prior/polarization/scripts/build_page.py

To redraw (JAX, a CPU job, about 1 min per band), submit from the repo root. It runs `build_page.py` as its last step:

    sbatch --partition=cpu,cpu-preempt --account=pi_mehuber_umass_edu --qos=normal reports/prior/polarization/scripts/polarization.sbatch

## Data

`data/polarization.json`, from job 66324891: `{device, windows, bands: {narrow, full}}`.

Each band carries these keys under the names of the retired single bank `prior.json`:
- `physical.orientation` (`psi [rad]`, `sin iota`, `phi0 [rad]`, plus `iota [rad]`);
- `flow`: the four Re/Im z± entries with `hist` and `std`;
- `radius`, `corr`, and `scatter` (`sin_iota`, `psi` and the rest; angles and logs to 4 decimals, `snr`, `rp` and `rm` to 4 significant digits);
- `problem.phase_reference`.

It adds new keys:
- `helicity` (axis ratio, V/I, log₁₀|z₊|/|z₋|, |z±|, A, arg z±), `axis_ratio_check`, `split_check`;
- `flow_std` (all eight), `systems` (physical, polar, Cartesian);
- `round_trip` (float64, float32, `poles`), `degeneracy` (variants, branches, returned ψ and φ₀, `pole_phase`);
- `phase` (start against middle), `singularity` (Fisher rows and the linearity checks), `test_vectors`;
- `seconds`, dropped from the page.
