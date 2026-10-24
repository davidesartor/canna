# The LisaGB prior: what each window asks the flow to cover (2026-10-26)

Five subreports, each a self-contained folder with its own page, README, data and scripts.

## Subreports

- [**f₀, ḟ, A**](f0-fdot-amplitude/README.md) ([page](f0-fdot-amplitude/f0-fdot-amplitude.html)): the
  Gaussian in (log₁₀ f₀, log₁₀ ḟ, log₁₀ A) fitted to a WD–WD catalogue, as a corner of the drawn
  sources against the catalogue, which sources the data see, and the flow coordinates.
  - ḟ follows f₀ at 3.48 per dex with 0.18 dex scatter, uncut. A rises only
    0.48 per dex.
  - Almost every source is invisible: 0.05% (narrow-B) and 0.84% (full-B) reach SNR 7, and 99.80%
    and 98.47% of windows hold none.
  - The visible ones sit at the top of the band, where the marginal-centred windows rarely go: the
    upper half of the band in log f₀ gets 3.08% of full-B windows.
- [**sky**](sky/README.md) ([page](sky/sky.html)): the galactic mixture, its frames and the (l, b)
  flow coordinates.
  - The prior traces the galactic plane, not a point: 95.2% of sources have |b| < 10°.
  - The four parts weigh 25.15 / 15.59 / 58.48 / 0.78%: bulge, thin disk, thick disk, isotropic
    floor.
  - LISA feels the frame through β: the median |β| is 9.3°, against 30° for an isotropic sky.
- [**polarization**](polarization/README.md) ([page](polarization/polarization.html)): the
  orientation angles, the z± chart and the phase reference.
  - sin ι, ψ and φ₀ are uniform as drawn. The flow predicts z±, the polarization ellipse's two
    circular components.
  - The angles are singular at the poles, where the Fisher condition number reaches 1.8e16. In z±
    it is a flat 1.17. The four angle branches give the same signal to 5e-13.
  - The warped Re/Im z± have std ≈ 2.47, 2.8× the other flow coordinates, but stay inside the
    network's reach.
  - Placing the phase reference at the window middle takes corr(f₀, φ) from 0.875 to 0.044 (full).

- [**response span**](response-span/README.md) ([page](response-span/response-span.html)): the
  fuzzy bound R(x, y), the smallest power of two R such that a source keeps at least a share x of its
  above-noise power inside its R response bins with probability y, over the sources the data can see.
  - R = 128 keeps 99% of the above-noise power for 99% of the sources, and all of it for 99.9%.
    That is 8× below the 1024 that autosizing took from the 5σ ḟ cut; full-B now uses 128, ḟ uncut.
  - The median visible source needs 18 bins. Only above 7.4 mHz (0.18% of them) does R reach 256.
  - narrow-B's R = 32 keeps every above-noise bin for 99.9% of its sources.
- [**source count**](source-count/README.md) ([page](source-count/source-count.html)): how many
  sources, and how many at SNR ≥ t, a window of w bins holds, K ~ Bin(N, p(f, w)) mixed over window
  centres, with a slider for t, buttons for w and one curve per N.
  - At N = 9M and w = 512, a window holds a median 172,651 sources. 90.8% hold none at SNR ≥ 7.
  - 6.3% of windows hold more than the 4 source tokens, and 99% hold at most 263 visible sources.
    The busiest centres, near 1.68 mHz, hold up to 365.

## Data

Each subreport writes `<name>/data/` (gitignored, minutes on CPU) with its own
`scripts/<name>.sbatch`. The three parameter reports draw 8192 windows per band from
`jr.split(jr.key(0), 8192)`, so their windows are the same draws. Their window centres follow the catalogue's f₀
marginal. response-span draws 65536 single full-B sources with f₀ log-uniform, reweighted to that
marginal, and integrates A out exactly; source-count reuses those draws.
