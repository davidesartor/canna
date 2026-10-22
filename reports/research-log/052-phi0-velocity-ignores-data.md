# F52 (2026-10-18): the φ₀ velocity ignores the data; φ₀ marginals look close only because the MCMC's are broad by projection

*Corrected by [F56](056-phi0-learned-without-template.md):* the measurements stand, the mechanism does not. The
helicity coordinates, with no phase reference, made φ₀ learnable (B-stft-h).

- The flow is uniform in φ₊ and φ₋ in every run, while the MCMC pins one of them.
- The φ₀ output is a function of x_t and t alone.
- Next: hand the network (or a sampler) the phase reference instead of having it learn one.

User's observation: on corner pages (e.g. `q0.10-pooled`) φ₀ "gets close" to the MCMC but is
wider, and the ψ–φ₀ correlation is absent.

**1. Measured combinations** (`reports/diagnostics/draw-widths/compare_draws.py at 09abb90`; circular widths, flow /
MCMC). The data measure φ₋ = φ₀ − 2ψ (weight (1 + sin ι)²) and φ₊ = φ₀ + 2ψ (weight (1 − sin
ι)²), F49.
- The MCMC pins one above SNR ~17: 0.001 rad on B's loudest, 0.03–0.13 rad elsewhere. The flow
  is uniform in both (2.4–3.7 rad): 29–2248× on B, 15–1469× on XS.
- φ₀ = φ∓ ± 2ψ inherits ψ's spread, so the MCMC's φ₀ is broad (0.6–1.0 rad) and the flow's
  uniform φ₀ lands within 1.4–2.6× of it. Where ψ is narrow (B q0.10 s0, XS q0.10 s0–s2) the
  gap shows: 19–41×.
- ψ (mod π/2) is learned, 0.9–2.0× on B. So the pages show a correct ψ, an independent uniform
  φ₀, and no ridge.

| B-stft | SNR | sin ι | φ₀ | ψ | φ₊ | φ₋ |
|---|---|---|---|---|---|---|
| loudest | 1668 | 0.96 | 2.6 | 1.8 | 2.1 | **2248** |
| q0.10 s0 | 78 | −0.40 | 32 | 1.7 | **101** | 29 |
| q0.10 s1 | 68 | 0.70 | 2.1 | 1.1 | 2.1 | **67** |
| q0.10 s2 | 61 | −0.89 | 1.4 | 1.1 | **65** | 1.6 |
| q0.10 s3 | 17 | −0.83 | 1.4 | 1.3 | **29** | 1.4 |

B-rope is the same to within 10%.

**2. The φ₀ velocity ignores the data** (the rotation test of
[`velocity_probe.py`](../posterior-eval/flow-vs-reference/scripts/velocity_probe.py), B-stft and B-rope;
rms velocity change over target rms, 512 training samples).
- Rotating the window by e^{iα} (shifts only φ₀ in the true posterior): φ₀ velocity changes
  0.001, the others 0.02–0.13.
- Swapping in another sample's data: φ₀ 0.01, the others 0.6–1.25.
- The rest is 85–95% invariant to the global phase: it reads amplitude-like features, enough
  for f₀, A, sky, ψ (via antenna-pattern amplitudes) and ι.

**Why, most likely.** Each bin's phase is φ₀ plus sub-bin f₀, chirp, Doppler and polarisation
terms, each broad over the prior, so nothing correlates with φ₀ until the data's phasor meets a
reference accurate to a fraction of a radian over two years. No partial credit on the way (a
parity-like plateau), unlike every other parameter. F51's representation and F48's decay don't
touch this. XS-fd escaped once (F48); which difference did it (2-bin tokens, Set coupling, no
decay) is open.

**Next: give the phase reference.**
1. Analytic extrinsics (training-free): per flow draw of (f₀, ḟ, sky) the four F-statistic
   templates make the likelihood Gaussian in four amplitudes; sampling them,
   importance-weighted by the (A, ι, ψ, φ₀) prior, gives φ±, the ridge and the amplitudes
   exactly.
2. Network inputs: matched-filter phasors ⟨y, h_k(x_t)⟩ at x_t's intrinsics, float32, eight per
   source, with F49's helicity coordinates; φ∓ is a linear read-out once x_t's f₀ is within a
   bin.
