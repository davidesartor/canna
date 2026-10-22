# F48 (2026-10-17): B-long is the best B model; φ₀ broke with the F39 recipe, most likely through the weight decay, not the path

- **B-long** (600k steps) is 0.75–0.88× B-rope's f₀ width in every band, halves the loudest
  source's sky, and replaces B-rope as the B model. φ₀ unchanged.
- **φ₀ was learned before F39** and has not been since. The φ₀ output rows of the AdamW head
  have decayed to 0.08–0.14× the other rows: most likely the decay on z introduced in F39.

**B-long** (14706580 → 14706581 + post 14706582): link 1 hit the 24 h limit at ~epoch 462, link
2 finished (7 h 23 min), post 38 min. Local as `outputs/21-lisa-B-long`. Held-out x 0.6202 vs
B-rope's 0.6399, still falling (0.6231 at epoch 500). Figures:
`outputs/plots/f48_long_and_phi0.pdf`.

| 240 B sources (x) | found | 7–15 | 15–40 | 40–100 | 100–300 | ≥ 300 | rank / in95 |
|---|---|---|---|---|---|---|---|
| B-rope | 215 | 0.91 | 0.055 | 0.037 | 0.037 | 0.036 | 0.51 / 1.00 |
| **B-long** | **221** | **0.20** | **0.047** | **0.031** | **0.031** | **0.031** | 0.50 / 1.00 |

- Calibration unchanged.
- **Against the MCMC:** loudest source's sky λ 66 → 30×, sin β 106 → 44×, ψ 2.3 → 1.7×. Mid-SNR
  sources within 1.0–1.6× in A, sky, ψ, ι and 1.6–3.6× in f₀.
- Training length is part of the loud floor.

**φ₀: correction to F47** (the user's objection). The network is conditioned on the data at
every s, so it can infer f₀ and φ₀ from y anywhere on the path; x_t's precision does not matter
to it. F47's shared-schedule argument holds only for B-template's residual. For the plain
networks:
1. **φ₀ was learned before F39.** Against the joint MCMC, R1, XS-late-768-cool and XS-fd gave
   φ₀ widths close to their ψ widths (R1's loudest φ₀ 6.6× vs ψ 6.5×; XS-fd's 9.1× vs 8.1×).
   From XS-sfplus (F39) on, every run gives the same φ₀ numbers to two decimals (20.0, 517,
   25.1 on XS; 47.3, 4.7, 1.8 on B): the prior draws.
2. **XS-fd's network moves φ₀** (checked out at 6a0661c): predicted φ₀ velocity has 39% of the
   target's spread, R² 0.15 at s < 0.5, 0.38 at 0.5–0.9, 0.12–0.16 at 0.9–0.99. Every run since
   F39: 0.3–0.5%.
3. **The information reaches the backbone in both:** a 0.03 rad rotation of the loudest
   source's phase moves the most affected data token by 6.5–8.7 bf16 ulps in B-rope (16-bin
   tokens), 7.6–12 in XS-fd (2-bin tokens). Neither token size nor the bf16 cast hides φ₀.
4. **The φ₀ output weights have decayed away.** Velocity head's last layer (AdamW): cos φ₀ /
   sin φ₀ row norms are 0.88× the other coordinates' in XS-fd, 0.14× in B-rope, 0.08× in
   B-long. All rows are ~4× smaller than XS-fd's (the decay's global shrinkage, F43); the φ₀
   rows a further 6–10× below that.
5. **Likely mechanism:**
   - F39 put a weight decay of 0.05 (per unit lr) on z on every matrix, the AdamW head
     included; XS-fd ran with 1e-6, effectively none (F34).
   - φ₀ is the slowest feature to learn: it needs demodulation against a reference phase
     depending on sub-bin f₀, sky and polarisation, so its gradient starts near zero.
   - The decay pulls the undefended φ₀ rows to zero; with them near zero, the backbone gets no
     gradient to build φ₀ features, so nothing pushes the rows back.
   - Other F39 changes (token size, SNR order, reconstruction head, NorMuon) are not excluded,
     but none gives an obvious mechanism, and the row norms single out the decay.

**Next.** The decay bounds z under Muon's fixed-size steps (F36, SF-NorMuon); that covers the
Muon matrices, not the AdamW parameters, whose steps are scale-adaptive.
- **Proposed run:** B-long's recipe with decay on the Muon matrices only, none on AdamW
  parameters (head, encodings, positions).
- **Readout:** velocity probe on its checkpoints; if this is the cause, the φ₀ rows and φ₀
  velocity spread grow from the first epochs.
