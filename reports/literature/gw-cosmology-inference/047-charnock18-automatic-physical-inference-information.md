# Automatic physical inference with information maximising neural networks

Charnock, Lavaux, Wandelt, 2018, PRD 97, 083004 · [arXiv:1802.03537](https://arxiv.org/abs/1802.03537)

**Evidence: A.** Peer-reviewed (Physical Review D); 104 citations in 8.7 yr (Semantic Scholar,
2026-10-19).

## What they do

IMNN: a network f: d → x (one output per parameter) trained only on simulations at a *fiducial* θ_fid, to maximise the Fisher information of its outputs under an assumed Gaussian likelihood of the summaries: F = ∂μ_fᵀ C_f⁻¹ ∂μ_f, with C_f the sample covariance of n_s fiducial-sim outputs and ∂μ_f from seed-matched finite-difference sims at θ_fid ± Δθ (or adjoint gradients of the simulator). Loss Λ = −½|F|², plus a |C_f| penalty to fix the arbitrary linear scale of the summaries. The trained summaries are then used for PMC-ABC with Fisher-metric distance ρ = √((x−x_o)ᵀF(x−x_o)).

## What they find

Gaussian variance (n_d=10, F_max=5): IMNN reaches F=5.15±0.39 (slight over-fit to fluctuations) with 1000 sims + 2×100 derivative sims; output ≈ Σd² + 58 (learned the sufficient statistic up to monotone map); PMC-ABC posterior matches analytic. Known noise: F≈1.25 (=max). Unknown noise σ²∈(0,2]: F=0.9 (vs 5/9 if σ²=2 known), posterior matches Rao–Blackwellised analytic. Wrong fiducial (θ_fid=2 vs truth 1): posterior still equals analytic — the network learned a θ-independent function. Lyα (581 pixels, A_s): F=0.015, wide posterior from one quasar. **LISA sine-Gaussian burst, f_c (SNR 34, 2048 time samples):** MOPED (linear, ∝∂μ/∂f_c) gives a spurious aliasing peak in the summary likelihood; the IMNN summary (5×10 net) gives a single peak at f_c=0.1 Hz.

## Implementation details

Architecture barely matters for the toy (from [5,5,5,5,5] to [2048,2048]), only convergence speed; He init, bias 0.1, leaky-ReLU α=0.01, dropout 10–50%, two "batches" of fiducial sims re-shuffled each epoch so μ_f and C_f fluctuate (acts as regulariser). Many epochs (800–10000). Seed matching for derivatives is essential to suppress variance. Gaussian approximation ln L ≈ −½(x−μ_f)ᵀF(x−μ_f) gives a quick asymptotic posterior that is right near the peak only.

## Caveats

Only 1-parameter problems; ABC downstream; Fisher information is a local (fiducial-point) criterion, so global optimality across a broad prior is not guaranteed; slight Fisher over-estimation on training sims. The "insensitive to fiducial" claim is shown only for the variance toy.
