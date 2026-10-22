# Robust and scalable simulation-based inference for gravitational wave signals with gaps

Mao, Lee, Edwards, 2025, arXiv v2 26 Dec 2025, astro-ph.IM · [arXiv:2512.18290](https://arxiv.org/abs/2512.18290)

**Evidence: C.** Not peer-reviewed; 0 citations in 0.8 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

FMPE (Gaussian base N(0,I), linear/OT path, target θ₁−θ₀, `flow_matching` package) conditioned on a summary s(d̂) learned **end-to-end** with the velocity net. Two summarisers: (1) 1D CNN on the raw gapped A-channel time series (4 strided conv blocks, strides 4/6/8/10, channels 64→512, BN+GELU, adaptive pool → 256-d); (2) for long data, 2D CNN on the log-modulus of the WDM wavelet transform (pywavelet, 2¹¹ freq bins, sliced to 1–10 mHz) with **asymmetric (3×9) kernels and geometric temporal dilation 1,2,4,8**, → 512-d. Velocity net: residual MLP for (θ_t, t) embedding, GLU-conditioned on s, then a residual MLP (GELU, dropout). Parameters log-transformed.
single GB-like source, θ = {a, f, ḟ} only; ι, φ0, ψ, β, λ **fixed** for all simulations. fastlisaresponse, equal-arm analytic orbits, TDI-1 A channel only. Noise Gaussian from the A PSD (Re/Im ~ N(0, S_n/4Δf)), synthesised in FD then iFFT. Gaps: durations U[3,12] h, exponential inter-gap times, ~80% duty cycle.

## What they find

- 30 d, Δt = 5 s (N = 518 400): 2×10⁴ clean training signals (+5000 validation) with noise and gap masks redrawn **on the fly**, noise amplitude curriculum (ramped from 0). Priors: ln a ± 0.1, ln f ± 10⁻⁵, ln ḟ ± 10⁻⁵ around truth (f0 = 2 mHz → ±20 nHz, **far below one 1/T bin of 386 nHz**). Injection SNR ≈ 45. 300 epochs, batch 64, Adam 1e-3 (250 ep) → 1e-4 (50 ep), cosine within stages, summariser lr ×5 in stage 1; ~3 h on one L40S. MAF with the same CNN: ~10 h.
- FM vs MAF: both localise f, ḟ; MAF misplaces/underestimates the amplitude a; p-p on 100 injections: FM on the diagonal, MAF clearly off for a, slightly for f, ḟ (overconfident).
- End-to-end vs two-stage (frozen denoising autoencoder encoder + log-std scale feature): two-stage has broader f, ḟ and **fails to recover a**; final validation FM loss 0.38 (end-to-end) vs 0.50 (two-stage).
- 90 d: wavelet pathway, 10⁴ training + 2500 validation signals, offline augmentation (N noise/gap realisations per clean signal, sampled per epoch), priors widened to log₁₀a ± 0.1, ln f ± 10⁻³ (±2 µHz ≈ 31 bins), ln ḟ ± 10⁻³. SNR ≈ 40. 1000 epochs, 5.5 h. Dropout 0.4.
- Kernel ablation (multi-D HPD coverage via KDE): asymmetric (3,9) dilated kernels calibrated; symmetric (3,3) with 512-d or 256-d summary both "significantly miscalibrated" (figure only, no numbers).

## Implementation details

On-the-fly noise/gap redraw over a small bank of clean signals; noise-amplitude curriculum; differential learning rate (summariser ×5) then a reset fine-tuning stage; anisotropic dilated kernels on time–frequency images.

## Caveats

3 parameters, one channel, fixed sky/orientation/phase, no MCMC reference (validation is MAF + p-p on 100 injections + one injection); prior boxes are tiny (30 d case is sub-bin in f); calibration figures only; no numbers for coverage error.
