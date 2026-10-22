# ℱ-statistic search for white-dwarf binaries in the first Mock LISA Data Challenge

Prix & Whelan, 2007, CQG 24 S565 · [arXiv:0707.0128](https://arxiv.org/abs/0707.0128)

**Evidence: B.** Peer-reviewed (Classical and Quantum Gravity); 17 citations in 19.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

A monochromatic GB signal in each TDI channel is exactly linear in four amplitudes, $h^I(t)=\sum_{\mu=1}^4 \mathcal A^\mu h^I_\mu(t)$, with $h_{1,3}=a^I(t)\{\cos,\sin\}\phi(\tau)$, $h_{2,4}=b^I(t)\{\cos,\sin\}\phi(\tau)$ ($a,b$ = antenna patterns, depending only on sky + f), and $\mathcal A^\mu(h_0,\iota,\psi,\phi_0)$ given by Eq. 4 (e.g. $\mathcal A^1=A_+\cos\phi_0\cos2\psi-A_\times\sin\phi_0\sin2\psi$). The log-likelihood ratio is quadratic: $\ln L=\mathcal A^\mu x_\mu-\tfrac12\mathcal A^\mu\mathcal M_{\mu\nu}\mathcal A^\nu$ with $x_\mu=(x|h_\mu)$, $\mathcal M_{\mu\nu}=(h_\mu|h_\nu)$; ML amplitudes $\mathcal A_{cand}=\mathcal M^{-1}x$, and $\mathcal F=\tfrac12 x^T\mathcal M^{-1}x$. Search only over Doppler params $\{\beta,\lambda,f\}$ with LIGO code (ComputeFStatistic_v2) in the long-wavelength limit (LWL).

## What they find

MLDC 1.1.1 single sources: Δf = 1.7/0.8/0.2 nHz and sky error 34.8/7.1/4.4 mrad at 1.1/3.0/10.6 mHz — Doppler params fine even where LWL fails. Amplitude 4-vector error |ΔA|/2 (in σ units, expected O(1)) = 1.5 / 9.6 / 108.7, angle φ_A = 0.06/0.53/2.2 rad: the LWL wrecks amplitudes above ~3 mHz. 1.1.3 blind: 17/20 recovered; confusion challenges (45 sources in 15 μHz) fail — secondary maxima within ~1.4e-4 f at wrong sky positions are indistinguishable from other sources.

## Implementation details

Under noise, $\Delta\mathcal A\sim\mathcal N(0,\mathcal M^{-1})$ exactly, i.e. the amplitude posterior (flat prior in $\mathcal A^\mu$) is a 4-D Gaussian with constant covariance $\mathcal M^{-1}$ given Doppler params; $E[2\mathcal F]=4+\rho^2$ (bias correction). Grid: sky mesh $d\alpha=\sqrt{2m}/(2\pi f R_{orb}/c)$, $df=\sqrt{12m}/(\pi T)$, mismatch m=0.3; SFT length 7 days; use uncorrelated TDI combos (X, Y−Z). Each signal produces secondary F-stat maxima within ~1e-4 f at different sky positions.

## Caveats

LWL only; no Bayesian posteriors, no phase-specific analysis; MLDC-1 sources with ḟ=0.
