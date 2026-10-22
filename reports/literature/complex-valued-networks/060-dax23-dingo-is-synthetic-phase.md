# Neural Importance Sampling for Rapid and Reliable Gravitational-Wave Inference

Dax, Green, Gair, Pürrer, Wildberger, Macke, Buonanno, Schölkopf, 2023, Phys. Rev. Lett. 130,
171403 · [arXiv:2210.05686](https://arxiv.org/abs/2210.05686)

**Evidence: A.** Peer-reviewed (PRL); 102 citations in 4.0 yr (Semantic Scholar, 2026-10-19).

## What they do

DINGO-IS. A DINGO network (an NPE normalizing flow with GNPE time alignment) acts as an
importance-sampling proposal. Each draw is reweighted by likelihood × prior / proposal, which
gives an unbiased posterior, an evidence estimate and a sample efficiency ε, the diagnostic
for a failing network. They run it on 42 BBH events from GWTC-2/3 with IMRPhenomXPHM and
SEOBNRv4PHM, which include higher modes and precession.

The part that matters here is the **phase parameter φc**: the binary's phase at a fixed
reference frequency. The network does not sample it. φc is filled in "synthetically" after
the flow has drawn every other parameter θ̃:

1. Evaluate the likelihood on a uniform grid of 5001 φc values in [0, 2π], with the waveform
   modes cached so that changing φc only re-weights them.
2. Interpolate between grid points and sample φc from p(φc | θ̃, d).
3. Fold the conditional's density into the importance weight.

## What they find

- On the network and phase, verbatim: "we found that Dingo often has difficulty resolving
  the phase parameter ϕc". The supplement says it "often infers the prior instead".
- They give two reasons. The conditional p(φc | θ̃, d) "is typically very tightly
  constrained", and "ϕc is strongly correlated with θ̃". A prior-wide proposal for φc
  therefore "leads to a substantially reduced sample efficiency."
- Synthetic phase is "similar to standard phase marginalization". Unlike analytic
  marginalisation, it stays valid with higher modes and *recovers* φc rather than
  integrating it out.
- Headline results: median ε ≈ 10.9% (XPHM) and 4.4% (SEOBNRv4PHM), about two orders of
  magnitude above standard samplers. The log-evidence has about 10× less statistical
  uncertainty than with Bilby-dynesty. For GW150914 they get −15831.87 ± 0.01, against
  Bilby's −15831.78 ± 0.10.

## Implementation details

- The network is the GNPE DINGO of Dax et al. 2021, conditioned on whitened
  frequency-domain strain split into real and imaginary parts. It is a real-valued
  embedding network followed by a neural spline flow, with no complex layers.
- φc enters each (ℓ, m) mode as a known phase factor, so a dense grid is cheap once the
  modes are cached. This is standard waveform structure; the paper does not dwell on it.
- The synthetic step uses the likelihood, so it is part of the inference pipeline and not
  a network output.

## Caveats

- This is a negative result about learning a phase. It comes from one network family (a
  real-valued embedding into a flow) and is a remark, not an ablation: they do not test
  other architectures, complex inputs or longer training for φc.
- They give no quantitative measure of how "often" the network infers the prior.
- The fix needs likelihood evaluations at inference time, so it is not available to a
  flow-only pipeline.
- Read from the arXiv HTML (v2), including the supplement passages quoted above.
