# Learning the ground state of a non-stoquastic quantum Hamiltonian in a rugged neural network landscape

Bukov, Schmitt, Dupont, 2021, SciPost Phys. 10, 147 ·
[arXiv:2011.11214](https://arxiv.org/abs/2011.11214)

**Evidence: B.** Peer-reviewed (SciPost Physics); 68 citations in 5.9 yr (OpenAlex,
2026-10-19; Semantic Scholar was rate-limiting). That is about 12 per year, short of the A
threshold.

## What they do

They run variational Monte Carlo with neural quantum states (NQS) on the frustrated J1–J2
Heisenberg model on a square lattice (4×4 and 6×6) at J2/J1 = 0.5. They compare two
representations of the complex wavefunction log ψ(σ) = log|ψ| + iφ:

1. **One holomorphic complex network**: complex weights and holomorphic activations, so a
   single network outputs log-amplitude and phase together.
2. **Two decoupled real networks**: one for log|ψ|, one for φ.

They then try to locate the bottleneck with three tests:

- **Partial learning problems**: train the phase network against exact amplitudes, and the
  amplitude network against exact phases.
- **Full-basis gradients**: exact gradients and S-matrix over about 15.8M states, so there
  is no Monte Carlo noise.
- **Hessian spectra** at the end of training.

## What they find

- **They favour two decoupled real networks.** Three reasons:
  - The Cauchy–Riemann constraint ties the phase gradient to the amplitude gradient:
    "fine-tuning of, e.g., only the phases is not possible using a holomorphic ansatz."
  - Holomorphic non-linearities are unbounded, since every non-constant entire function is.
    Their poles give divergent gradients, so the complex nets here use quartic-polynomial
    activations.
  - Unbounded log-amplitudes lead to runaway instabilities.

  Without the Marshall sign rule built in, "the holomorphic ansatz fails to find a good
  approximation to the ground state even for deeper networks."
- **The phase gradient is drowned out.** Early in training the log-amplitude contribution
  to the force is about two orders of magnitude larger than the phase contribution.
- **A simple sign pattern acts as an attractor.** "The Marshall-Peierls sign rule appears to
  be a universal attractor (in the optimization dynamics) even for unbiased networks."
  - In the exact 6×6 ground state, about 50% of configurations violate the rule, but they
    carry only 1.95% of the norm.
  - Sampling from |ψ|² therefore rarely visits them, and the training signal for their sign
    is weak.
- **The bottleneck is the landscape, not expressivity or sampling.**
  - Phase-only training with exact amplitudes still stalls on a plateau near the Marshall
    rule.
  - Full-basis gradients reach about the same energy as MC sampling.
  - The Hessian has large negative eigenvalues, about 10², along sparse directions, so the
    nets have not reached a minimum.
  - Seeds and optimizers end at very different saddles. In one 4×4 run, round-off at about
    1e-15 is enough to change the trajectory.

## Implementation details

- The amplitude network's output passes through a bounded tanh (a = 8, learned shift),
  giving a log-amplitude range of 16. That covers ratios up to about 10⁷ and stops runaway
  growth.
- First layer ln cosh(z), which is even and so respects spin inversion. Deeper layers use
  z − tanh(z)/2, which is odd and linear near 0.
- The input is symmetrised. Training is stochastic reconfiguration with Runge–Kutta
  adaptive step control, which is much more stable than fixed-step SGD.
- In some runs the Marshall sign is supplied through a gauge choice. The unbiased phase nets
  learn it without that.

## Caveats

- One Hamiltonian at one coupling. The authors say their data "does not predict whether
  this effect is a generic, Hamiltonian-independent feature."
- The objective is VMC energy minimisation with samples from |ψ|², not supervised
  regression. The "weak signal on low-weight configurations" mechanism depends on that
  sampling.
- The details above come from the first half of the ar5iv HTML; the conclusions section
  was only partly read.
