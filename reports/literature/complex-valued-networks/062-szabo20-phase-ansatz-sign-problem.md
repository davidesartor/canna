# Neural network wave functions and the sign problem

Szabó, Castelnovo, 2020, Phys. Rev. Research 2, 033075 ·
[arXiv:2002.04613](https://arxiv.org/abs/2002.04613)

**Evidence: A.** Peer-reviewed (PRR); 100 citations in 6.7 yr (Semantic Scholar, 2026-10-19).

## What they do

The paper diagnoses why convolutional NQS fail on states with nontrivial sign structure,
and proposes a phase ansatz built for it. The test case is the J1–J2 Heisenberg model on a
10×10 lattice, at J2 = 0 and J2/J1 = 0.5. The wavefunction splits into two real-valued
networks, log ψ = A(σ) + iΦ(σ):

- **A**: a 6-layer real CNN with ReLU and a final ln|x| that outputs the log-modulus.
- **Φ, the "phasor sum"**: a single layer computing

  Φ(σ) = arg Σ_{n,r} exp(i b_n + i Σ_{r′} w_{n,r−r′} σ_{r′})

  with real convolutional weights and 24 replicas. "The activation layer is replaced by
  summing the convolutional output as phasors": every (filter, position) term is a unit
  complex number, and the phase is the argument of their coherent sum.

**Training has two stages**, both with stochastic reconfiguration on 5000 Metropolis
samples:

1. Phase only, with amplitudes frozen to uniform (A ≡ 0): 2 × 10⁴ steps.
2. Joint, starting from the stage-1 phase: 6 × 10⁴ steps.

The rationale: optimal sign structures depend only weakly on the amplitudes, so a converged
phase is a good warm start.

## What they find

- Ordinary real CNNs and deep nets "struggle to converge to ground states with a nontrivial
  sign structure". Their Marshall-adjusted phases come out "essentially random". They also
  grow "unphysically strong peaks" in the amplitude, after which Monte Carlo sampling cannot
  recover.
- They offer a hypothesis for why. Deep nets have "an inductive bias for functions that
  only change significantly upon large-scale changes of the input", while a sign structure
  flips under local changes.
- Results with the phasor-sum ansatz:
  - J2 = 0: E = −0.671275(5) per site against exact −0.671549(4), a relative error of
    about 4e-4. It needs 5145 parameters, while the reference CNN needs 7676 and has the
    exact Marshall rule hard-coded.
  - J2/J1 = 0.5: −0.494757(12) against a best reference of −0.49755(1).
- **The frustrated case converges to the wrong attractor.** The optimiser "homes in" on the
  Marshall sign rule, although the exact ground state is estimated not to obey it (average
  Marshall-adjusted sign ≈ 0). The energy alone did not reveal this, so they warn against
  using "the energy as the only criterion".
- Suggested ways forward:
  - updates that allow "large yet controlled changes" to the wavefunction;
  - explicitly imposed symmetries;
  - the interpretable phase ansatz as a design tool.

## Implementation details

- The S matrix has blocks that differ by up to eight orders of magnitude between the
  amplitude and phase parameters. Each block is regularised by 1e-5 × its mean diagonal.
- Learning rates are η = 0.01 → 0.05 in stage 1 and 0.001 → 0.05 in stage 2.
- The phase is not initialised to the Marshall pattern but to a ferromagnet-like guess.

## Caveats

- One model on one lattice size. The phasor sum is a hand-designed inductive bias for spin
  sign structures, and it is not shown to transfer.
- The failure analysis is qualitative. There is no gradient-variance or Hessian study (see
  Bukov et al. 2021 for that).
- Details are from the ar5iv HTML.
