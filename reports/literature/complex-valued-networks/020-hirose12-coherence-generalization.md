# Generalization characteristics of complex-valued feedforward neural networks in relation to signal coherence

Hirose & Yoshida, 2012, IEEE Trans. Neural Networks and Learning Systems 23(4):541–551 ·
[DOI:10.1109/TNNLS.2012.2183613](https://doi.org/10.1109/TNNLS.2012.2183613)

**Evidence: A.** Peer-reviewed (IEEE TNNLS); 310 citations in 14 yr (Semantic Scholar,
2026-10-19; OpenAlex gives 299).

## What they do

The paper asks *when* a complex-valued network (CVNN) generalises better than a real-valued
one, and answers in terms of the **coherence** of the signals: how strongly the data behave
like waves with a well-defined phase relation across samples. The task is function
approximation of wave-like data, e.g. interpolating temporal signals from samples. They
compare:

- a CVNN with an **amplitude–phase-type activation** (squash the modulus, keep the phase:
  `tanh(|z|) e^{i arg z}`, Hirose's standard activation);
- a **bivariate** real network (each complex number fed as a (Re, Im) pair, real weights
  mixing them freely);
- a **dual-univariate** real network (Re and Im processed by two independent real
  networks).

Coherence of the training and test signals is varied, and generalisation error is measured
as a function of it, in simulation and in a real-world experiment.

## What they find

Per the abstract: CVNNs with amplitude–phase activations show **smaller generalisation
error** than both real-valued alternatives, and the advantage is tied to signal coherence.
The authors explain it through (i) the **degree of freedom in learning** and (ii) the
**circularity** of the neural dynamics. (The usual form of Hirose's argument, not verified
in this paper's text: a complex weight is a rotation-and-scaling of the plane, 2 real dof,
whereas a real 2×2 block has 4 dof and can shear or reflect, so the CVNN is restricted to
the phase-rotation/amplitude-scaling maps that coherent wave data need; and the
amplitude–phase activation commutes with phase rotation.)

The follow-up (Hirose & Yoshida 2013, Neural Comput. Appl. 22:1357, 12 citations) splits
the error into amplitude and phase parts on a task with phase shifts and amplitude changes
and reports, per its abstract, that CVNNs generalise better than real networks with
**doubled input and output neurons**, "in particular when the signals have high coherence".

## Implementation details

Fully connected, shallow feedforward networks. Amplitude–phase activation
`f(z) = tanh(|z|) exp(i arg z)` (phase-equivariant: `f(e^{iα}z) = e^{iα} f(z)`). The
real-valued baselines receive the same information as stacked (Re, Im). Exact widths,
parameter counts and the coherence definition could not be verified (see Caveats).

## Caveats

- **Abstract-level only**: the full text is paywalled and was not readable; widths,
  parameter matching, the coherence measure and the numeric error curves are not reported
  here. The "doubled inputs/outputs" matching in the 2013 follow-up is a feature-count match,
  under which the real network has *more* parameters than the CVNN.
- Small shallow networks on low-dimensional wave-interpolation tasks; no deep networks, no
  attention, no modern optimisers.
- The claim is conditional: the CVNN's edge is attributed to *high* coherence. The abstract
  does not say what happens at low coherence; the framing implies the gap shrinks.
- Despite its age this is the most-cited statement of the "complex helps when the data are
  coherent waves" hypothesis, and most later surveys cite it for that.
