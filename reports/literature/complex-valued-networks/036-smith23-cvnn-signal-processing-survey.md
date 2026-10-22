# Complex-valued neural networks for data-driven signal processing and signal understanding

Smith, 2023, arXiv preprint · [arXiv:2309.07948](https://arxiv.org/abs/2309.07948)

**Evidence: C.** Unreviewed preprint with 16 citations in ~3 yr (Semantic Scholar,
2026-10-19).

## What they do

A survey-plus-library paper for RF/radar/communications ("RF signal understanding"), mainly
1-D data, built around `complextorch`, a thin PyTorch wrapper. Catalogues the complex layers
and activations a signal-processing user needs, including **complex attention**.

## What they find

- **Layers**: complex linear/conv (1-D/2-D/3-D, transposed), implemented with real PyTorch
  ops and **Gauss's trick** (3 real multiplications/convolutions per complex product instead
  of 4); complex scaled dot-product **multi-head attention**, plus efficient/multi-channel
  attention variants (CV-ECA, CV-MCA); batch and layer norm by 2×2 **whitening** ("necessary
  to ensure a circular distribution with equal variance"); pooling and dropout on Re/Im
  separately; manifold (wFM) convolutions.
- **Softmax/masking options for complex scores**: split (Type-A) softmax on Re and Im,
  **phase-preserving softmax** (softmax of the magnitude, phase kept), magnitude softmax,
  min-max magnitude normalisation, complex ratio mask.
- **Activations**: split Type-A (split tanh/sigmoid/abs, CReLU, CPReLU), polar Type-B (polar
  tanh/squash/log, modReLU — magnitude maps with phase unchanged), and fully complex (complex
  sigmoid, zReLU, cardioid, siglog).
- **Losses**: split (L1, MSE, SSIM) and polar (noted as largely unexplored).

## Implementation details

No experiments, no complex-vs-real comparison, no parameter matching; precision (fp16/bf16)
and optimisers not discussed (PyTorch's native complex autograd is relied on).

## Caveats

- A software overview, not evidence: the "superior modeling performance" claim rests on
  citations. Some descriptions of the phase-preserving softmax and complex ratio mask are
  internally inconsistent on the page.
- Useful mainly as a checklist of building blocks for a complex transformer (attention score
  → softmax choice, whitening norm).
