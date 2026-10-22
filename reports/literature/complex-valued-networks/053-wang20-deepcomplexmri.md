# DeepcomplexMRI: Exploiting Deep Residual Network for Fast Parallel MR Imaging with Complex Convolution

Wang et al., 2020, Magnetic Resonance Imaging ·
[arXiv:1906.04359](https://arxiv.org/abs/1906.04359) ·
[DOI:10.1016/j.mri.2020.02.002](https://doi.org/10.1016/j.mri.2020.02.002)

**Evidence: A.** Peer-reviewed (Magnetic Resonance Imaging); 218 citations in 6.5 yr
(Semantic Scholar, 2026-10-19).

## What they do

The paper proposes a residual complex-valued CNN for multi-coil (parallel) MRI reconstruction
from undersampled k-space.

- The convolutions are complex "to take into account the correlation between the real and
  imaginary parts", rather than treating Re and Im as independent channels.
- A k-space data-consistency step is inserted between network blocks.

## What they find

- The method beats the state-of-the-art parallel-imaging and deep-learning reconstructions it
  is compared against, per the abstract, in both image quality and speed.

## Implementation details

- Complex convolutions, residual blocks, and interleaved data consistency in k-space.
- The input is complex multi-coil data.

## Caveats

- **Abstract-only.** The full PDF was not read. No numbers are reported here, and **whether a
  parameter-matched real (Re, Im) twin was compared is unknown**; the gain over other methods
  may reflect architecture and data consistency rather than complex arithmetic.
