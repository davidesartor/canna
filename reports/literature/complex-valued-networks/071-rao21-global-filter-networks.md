# Global Filter Networks for Image Classification

Rao, Zhao, Zhu, Lu, Zhou, 2021, NeurIPS · [arXiv:2107.00645](https://arxiv.org/abs/2107.00645)

**Evidence: A.** Peer-reviewed (NeurIPS 2021); 805 citations in 5.3 yr (Semantic Scholar,
2026-10-19).

## What they do

GFNet replaces self-attention in a ViT/ResMLP-style block with a **global filter layer**:

X ← F⁻¹[K ⊙ F[X]].

- F is a 2-D FFT over the token grid.
- K ∈ ℂ^{H×W′×D} is a learnable complex filter, one per frequency and channel.
- ⊙ is elementwise.

The layer is a depthwise global circular convolution whose kernel is parameterised directly
in frequency space. It costs O(HWD log HW), log-linear in the number of tokens, against
quadratic for attention. The FFN and the rest of the block stay real.

## What they find

- **ImageNet top-1:** GFNet-Ti 74.6% (7M params, 1.3 GFLOPs), GFNet-S 80.0% (25M, 4.5G) and
  GFNet-B 80.7% (43M, 7.9G). Hierarchical GFNet-H-B reaches 82.9% (54M, 8.4G).
- These are competitive with DeiT/Swin-class models at similar FLOPs.
- The authors also claim good transfer, robustness and efficiency on downstream tasks.

## Implementation details

- **Weight storage:** the reference code keeps K as a real `nn.Parameter` of shape
  (h, w, dim, 2) and converts it with `torch.view_as_complex` in the forward pass. The
  optimiser therefore sees two independent reals per complex weight.
- **Init:** N(0, 1) × 0.02.
- **Transforms:** `rfft2`/`irfft2` with `norm='ortho'`. Only the non-negative half of the
  spectrum is parameterised (w = W/2 + 1), so the output is real.
- The filter is tied to the token-grid size (default 14 × 8 in frequency). Other resolutions
  need the filter interpolated.

## Caveats

- **The filter is static.** It is the same for every input and does not depend on the
  content, unlike attention. Data-adaptive spectral mixing was added in follow-ups such as
  AFNO (Guibas et al. 2021), which uses a block-diagonal complex MLP per mode with soft
  thresholding.
- Vision classification only. Phase handling of the input spectrum is not studied: the
  layer is linear and shift-equivariant by construction.
- The model table is from the GitHub README; the abstract is from arXiv. The full PDF was
  not parsed.
