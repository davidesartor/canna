# Complex-Valued Convolutional Neural Network and Its Application in Polarimetric SAR Image Classification

Zhang et al., 2017, IEEE Transactions on Geoscience and Remote Sensing ·
[DOI:10.1109/TGRS.2017.2743222](https://doi.org/10.1109/TGRS.2017.2743222)

**Evidence: A.** Peer-reviewed (IEEE TGRS); 622 citations in 9 yr (Semantic Scholar,
2026-10-19; OpenAlex 677).

## What they do

The paper builds a CNN in which every element is complex: inputs, weights, biases, the
convolution, the activation and pooling. It is trained by complex-domain backpropagation (the
gradient is taken with respect to Re and Im of each weight). The input is the complex
polarimetric coherency/covariance data of a PolSAR image, which carries the relative phase
between polarisation channels. The task is per-pixel land-cover classification.

## What they find

- On the benchmark PolSAR scenes **Flevoland** and **Oberpfaffenhofen**, the CV-CNN's
  classification error is **lower than that of a real-valued CNN "with the same degrees of
  freedom"**, according to the abstract.
- Overall accuracy is "comparable" to state-of-the-art PolSAR classifiers.
- The abstract attributes the gain to keeping the phase information of the complex data,
  rather than reducing it to real features.

## Implementation details

- The network is complex end-to-end, up to a real-valued classification output.
- The complex backpropagation is derived in the paper.
- The real baseline is matched in degrees of freedom, i.e. in real parameter count.

## Caveats

- **Abstract-only.** The full text was not read, so no accuracy numbers, layer sizes or
  activation choices are given here, and how the real baseline encodes the complex input was
  not checked.
- Classification on a few small benchmark scenes; the labels are spatially correlated.
