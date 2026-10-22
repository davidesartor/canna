# Complex-Valued Convolutions for Modulation Recognition using Deep Learning

Krzyston et al., 2020, IEEE ICC Workshops 2020 ·
[DOI:10.1109/ICCWorkshops49005.2020.9145469](https://doi.org/10.1109/ICCWorkshops49005.2020.9145469)

**Evidence: B.** Peer-reviewed (IEEE ICC Workshops); 28 citations in 6 yr (Semantic Scholar,
2026-10-19; OpenAlex 44).

## What they do

The paper observes that a standard real CNN on I/Q stacked as a 2-row image does not compute
the complex product. Its convolution kernels treat I and Q as separate rows. The authors
add a linear combination of the real convolution outputs that reproduces the complex
convolution (Re·Re − Im·Im, Re·Im + Im·Re) inside an otherwise real modulation classifier.

## What they find

- Enabling complex convolution raises classification accuracy across SNR "by up to 35%",
  per the abstract.

## Implementation details

- Complex convolution is emulated with real convolutions plus a fixed linear recombination.
- I/Q input in the RadioML style.

## Caveats

- **Abstract-only.** The "up to 35%" is a peak over SNR, not an average. The SNR at which it
  occurs, the baseline accuracy and the parameter counts were not checked.
- The baseline is a small CNN whose first layer cannot mix I and Q, which is a weak real
  architecture. The gain may say more about that first layer than about complex networks.
- Workshop paper.
