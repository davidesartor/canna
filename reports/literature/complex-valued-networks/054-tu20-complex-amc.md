# Complex-Valued Networks for Automatic Modulation Classification

Tu et al., 2020, IEEE Transactions on Vehicular Technology ·
[DOI:10.1109/TVT.2020.3005707](https://doi.org/10.1109/TVT.2020.3005707)

**Evidence: A.** Peer-reviewed (IEEE TVT); 264 citations in 6 yr (Semantic Scholar,
2026-10-19; OpenAlex 313).

## What they do

The paper takes three published real-valued modulation classifiers that read raw I/Q samples
and builds complex-valued counterparts of each. The complex versions use complex
convolution, complex batch normalisation, complex weight initialisation and complex dense
layers. Real and complex versions are compared on the RadioML 2016.10A dataset (11
modulations across a range of SNRs).

## What they find

- The complex-valued networks are "superior" to their real counterparts, i.e. more accurate,
  according to the abstract.

## Implementation details

- The building blocks follow Trabelsi et al.'s deep complex networks: complex conv, complex BN
  and complex init.
- The real baselines take I/Q as 2 real channels, which is the RadioML convention.

## Caveats

- **Abstract-only.** No accuracy margins, SNR breakdown or parameter-matching details were
  checked. Whether the complex models were matched to the real ones on real parameters or on
  channel count is unknown.
