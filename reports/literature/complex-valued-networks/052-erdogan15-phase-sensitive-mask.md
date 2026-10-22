# Phase-Sensitive and Recognition-Boosted Speech Separation Using Deep Recurrent Neural Networks

Erdogan et al., 2015, ICASSP 2015 ·
[DOI:10.1109/ICASSP.2015.7178061](https://doi.org/10.1109/ICASSP.2015.7178061)

**Evidence: A.** Peer-reviewed (ICASSP); 681 citations in 11.5 yr (Semantic Scholar,
2026-10-19; OpenAlex 667).

## What they do

The paper proposes the **phase-sensitive mask (PSM)**. The network still predicts a real mask
on the magnitude spectrogram, but the training target accounts for the phase difference θ
between clean and noisy STFT, |S|/|Y|·cos θ. Equivalently, the loss measures the error of
the masked complex spectrum against the clean one. This "phase-sensitive SNR-based objective"
replaces a magnitude-only mask target.

- **Network**: deep bidirectional LSTMs.
- **Extra inputs**: speech-recognition alignment information, the "recognition-boosted"
  part.

## What they find

- The phase-sensitive objective gives **uniformly better SDR** than magnitude-only
  objectives, per the abstract.
- Adding recognition alignment features improves the results further.

## Implementation details

- The mask is real-valued and multiplies the magnitude; the noisy phase is reused at
  synthesis.
- Phase enters only through the target or loss, not through the network inputs or outputs.

## Caveats

- **Abstract-only.** No numbers are reported here. The PSM formula above is the standard
  definition later used by Williamson et al. 2016 (file 49), not copied from this paper's
  tables.
- No complex-valued network; this is the earliest of the phase-aware targets in speech
  enhancement.
