# Enhanced Radar Imaging Using a Complex-Valued Convolutional Neural Network

Gao et al., 2018, IEEE Geoscience and Remote Sensing Letters ·
[arXiv:1712.10096](https://arxiv.org/abs/1712.10096) ·
[DOI:10.1109/LGRS.2018.2866567](https://doi.org/10.1109/LGRS.2018.2866567)

**Evidence: A.** Peer-reviewed (IEEE GRSL); 189 citations in 8 yr (Semantic Scholar,
2026-10-19; OpenAlex 209).

## What they do

A complex-valued CNN that turns raw radar echoes into a sharpened image. The setting is
turntable ISAR at THz frequencies (213.6–226.4 GHz).

- **Pipeline**:
  1. A fixed, untrained "fully connected" layer A^H y (implemented by FFT) maps echoes to a
     coarse complex image.
  2. Complex convolutions 25×25×1×6, 15×15×6×12 and 5×5×12×12, each with split cReLU.
  3. A final 3×3×12×1 layer with an "Abs" output neuron that returns the magnitude image.
- **Size**: 47,316 real degrees of freedom.
- **Real baseline**: same depth on 2 channels (Re, Im), widths 6/24/24/1 with ReLU, 54,516
  degrees of freedom (7200 more than the complex net).
- **Training data**: ideal point scatterers with complex Gaussian coefficients. The target is
  a Gaussian point spread function convolved with |o|, so the network is supervised on
  magnitude.
- **Training**: MSE; MATLAB; momentum SGD 0.9, weight decay 0.001, lr 3e-5 (1e-5 for the last
  layer); batch 50; 50k examples; 5 epochs; 16 h on a Titan Xp.

## What they find

- **Table II**, RMSE at −10 / −5 / 0 / 5 / 10 dB, with runtime:

  | Method | −10 dB | −5 dB | 0 dB | 5 dB | 10 dB | Time |
  |---|---|---|---|---|---|---|
  | FFT | 0.1987 | 0.1778 | 0.1705 | 0.1682 | 0.1675 | 0.042 s |
  | SPGL1 (sparse) | 0.0568 | 0.0560 | 0.0568 | 0.0574 | 0.0683 | 27.66 s |
  | RV-CNN | 0.0456 | 0.0308 | 0.0262 | 0.0251 | 0.0248 | 0.083 s |
  | CV-CNN | 0.0434 | 0.0289 | 0.0255 | 0.0247 | 0.0245 | 0.071 s |

- **Size of the gain**: the CV-CNN is slightly better than the real net at every SNR while
  using fewer parameters. The relative gain is about 5% at −10/−5 dB and about 1% at 10 dB,
  i.e. **largest at low SNR**.
- **Cost**: both CNNs are hundreds of times faster than sparse recovery.

## Implementation details

- The physics operator A^H is a fixed first layer, so learning starts from a matched-filter
  image.
- Complex features are kept up to the last layer; the output is |·|.
- The real baseline takes (Re, Im) and has more parameters.

## Caveats

- Trained on idealised point-scatterer simulations. Lab measurements of an airplane model are
  shown only qualitatively.
- Small gains, a single training run, and no reported variance.
- Only magnitude is supervised, so the phase quality of intermediate features is not measured.
