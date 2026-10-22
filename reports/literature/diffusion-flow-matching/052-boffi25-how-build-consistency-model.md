# How to build a consistency model: Learning flow maps via self-distillation

Boffi et al., 2025, arXiv preprint · [arXiv:2505.18825](https://arxiv.org/abs/2505.18825)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 120 citations in 1.4 yr
(Semantic Scholar, 2026-10-19).

## What they do

- Tangent condition: with X = x + (t−s)v_{s,t}, the diagonal v_{t,t} equals the drift b_t. So v_{t,t} is trained with plain flow matching on the diagonal.
- The off-diagonal is trained by self-distilling the model's own diagonal through one of three losses: Lagrangian (LSD), Eulerian (ESD) or progressive/semigroup (PSD).
- This turns any distillation scheme into direct training from scratch with a single network. CM, CTM, Shortcut, MeanFlow and AYF are shown to be special cases: MeanFlow and AYF are Eulerian, Shortcut is a discretised PSD.
- W₂ bounds hold for LSD and ESD. The authors could not prove one for PSD, citing compounding error and distribution shift from bootstrapping small steps.
- **Key results with numbers** (fixed training time, LSD / PSD-M / PSD-U FID by NFE):
| dataset | NFE | LSD | PSD-M | PSD-U |
|---|---|---|---|---|
| CIFAR-10 | 1 | 8.10 | 12.81 | 13.61 |
| CIFAR-10 | 4 | 3.34 | 5.96 | 6.03 |
| CelebA-64 | 1 | 12.22 | 19.6 | 18.8 |
| CelebA-64 | 16 | 1.96 | 5.09 | 5.63 |
| AFHQ-64 | 1 | 11.19 | 18.9 | 14.5 |
| AFHQ-64 | 16 | 5.61 | — | — |
- ESD diverged on every image dataset in every stopgrad configuration tried, because of large parameter-gradient norms from backpropagating through spatial JVPs (Fig. 4). On the checkerboard it trains but blurs the boundaries.
- LSD saturates at about 4 steps on the checkerboard.

## Implementation details

- The diagonal fraction η = 0.75 of each batch goes to the flow-matching loss and the rest to distillation; η also sets the cost per step.
- The stopgrad goes on the teacher side: LSD regresses ∂_t X_{s,t}(I_s) onto sg(v_{t,t}(X_{s,t}(I_s))) (Eq. 94).
- The teacher uses the instantaneous parameters, not EMA. An EMA teacher "offered no gain and sometimes led to instability"; EMA is used only for evaluation.
- A learned log-variance weight w_{s,t} (an EDM2-style uncertainty weight in two times) "significantly stabilizes" training and allows a larger learning rate.
- PSD: drop the (t−s)² factor and write the loss in v-space.
- Embed s and (t−s) rather than s and t, and add them; positional rather than Fourier time embeddings (Fourier was less stable, following sCM).
- RAdam at lr 1e-2 with sqrt decay after 35k steps; grad clip 1.0; EMA 0.9999.
- A CFG flow map is possible by also sampling the guidance scale.

## Caveats

- Small budgets and minimal tuning.
- No comparison to the state of the art or against a pure-FM baseline at equal compute.
- Unconditional only.
- The ESD failure may partly reflect missing sCM-style stabilisers (tangent normalisation, warmup).
