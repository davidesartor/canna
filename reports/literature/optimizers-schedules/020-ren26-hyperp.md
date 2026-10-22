# Rethinking Language Model Scaling under Transferable Hypersphere Optimization, HyperP / MuonH (Ren et al., 2026)

[arXiv:2603.28743](https://arxiv.org/abs/2603.28743), preprint.

**Evidence: C.** Not peer-reviewed; 12 citations in 0.6 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

- Run Muon on the Frobenius sphere: both the weight and the update are normalised every step.

## What they find

- Weight decay has no first-order effect on the sphere, so the norm/decay interaction goes away.
- The lr transfers across width, depth and data.

## Limits

- LLMs only. The Muon analogue of EDM2's forced weight normalisation
  ([014](014-karras23-edm2.md)).
