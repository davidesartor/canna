# Equivariant flow matching

Klein et al., 2023, NeurIPS 2023 · [arXiv:2306.15030](https://arxiv.org/abs/2306.15030)

**Evidence: A.** Peer-reviewed (Neural Information Processing Systems); 171 citations in 3.3 yr
(Semantic Scholar, 2026-10-19).

## What they do

- OT-CFM with a small batch cannot find good pairs when the target has N! symmetric copies: the number of permutations grows as N!, the number of batch pairs only as B².
- The fix is to replace the cost with the orbit-minimised cost c̃(x₀,x₁) = min_{g∈G} ‖x₀ − ρ(g)x₁‖².
- **Thm 1:** for G-invariant measures and a compact isometric G, the OT map is G-equivariant and is also OT for c̃. Training pairs are therefore moved along their orbits to the minimal-cost representative.
- Pair this with an equivariant vector field, here an EGNN.

## What they find

- **LJ13:**
  - OT-FM: −16.09 / 54.4% / 2.84.
  - Equivariant OT-FM: −16.07 / 58.0% / **2.15**.
- **LJ55:**
  - OT-FM: −88.45 / 3.7% / 7.53.
  - Equivariant OT-FM: **−89.27 / 4.4% / 3.52**.
  - The batch transport cost is ~10× lower, and paths are half as long.
  - A fixed-step rk4 with 20 steps suffices, a 3× speed-up in samples per NFE.
- **Alanine dipeptide (22 atoms, few identical):**
  - Plain OT-FM wins on NLL (−107.54 vs −106.78) and on ESS (1.41% vs 0.69%).
  - Equivariant OT-FM still gives shorter paths (9.46 vs 10.19).
  - **So alignment helps most when there are many identical particles, and can hurt otherwise.**
- **DW4 (4 particles, i.e. 4! = 24 permutations, like ours):**
  - ESS: OT-FM 92.4% vs equivariant OT-FM 88.7%; naive FM gets 93.0%.
  - Path length is 2.94 vs 2.92 vs 3.41.
  - With 4 identical items the alignment buys almost nothing.
- **A non-equivariant MLP vector field failed on every system except DW4.**

## Implementation details

- The exact min over S(N)×O(D) is approximated sequentially: Hungarian for the permutation, then Kabsch for the rotation.
  - The other orders and repeated alternation were tried (A.9). Rotation-first is worse; alternating helps only slightly.
- **Cost:** the alignment is computed for every element of the B×B cost matrix, which is expensive: 20 s per batch of 256 on one CPU for LJ55.
  - So batches are pre-reordered offline or in parallel on CPUs.
  - For alanine they reduced the batch to 32 with on-the-fly reordering, which took 25 h of training.
- Mean-free prior plus an update that conserves the centre of mass gives translation invariance.
- A permutation-equivariant flow trained without alignment still shows no path crossings. The paths are curved but shorter than the training pairs (A.1).
- Training: batch 256, lr 5e-4 then 5e-5, 200–1000 epochs.
- ESS uses Kish's formula. Reweighting gives unbiased observables (free energy 4.10±0.08 vs umbrella 4.10±0.26).

## Caveats

- Unconditional. Base is a Gaussian, not a prior draw.
- Gains appear only at large N (LJ55). At N = 4 (DW4) they are nil or negative.
- The theory assumes the target is G-invariant. An **ordered** (identifiability-constrained) target is not, so Thm 1 does not cover "align to an ordered target".
