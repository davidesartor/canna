# The Unreasonable Effectiveness of Deep Evidential Regression

Meinert, Gawlikowski & Lavin, 2023, AAAI 2023 (vol. 37 no. 8, pp. 9134-9142) · [arXiv:2205.10060](https://arxiv.org/abs/2205.10060)

**Evidence: B.** Peer-reviewed (AAAI); 41 citations in 3.4 yr (OpenAlex, 2026-10-25). Semantic Scholar rate-limited (429).

## What they do

An analysis of Deep Evidential Regression (DER, Amini et al. 2020; file 005), mainly on a synthetic cubic set, a synthetic binary pulse and NYU Depth v2 (ApolloScape as OOD).
- DER outputs γ = θ₁, ν = softplus(θ₂), α = softplus(θ₃)+1, β = softplus(θ₄) (Eq. 6). The NIG marginal is a Student-t with scale w_St = √(β(1+ν)/(αν)), and the projection cannot recover all four parameters. Eq. 5: ∂ log L^NIG/∂ν = 0 along β(ν) ∝ 1/(1+ν⁻¹), a data-independent path with ν → 0.
- So the NLL does not constrain the uncertainty split of Eq. 3. Only the evidence regulariser |y−γ|·Φ (Eq. 4) breaks the degeneracy. The authors note Φ = ν + 2α "would actually be better motivated".
- Epistemic uncertainty u_ep² = β/(ν(α−1)) is a "point-wise convergence speed" proxy, so "only relative changes are conclusive".
- Dynamics (Fig. 1, Fig. 6): the ν-gradient scales with the residual, so early residuals set how small ν gets. γ and w_St gradients create a curved valley that optimisers overshoot, and the regulariser only bites once w_St ≈ 3. The α gradient conflicts with the NLL and stalls near α ≈ 2 on [-4, 4].
- Fixes: u_al′ = w_St and u_ep′ = u_ep/u_al = 1/√ν (Eq. 9-10). A normalised-residual regulariser λ|(y−γ)/w_St|^p Φ with p in {1, 2} (Eq. 11). A Gaussian variant L′ = log σ² + (1+λν)(y−γ)²/σ², σ² = β/ν (Eq. 12).

## What they find

Results are in figures, with no RMSE, NLL or AUROC tables.
- Cubic, true σ = 3 (Fig. 2a): DER's aleatoric term shows a spurious peak of about 0.7 in the plateau region. The redefined u_al′ absorbs most of it (Fig. 2b).
- Naive Eq. 8 (add λ|y−γ|ν to a generic NLL): momentum optimisers drive ν to about 0 "within machine precision even for very small coupling constants λ". DER avoids this only through the w_St coupling, and that depends on λ.
- Binary pulse (Fig. 3c): the Eq. 11 regulariser removes the aleatoric part from the epistemic estimate more cleanly. On the cubic set it shows no significant difference from DER.
- Eq. 12 variant (Fig. 2c, λ = 2): looks like DER, with significantly higher sample variance.
- Depth (Fig. 4): u_ep′ is less helpful for calibration than u_ep, and u_ep ≈ u_al′. A plain NLL-trained normal (λ = 0) is "less well calibrated, yet more powerful" for separating ID from OOD entropy.
- Their conclusion is that DER is "more a heuristic than an exact uncertainty quantification" method.

## Implementation details

- Fig. 1: 50 seeds, 500 epochs, Adam 5e-4, 1k points, σ² = 9, test on [-7, 7], λ = 0.01.
- Fig. 2c: λ = 2, lr 5e-3, θ₃ ignored.
- Fig. 3: 1k points, λ = 0.01, lr 1e-3, 600 epochs.
- Depth: Amini's released scripts at commit d1d8e39, with σ swapped to 1/√ν or w_St.

## Caveats

- No mean-fit metrics are reported, and stop-gradients are not discussed. The Eq. 12 variant still fits γ, but accuracy is not quantified.
- It supports a per-component σ only indirectly. The finding that a σ-weight coupled to the residual (λν|y−γ|) collapses to zero under momentum optimisers is the transferable warning for any learned scale trained jointly with the mean.
- The authors say they "cannot fundamentally fix the underlying issue" of a projection loss not unfolding all degrees of freedom.
- Evidence is mostly synthetic and 1-D, with one shallow architecture and a single OOD pairing. The α-w_St coupling leaves a "small offset" from the true noise level that is only briefly quantified.
- The paper argues against reading the decomposition as calibrated, so it gives no guidance on weighting a flow-matching loss by σ.
