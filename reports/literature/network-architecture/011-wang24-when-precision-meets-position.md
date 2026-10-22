# When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training

Wang et al., 2024 · [arXiv:2411.13476](https://arxiv.org/abs/2411.13476)

**Evidence: C.** Not peer-reviewed; 17 citations in 1.9 yr (Semantic Scholar, 2026-10-19). Its
claims are the authors' alone: a hypothesis, not a result.

## What they do

In bf16, RoPE's relative-position property fails. The logit drifts with absolute position, and the first token dominates the error.

## What they find

AnchorAttention gives RULER 66.40 → 73.25 at 128K.

## Implementation details

A shared anchor token.

## Caveats

Long-context LLMs.
