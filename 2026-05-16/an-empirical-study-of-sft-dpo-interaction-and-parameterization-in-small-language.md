---
title: "An Empirical Study of SFT-DPO Interaction and Parameterization in Small Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20100"
---

## Executive Summary
This paper empirically investigates the interaction between supervised fine-tuning (SFT) and direct preference optimisation (DPO) for small language models, alongside parameter-efficient adaptation strategies like LoRA. It finds that for GPT-2-scale models and modest data, full-parameter SFT consistently outperforms alternatives, while DPO and LoRA provide only marginal gains. Engineers should prioritise data scaling and full fine-tuning over DPO or LoRA in small-model regimes to avoid wasted effort.

## Why This Matters for Practitioners
If you're building production systems with small language models (under 1B parameters) and constrained compute, this paper directly challenges common assumptions: DPO and LoRA are not universally beneficial. For paraphrase detection, adding DPO after SFT improved F1 by just 0.22 points (89.21 to 89.43) while requiring extra training time. LoRA (rank 8) reduced accuracy by 2.15 points versus FFT (87.70 vs 89.87 F1). We recommend: (1) always use full fine-tuning (FFT) rather than LoRA for small models, (2) skip DPO if your task uses clear binary labels (e.g., paraphrase detection), and (3) invest in more training data instead of adding preference stages.

## Problem Statement
Today's engineering teams often treat DPO and LoRA as "universal solutions" for small models, like adding a pressure regulator to a cracked pipe. Just as a plumber wouldn't waste time on a gasket when the main line is broken, practitioners waste resources on DPO and LoRA when the core issue, insufficient data or poor parameter updates, remains unaddressed.

## Proposed Approach
The authors conducted controlled experiments comparing three training pipelines (SFT-only, DPO-only, SFT→DPO) and two parameterisation strategies (FFT vs LoRA) on two tasks: paraphrase detection (classification) and sonnet generation (generation). They evaluated using Quora Question Pairs (283k examples) and Shakespearean sonnets (131 poems), with preference pairs constructed by comparing correct/incorrect labels (classification) or reference vs generated continuations (generation).

```python
def dpo_loss(chosen_logprobs, rejected_logprobs, beta=0.2):
    # Encourages higher probability for chosen responses
    return -torch.logsigmoid(beta * (chosen_logprobs - rejected_logprobs))
```

## Key Technical Contributions
The paper's most significant contributions lie in quantifying trade-offs for small-scale adaptation:

1. **Preference signal alignment**: DPO's effectiveness depends entirely on how closely preference pairs mirror supervised objectives. For paraphrase detection, the authors used correct labels as "chosen" and incorrect as "rejected", making DPO competitive with SFT from scratch (88.83% accuracy vs SFT-only 89.21% F1). This eliminated the need for SFT warm start because the preference signal was mathematically equivalent to the supervised loss.

2. **LoRA rank paradox**: Higher LoRA ranks (r=16) underperformed lower ranks (r=8) at identical training time due to "ineffective parameter utilisation". At epoch 3, r=16 introduced more parameters receiving the same updates, causing suboptimal convergence. This contradicts the common assumption that higher rank always improves performance.

3. **Compute-bound hardware impact**: LoRA's memory benefits did not translate to faster training on H100 GPUs (compute-bound hardware). Training time for LoRA (r=8) matched FFT at 21.5 minutes per 15 epochs, proving LoRA's main advantage is memory savings, not speed at this scale.

## Experimental Results
On paraphrase detection (283k examples):
- FFT SFT: 89.87% accuracy (F1 89.21)
- FFT DPO (SFT@9→DPO): 90.05% accuracy (F1 89.43), +0.22 F1 points
- LoRA (r=8) SFT: 87.70% accuracy (F1 87.00)
- LoRA DPO: 88.48% accuracy (F1 87.76), only 0.76 F1 points improvement

On sonnet generation (131 poems):
- Best SFT (T=1.5): 41.29 chrF
- DPO V1 (same sonnets): 41.48 chrF, +0.19
- DPO V3 (top-K pairs): 41.46 chrF, no improvement

The authors note these gains are descriptive (not statistically significant) and lack context on how they compare to seed-to-seed variance (e.g., sonnet generation had 0.573 standard deviation at T=2.0).

## Related Work
This work directly challenges common assumptions about DPO and LoRA for small models. It builds on DPO's promise [1] but shows its limitations in small-scale regimes, while extending LoRA's efficiency claims [2] by proving FFT is equally fast on modern GPUs. Unlike prior work that assumed DPO would scale down, this paper quantifies that preference optimisation provides "only marginal returns" for small models and modest data.

## Limitations
The authors acknowledge the study is limited to GPT-2-scale models (124M parameters) and two specific tasks, so results may not generalise to larger models or complex generation tasks. They also note DPO could be more effective with diverse preference pairs, but their dataset was too small to explore. Our assessment: the analysis is thorough for small models but overlooks scenarios with very large datasets (e.g., 1M+ examples) where DPO might provide clearer benefits.

## Appendix: Worked Example
Let's walk through the best-performing strategy (SFT@9→DPO) for paraphrase detection:

1. **Task setup**: Quora Question Pairs dataset (283,011 training examples, 12,786 dev examples) for binary classification.
2. **SFT phase**: Train GPT-2 (124M) with full fine-tuning (FFT) for 9 epochs using 283k examples. Dev accuracy: 89.87% (F1 89.21).
3. **Preference pair construction**: For each dev example, set "chosen" = correct label, "rejected" = incorrect label → 12,786 pairs.
4. **DPO phase**: Apply DPO loss (β=0.2, LR=5×10⁻⁶) for 6 epochs. The loss function maximises the log-probability difference between correct and incorrect labels: 
   `L = -log(σ(0.2 × (log P(correct) - log P(incorrect))))`
5. **Result**: Dev accuracy improved to 90.05% (F1 89.43), a 0.18 F1 point gain. This minimal improvement occurred because the preference signal perfectly aligned with the supervised objective, eliminating the need for a warm start.

## References

- **Code:** https://github.com/Harry20030331/cs224n_project.
- Yuming Feng, Christy Yang, "An Empirical Study of SFT-DPO Interaction and Parameterization in Small Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20100

Tags: #nlp #sft #dpo #lora
