---
title: "Adaptive Layerwise Perturbation: Unifying Off-Policy Corrections for LLM RL"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19470"
---

## Executive Summary
Adaptive Layerwise Perturbation (ALP) addresses off-policy instability in LLM reinforcement learning by injecting learnable noise into hidden states across all layers during training. This unified approach to handling both policy staleness and training-inference mismatch prevents heavy-tailed importance ratios, maintains training stability, and improves final performance on both single-turn and multi-turn reasoning tasks. Senior engineers working with LLM-based agents should consider ALP to avoid training collapse and improve exploration in production systems.

## Why This Matters for Practitioners
If you're running LLM agents in production that require iterative refinement through reinforcement learning (like math problem solvers or tool-using assistants), the training process is likely plagued by instability from the moment the inference engine (vLLM, SGLang, etc.) diverges from the training distribution. This paper shows that the common technique of truncating importance ratios (MIS/Bypass) leads to over-truncation and early plateau, causing your agent to plateau at suboptimal performance. The solution is simpler than you think: inject small learnable perturbations into hidden states across all transformer layers during training, using the perturbed policy as the numerator of the importance ratio. This requires only a few lines of code and avoids the complexity of maintaining multiple ratios and thresholds. For your next RL training run, replace your standard importance ratio calculation with ALP and expect both more stable training and up to 2.6% higher average accuracy on math benchmarks.

## Problem Statement
Imagine you're training a self-driving car's navigation system with a simulator, but in production, the car's sensors have different noise profiles than the simulator. The training data becomes increasingly mismatched with real-world conditions, causing the navigation system to overreact to small sensor variations. Similarly, in LLM reinforcement learning, the "inference engine" (vLLM, SGLang, etc.) creates a training-inference mismatch that grows with each policy update, leading to heavy-tailed importance ratios that destabilise training and prevent further exploration.

## Proposed Approach
ALP injects small learnable Gaussian perturbations into the input hidden states across all layers during policy updates. The resulting perturbed policy becomes the numerator of the importance ratio against the unchanged inference policy. This unifies the treatment of both policy staleness and training-inference mismatch within a single ratio, eliminating the need for multiple thresholding rules. The architecture maintains the standard RL training loop but modifies the importance ratio calculation to use the perturbed policy as the numerator.

```python
def calculate_importance_ratio(perturbed_policy, inference_policy, token):
    # Calculate token-level importance ratio using perturbed policy as numerator
    ratio = perturbed_policy(token) / inference_policy(token)
    return ratio
```

## Key Technical Contributions
ALP's novelty lies in its representation-level perturbation strategy, which fundamentally changes how we address off-policy instability. The specific mechanisms include:

1. **Layerwise perturbation of hidden states**: Unlike prior work that modifies output logits or uses partial layer perturbations, ALP injects learnable Gaussian noise into the input hidden states of each transformer layer during policy updates. This allows the perturbation to affect the entire policy family, not just the final output, which is critical for maintaining stability through the full network depth.

2. **Single unified ratio formulation**: Instead of using multiple ratios (as in MIS), ALP uses a single importance ratio of the perturbed training policy against the inference policy. This eliminates the need for thresholding multiple ratios and avoids over-truncation that causes early plateauing.

3. **Effective perturbation scale**: The paper demonstrates that σ should be large enough to cover mismatch noise but small enough to preserve the original policy. This balance is achieved by dynamically adapting the perturbation scale relative to the system-induced bias, which the authors quantify in their analysis.

4. **Comprehensive layer coverage**: The ablation studies show that perturbing all layers (0-27 in their model) is significantly more effective than partial-layer or logits-only approaches. This is because hidden state perturbations across the full depth most effectively enlarge the policy family to cover inference-time mismatch noise.

## Experimental Results
On single-turn math reasoning tasks (Math500, Minerva Math, Olympiad Bench, AIME2024, AIME2025), Seq-ALP achieved the highest average score of 50.53%, outperforming Seq-Bypass (46.66%) and Token-MIS (48.74%). For single-turn tasks, Token-ALP attained the best overall average score (37.87%), with Seq-ALP second (36.83%). The improvement is quantifiable: ALP consistently maintained more stable training dynamics, with lower gradient norms and KL divergence, as shown in Figure 3. On multi-turn tool-integrated reasoning (TIR) tasks, Seq-ALP achieved the highest average accuracy (50.53%), with improvements across all benchmarks except AIME25 (where Token-MIS slightly outperformed). Crucially, ALP also demonstrated improved exploration efficiency, with higher Pass@k scores across moderate-to-large rollout budgets (k = 16-256), indicating more diverse solution trajectories.

## Related Work
ALP builds on prior work in perturbation-based training for robustness, but fundamentally shifts the perspective from output-level to representation-level perturbations. While methods like certified robustness (Cohen et al., 2019) or diffusion model perturbations (Ho et al., 2020) focus on specific application domains, ALP addresses the core geometry issue in LLM RL training. Unlike prior off-policy correction methods that split the problem into multiple ratios (MIS/Bypass), ALP unifies the treatment of staleness and inference mismatch within a single ratio. The paper also demonstrates that ALP outperforms these baselines by substantially reducing the tail of importance ratios and maintaining more stable KL divergence.

## Limitations
The paper doesn't test ALP on extremely large language models (beyond 7B parameters) or with different inference engines beyond vLLM and SGLang. The authors acknowledge that perturbation scale (σ) requires tuning to match the system-induced bias, though they provide a theoretical bound for this tuning. The ablation study doesn't explore the impact of different perturbation distributions (beyond Gaussian), which could be relevant for specific model architectures. Additionally, while ALP improves exploration efficiency, it doesn't explicitly address the cold-start problem for new agents with no prior policy.

## Appendix: Worked Example
Let's walk through a single token in a math reasoning task with ALP. Starting with a 1.5B parameter model (Qwen2.5-Math-1.5B), the input hidden state for a token has a dimension of 2048. The perturbation δ is sampled from a Gaussian distribution N(0, σ²I) where σ=0.1 (determined through the authors' analysis). For a specific token, the original hidden state is h = [0.2, -0.4, 0.7, ..., 0.0] (length 2048). After perturbation, it becomes h + δ = [0.21, -0.38, 0.73, ..., 0.02] (each element perturbed by a small random value). This perturbed hidden state flows through the transformer layers, producing a slightly different output distribution. If the inference policy's probability for the next token was 0.05, the perturbed training policy might yield 0.06, resulting in an importance ratio of 0.06/0.05 = 1.2. This ratio is used in the loss function without additional thresholding, allowing the model to learn from the perturbed distribution while staying within a trust region. This process happens for every token and layer during every training iteration, gradually smoothing the optimisation landscape.

## References

- **Code:** https://github.com/Chenluye99/Adaptive-Layerwise-Perturbation.
- Chenlu Ye, Xuanchang Zhang, Yifan Hao, Zhou Yu, Ziji Zhang, Abhinav Gullapalli, Hao Chen, Jing Huang, Tong Zhang, "Adaptive Layerwise Perturbation: Unifying Off-Policy Corrections for LLM RL", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19470

Tags: #machine-learning #ai-applications #reinforcement-learning #off-policy-correction #hidden-state-perturbation #training-inference-mismatch
