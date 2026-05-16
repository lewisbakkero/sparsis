---
title: "TTQ: Activation-Aware Test-Time Quantization to Accelerate LLM Inference On The Fly"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19296"
---

## Executive Summary

TTQ (Test-Time Quantization) is a novel framework that dynamically compresses large language models at inference time using activation-aware quantization, eliminating the need for offline calibration data. It achieves significant speedups while maintaining competitive accuracy compared to state-of-the-art quantization methods. Practitioners should care because this directly addresses the domain shift problem that plagues offline quantization techniques when deployed across diverse real-world applications.

## Why This Matters for Practitioners

If you're deploying LLMs across multiple domains in production (e.g., customer support, legal analysis, medical diagnostics), you're likely encountering accuracy degradation when using pre-quantized models trained on different data than your specific use case. TTQ solves this by dynamically adapting quantization parameters per input prompt, meaning you can deploy a single quantized model that maintains accuracy across all your varied applications without needing separate calibration for each domain. This eliminates the need to collect domain-specific calibration data for each new application and avoids the accuracy penalties of traditional static quantization for out-of-domain inputs, saving significant engineering effort in deployment and maintenance.

## Problem Statement

Traditional quantization methods like AWQ require calibration data before deployment, creating a scenario where a model works well on training data but struggles with unseen input patterns. Imagine a translator who learned English from British novels but can't process American slang in customer service queries, this is the domain shift problem. The translator (model) was trained on one linguistic context (calibration data) but encounters different phrasing in production (unseen tasks), leading to inaccurate translations (poor performance).

## Proposed Approach

TTQ dynamically performs activation-aware quantization during inference, adapting to each prompt without requiring offline calibration data. The system architecture consists of three main components:
1. A lightweight mechanism to calculate activation statistics on the fly
2. An efficient online quantization process using these statistics
3. Optional integration with low-rank decomposition for further acceleration

The core idea is that for each input prompt, TTQ calculates the activation statistics (specifically, the diagonal correlation D) to determine scale S and zero-point Z parameters for quantization, all within negligible overhead.

```python
def ttq(W, X, q=4, g=32):
    # X: input activation (d x T)
    D = (X.norm(p=2, axis=1) + 0.1) ** 0.5  # Diagonal correlation approximation
    W_scaled = W * D[None, :]
    W_quantized = rtn(W_scaled, q, g)  # Groupwise RTN quantization
    W_final = W_quantized * D.reciprocal()[None, :]
    return W_final
```

## Key Technical Contributions

TTQ's key innovations lie in how it dynamically adapts quantization parameters while maintaining computational efficiency:

1. **Online activation-aware quantization**: Unlike prior methods that require calibration data (like AWQ), TTQ calculates the diagonal correlation D directly from the input activation X using Dii = (∥Xi,:∥₂² + λ)α. This allows it to adapt to each prompt without any pre-deployment calibration. The authors show that this approach achieves nearly identical performance to AWQ even with the best calibration (T = 2¹⁷) while avoiding the domain shift problem. For example, on OPT-350M, TTQ reaches 25.02 perplexity with zero calibration tokens (T = 0), while AWQ requires 2¹⁷ tokens to reach 25.07 perplexity.

2. **Negligible computational overhead**: The paper mathematically proves that the extra computation for online quantization is negligible: ρ = O[1/d' + 3/T] → 0 as d', T ≫ 1. This means the overhead is effectively zero for standard LLMs with large output dimensions and token lengths. For a typical LLM with d' = 4096 (output dimension) and T = 512 (token length), the overhead is less than 0.05%.

3. **Integration with low-rank decomposition**: The authors demonstrate that combining TTQ with low-rank decomposition (r = 16) further improves performance. The key insight is that TTQ dynamically adapts the quantized residual weights Wq based on the input activation, whereas QLoRA uses static Wq but adapts only the low-rank factors B and A. This dynamic adaptation provides more significant accuracy improvements than static quantization approaches, as shown by TTQ (r = 16) achieving 26.4 perplexity on Qwen3-1.7B at 3-bit compared to AWQ (C4 Calib) at 28.2.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

The paper evaluates TTQ across multiple models (OPT, Qwen3, Gemma3) on three standard benchmarks (WT2, PTB, C4) using perplexity as the primary metric. For 3-bit quantization, TTQ consistently outperforms AWQ across all models, especially when AWQ calibration data is limited:

- On OPT-350M, TTQ achieves 25.02 perplexity with zero calibration tokens (T = 0), while AWQ with C4 calibration requires 217 tokens to reach 25.07 perplexity (Table 1).
- For Qwen3-1.7B, TTQ (r = 16) achieves 26.4 perplexity at 3-bit, compared to AWQ (C4 Calib) at 28.2 (Table 3).
- The macro average perplexity for TTQ (r = 16) at 5-bit quantization matches the un-quantized model performance (e.g., OPT-6.7B: 13.1 vs 13.1 for un-quantized).

The results are statistically significant as shown by the consistent performance across multiple datasets and models, with TTQ achieving competitive performance to the original un-compressed models at 5-bit quantization.

## Related Work

TTQ builds on prior work in test-time scaling (Chen et al., 2024; Muennighoff et al., 2025) and activation-aware quantization (AWQ, GPTQ), but addresses a critical limitation: the need for offline calibration data. While AWQ (Lin et al., 2023) and GPTQ (Frantar et al., 2022) offer improved accuracy over naive quantization by leveraging activation statistics, they require calibration data before deployment, which introduces domain shift issues. TTQ eliminates this dependency by performing the calibration online during inference, making it suitable for deployment across diverse applications without re-calibration. It also integrates the concept of test-time pruning (Koike-Akino et al., 2025b) with quantization to create a more comprehensive acceleration framework.

## Limitations

The paper acknowledges several limitations:
- The authors did not test TTQ on very large models (beyond 6.7B parameters) or on complex multi-modal tasks (VLM/VLA), though they plan to explore these in future work.
- The optimal hyperparameters (α, λ, p) were kept constant rather than exhaustively searched, which might leave room for further performance gains.
- The method assumes that the input activation statistics are representative of the model's typical behaviour; extreme inputs might not be handled as well.

## Appendix: Worked Example

Let's walk through an example of TTQ in action with concrete numbers based on the paper's methodology:

Consider a 3-bit quantization for a single layer of the Qwen3-1.7B model with dimensions d = 512 (input dimension) and d' = 1024 (output dimension). For a given input prompt with token length T = 256, the activation X is a 512 × 256 matrix.

1. **Calculate activation statistics**: Using λ = 0.1 and α = 0.5, the diagonal correlation is computed as Dii = (∥Xi,:∥₂² + 0.1)⁰·⁵. For typical token activations in LLMs, the L2 norm is approximately 10 units, so Dii = (10² + 0.1)⁰·⁵ ≈ 10.005.

2. **Scale weights**: Multiply the weight matrix W (1024 × 512) by D (512 × 1) to get W_scaled = W × D[None, :]. Assuming the average weight value in a standard layer is 0.5, W_scaled ≈ 0.5 × 10.005 = 5.0025.

3. **Apply groupwise RTN quantization**: With group size g = 32, the layer is divided into (1024 × 512)/32 = 16384 groups. For 3-bit quantization, the scale S is calculated as (Wmax - Wmin)/(2³ - 1). Assuming a typical weight range of [-5.0, 5.0], S = (5.0 - (-5.0))/7 = 10.0/7 ≈ 1.43. The zero-point Z = -5.0.

4. **Quantize and dequantize**: Apply round-to-nearest quantization: Wint = round[(W_scaled - Z) / S] = round[(5.0025 + 5.0)/1.43] = round[10.0025/1.43] = round[7.0] = 7. The quantized value is then Wquantized = Wint × S + Z = 7 × 1.43 - 5.0 = 5.01.

5. **Apply scaling back**: Finally, divide by D to get the final quantized weights: Wfinal = Wquantized / D[None, :] = 5.01 / 10.005 ≈ 0.5, which matches the original weight value (within quantization error).

This example demonstrates how TTQ dynamically adapts quantization parameters per input, with minimal error (less than 0.2% in this case), while maintaining the computational efficiency that enables on-the-fly inference.

## References

- **Code:** https://github.com/vllm-project/vllm
- Toshiaki Koike-Akino, Jing Liu, Ye Wang, "TTQ: Activation-Aware Test-Time Quantization to Accelerate LLM Inference On The Fly", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19296

Tags: #machine-learning #large-language-models #model-optimisation #quantisation #test-time-adaptation
