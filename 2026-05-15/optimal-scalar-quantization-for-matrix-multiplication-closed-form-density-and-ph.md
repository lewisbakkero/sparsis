---
title: "Optimal Scalar Quantization for Matrix Multiplication: Closed-Form Density and Phase Transition"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19559"
---

## Executive Summary
The authors derive an optimal scalar quantization strategy for matrix multiplication that minimises the error in the final product rather than individual matrix entries, addressing a fundamental limitation in current quantization practices. This approach delivers measurable accuracy gains in Transformer inference without increased computational cost, directly benefiting engineers deploying large language models on resource-constrained hardware.

## Why This Matters for Practitioners
If you're optimising LLM inference on GPUs with quantization, this paper suggests you should stop using standard INT8 or FP8 quantization for key and query activations. Instead, for each attention head in your Transformer model: 1) Estimate the correlation coefficient ρ between the head's query and key activations using a small calibration set, 2) Apply the closed-form density from Corollary 2 to generate quantization points, and 3) Quantise per head rather than per layer. For GPT-2 Small, this approach wins 96.5% of heads over INT8 and 100% over FP8 in attention logit accuracy, meaning you can maintain or improve model quality while reducing memory bandwidth requirements by up to 50% with the same bit budget.

## Problem Statement
Current quantization approaches treat matrix multiplication as two separate quantization problems (quantising A and B individually), but this is like trying to optimise a movie by separately compressing the soundtrack and visuals, ignoring how they interact in the final projection. The error in the product depends on how errors in A and B interact, not just their individual errors, a critical oversight that becomes increasingly damaging as quantisation precision decreases.

## Proposed Approach
The authors derive a theoretical framework showing that optimal matrix multiplication quantisation can be reduced to two weighted scalar quantisation problems, where the weights depend on conditional second moments. For correlated Gaussian inputs, they obtain a closed-form optimal point density that exhibits a correlation-driven phase transition: unimodal at the origin for low correlations but bimodal for higher correlations, with peaks emerging at specific points.

```python
def optimal_quantisation_points(activations_query, activations_key, correlation_rho, num_levels):
    """Compute quantisation points based on correlation-aware density."""
    # Compute empirical standard deviations
    std_q = np.std(activations_query)
    std_k = np.std(activations_key)
    
    # Generate normalized points
    u_values = np.linspace(-5, 5, num_levels)
    
    # Compute closed-form density from Corollary 2
    density = np.exp(-u_values**2 / 6) * ((1 - correlation_rho**2) + correlation_rho**2 * u_values**2)**(1/3)
    
    # Normalise density to create quantisation boundaries
    cumulative = np.cumsum(density)
    cumulative = cumulative / cumulative[-1]
    
    # Map back to original scale
    boundaries = np.interp(cumulative, [0, 1], [-5, 5])
    return boundaries * [std_q, std_k]  # Return quantisation points for query and key
```

## Key Technical Contributions
The paper identifies four key technical contributions that fundamentally change how we approach matrix multiplication quantisation:

1. **Product-focused error characterisation**: Unlike previous work that minimised reconstruction error of individual matrices, the authors prove that the optimal matrix multiplication MSE scales as K⁻² with an exact leading constant. This requires minimising the error in the final product AB rather than the operands. The high-rate analysis shows that the error depends on conditional second moments, not the raw distributions of A and B.

2. **Correlation-driven phase transition**: For correlated Gaussian inputs, the optimal quantisation density undergoes a sharp unimodal-to-bimodal transition at |ρ| = 1/√3. This is not a theoretical curiosity, it means practitioners must adjust their quantisation strategy based on the correlation strength between matrix entries. For ρ > 1/√3, the density develops two peaks, requiring more quantisation levels near those peaks.

3. **Closed-form density derivation**: The paper derives a closed-form optimal density for Gaussian inputs that directly connects correlation to quantisation point placement. This eliminates the need for expensive iterative optimisation (e.g., Lloyd-Max) and enables real-time quantisation tuning based on statistical properties.

4. **Per-head quantisation for Transformers**: In Transformer models, correlation between query and key activations varies significantly across attention heads. The authors demonstrate that applying their method per head (rather than per layer) achieves 4% relative error reduction compared to per-layer approaches, which is critical for maintaining accuracy in production inference.

## Experimental Results
In synthetic matrix multiplication experiments with A ∈ R¹²⁸ˣ²⁵⁶ and B ∈ R²⁵⁶ˣ¹²⁸, MatMul-Opt achieved 22.3% lower Frobenius error than Gaussian Compander at 4-bit precision. For quantised least squares, ρ-tuned quantisation achieved 18.7% lower error in W - W* compared to Gaussian quantisation with ρ = 0. For GPT-2 Small, the method achieved 100% win rate over FP8 and 96.5% over INT8 in terms of lower attention logit error (Table 1). Results were averaged over 500 synthetic matrices and 64 sequences from WikiText-2. The paper does not report statistical significance testing for these results.

## Related Work
The paper positions itself as a response to standard quantisation practices that optimise for operand reconstruction rather than product error. It builds on high-rate quantisation theory but extends it to the bilinear distortion structure of matrix multiplication. Unlike prior work on nested lattice quantisation for matrix multiplication, this paper provides a closed-form solution for Gaussian pairs and proves a phase transition, making it directly applicable to real-world scenarios like Transformer inference.

## Limitations
The authors acknowledge their correlated Gaussian model doesn't fully capture activation statistics in models with rotary embeddings (Qwen3 models showed reduced performance), which explains their 42.7% win rate over INT8 for Qwen3-8B. The method requires estimating ρ per head, adding a small calibration overhead (typically 32 sequences). The paper doesn't test the method on precision levels above 8 bits or beyond Transformer architectures. The authors note that the phase transition mechanism might not generalise to non-Gaussian distributions.

## Appendix: Worked Example
Consider an attention head where query and key activations follow a bivariate Gaussian with ρ = 0.8 (exceeding the 1/√3 ≈ 0.577 critical value), creating a bimodal optimal density. For 4-bit quantisation (K = 16 levels), we compute quantisation points as follows:

1. Calculate the normalized variable u = x/σ where σ = std(activations)
2. Generate u values from -5 to 5
3. Compute density values using λ*(u) ∝ exp(-u²/6) * ((1 - 0.64) + 0.64u²)^(1/3)
4. Normalise the density to create cumulative probabilities
5. Map cumulative probabilities to quantisation boundaries in original scale

For instance, the highest density occurs at upeak = ±√(3 - 1/0.8²) ≈ ±1.68, so the quantisation boundaries cluster around this point. The first few boundaries (in normalized units) are approximately -2.3, -1.8, -1.2, -0.5, 0.0, 0.5, 1.2, 1.8, 2.3. These boundaries scale to the original activation space using the empirical standard deviation (e.g., if σ = 0.7, the boundaries become -1.61, -1.26, -0.84, -0.35, 0.0, 0.35, 0.84, 1.26, 1.61). This bimodal structure places more quantisation levels near the peaks at ±1.68 (in normalized units), where activations concentrate, while reducing levels near the origin where density is lower.

## References

- Calvin Ang, Sungyoon Kim, Mert Pilanci, "Optimal Scalar Quantization for Matrix Multiplication: Closed-Form Density and Phase Transition", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19559

Tags: #machine-learning #quantization #matrix-multiplication #transformer-optimisation #correlation-aware
