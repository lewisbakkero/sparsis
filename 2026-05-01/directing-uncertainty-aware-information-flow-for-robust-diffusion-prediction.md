---
title: "Directing Uncertainty-Aware Information Flow for Robust Diffusion Prediction"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37000"
---

## Executive Summary
SIEVE is a novel framework for information diffusion prediction that explicitly models user participation uncertainty, a critical factor ignored by most existing methods. By introducing controllable uncertainty injection and uncertainty-aware directed graph aggregation, SIEVE significantly outperforms state-of-the-art approaches across four public datasets, offering engineers a practical way to build more robust social network analytics systems.

## Why This Matters for Practitioners
If you're building social media analytics or content recommendation systems, SIEVE's approach directly addresses a hidden flaw in current graph-based diffusion models: treating all user interactions as equally reliable signals. This oversight causes brittle models that fail during viral events or in fragmented attention environments. The paper demonstrates that simply implementing uncertainty-aware aggregation can improve Hit@10 scores by 3.5-17.5% across datasets (Android: 22.5% relative improvement over runner-up PMRCA). For engineers, this means: (1) When designing diffusion prediction systems, explicitly model interaction reliability rather than assuming uniform participation; (2) Use layer-specific perturbation control (like SIEVE's ϵl) to adapt to network structure; (3) For sparse networks like Douban, implement aggressive edge pruning (β=0) rather than default smoothing techniques. Avoid the common mistake of treating all user engagements as equally valuable, this directly impacts your model's real-world reliability.

## Problem Statement
Imagine mapping a city's traffic flow using only the speed of your own car, ignoring that some drivers weave through lanes while others are stuck in gridlock. Current diffusion prediction models do exactly this: they treat all observed user interactions as equally reliable signals, ignoring that some "engagements" are fleeting taps while others represent genuine interest. The authors call this the "participation homogeneity assumption" (PHA), and it's like building a roadmap using GPS data from only one driver, it captures a path but misses the reality of varying driving behaviours. This assumption leads to fragile topologies and uncertainty contamination, causing prediction failures during real-world viral events.

## Proposed Approach
SIEVE addresses user participation uncertainty at two levels: through robust representation learning and uncertainty-aware graph aggregation. It processes topic diffusion cascades through a graph neural network with two key modifications: (1) Feature-level uncertainty injection during representation learning, and (2) Dynamic asymmetric aggregation that suppresses uncertainty propagation. The system takes a topic's observed cascade history as input, processes it through the enhanced GNN, and predicts the next participating user.

```python
def sieve_gnn_layer(layer, Z_prev, U_prev, A, epsilon_i, beta):
    """SIEVE's GNN layer with uncertainty-aware aggregation"""
    # Compute asymmetric aggregation weights
    S = np.zeros_like(A)
    for i in range(len(U_prev)):
        for j in range(len(U_prev)):
            if A[j, i] == 1:
                confidence = np.exp(-U_prev[j])
                x = confidence - epsilon_i[i]
                if x >= 0:
                    weight = gamma * sigmoid(x)
                else:
                    weight = beta
                S[j, i] = weight
    
    # Apply uncertainty injection
    Z_injected = Z_prev + uncertainty_injection(Z_prev, layer)
    
    # Update representations
    Z_current = activation(S @ Z_injected @ W)
    return Z_current
```

## Key Technical Contributions
SIEVE's core innovation lies in the precise mechanisms for uncertainty modelling rather than just identifying the problem:

1. **Layer-specific uncertainty injection**: Unlike standard data augmentation, SIEVE adds a layer-specific perturbation with dual constraints: L2 norm capped at ϵl and sign constrained to match the original representation. This targets the key insight that user participation uncertainty affects interaction strength (magnitude) rather than semantic direction (sign). The perturbation is calculated as Δ⁽ˡ⁾ᵢ = ϵˡ · (∆⁽ˡ⁾ᵢ ⊙ sign(z⁽ˡ⁻¹⁾ᵢ)) / ||∆⁽ˡ⁾ᵢ ⊙ sign(z⁽ˡ⁻¹⁾ᵢ)||₂ + δ, ensuring it simulates marginal fluctuations around the core state.

2. **Local stability-based uncertainty proxy**: SIEVE avoids the impracticality of inferring Shannon entropy from high-dimensional embeddings by using a local contrastive loss to measure representation stability. For node i, U⁽ˡ⁻¹⁾ᵢ = ½(l(z⁽ˡ⁻¹⁾ᵢ, ˜z⁽ˡ⁻¹⁾ᵢ) + l(˜z⁽ˡ⁻¹⁾ᵢ, z⁽ˡ⁻¹⁾ᵢ)), where l is an InfoNCE-like loss over the node and its first-order neighbours. This local design captures stability within the immediate environment, with higher U indicating greater uncertainty.

3. **Dynamic suppression factor β**: The paper demonstrates that optimal uncertainty suppression (β) is dataset-dependent, not a fixed hyperparameter. On sparse Douban (L=21.76), β=0 (hard pruning) works best, while on Memetracker (L=16.04), larger β improves performance. The authors show this requires engineers to tune β per network type rather than defaulting to conservative values.

## Experimental Results
SIEVE significantly outperforms state-of-the-art methods across four public datasets using standard metrics Hit@10 and MAP@10. The paper provides specific numbers:

- **Android**: SIEVE Hit@10 = 0.1532 (vs. PMRCA 0.1249, +22.5% relative improvement)
- **Christianity**: SIEVE Hit@10 = 0.4316 (vs. PMRCA 0.3866, +11.6% relative improvement)
- **Douban**: SIEVE Hit@10 = 0.3077 (vs. PMRCA 0.2766, +11.2% relative improvement)
- **Memetracker**: SIEVE Hit@10 = 0.5172 (vs. PMRCA 0.4904, +5.5% relative improvement)

The paper doesn't specify statistical significance testing (p-values), but reports using five random seeds and averaging results. The improvement is most pronounced on longer cascades (Android has average cascade length 42.05), while Memetracker's shorter cascades (L=16.04) show smaller relative gains.

## Related Work
SIEVE positions itself as addressing a critical gap overlooked by existing diffusion prediction work. While prior methods focus on refining network structure (e.g., DyHGCN, MSHGAT) or leveraging temporal dynamics (e.g., GRASS, MINDS), none systematically model interaction reliability heterogeneity. The authors contrast their approach with attention mechanisms (like GAT), noting these learn neighbour relevance but don't model information source reliability. SIEVE builds on but significantly extends prior work by explicitly modelling and mitigating uncertainty at the representation learning and aggregation levels.

## Limitations
The paper acknowledges SIEVE's benefits are most pronounced on datasets with longer cascades (Android, L=42.05), while on shorter-cascade datasets like Memetracker (L=16.04), the relative improvement is smaller. The stability-based uncertainty proxy may not align well with certain network dynamics (e.g., Memetracker's chaotic dynamics), requiring dataset-specific tuning of the suppression factor β. The paper doesn't test how SIEVE performs on extremely large-scale networks (>100M nodes), and the ablation studies focus on prediction accuracy without measuring inference latency.

## Appendix: Worked Example
Let's walk through SIEVE's uncertainty-aware aggregation with concrete numbers from the Android dataset (average cascade length 42.05):

1. **Input**: Consider user node u with representation z⁽ˡ⁻¹⁾ᵤ and uncertainty proxy U⁽ˡ⁻¹⁾ᵤ = 0.8 (high uncertainty). Its neighbour v has representation z⁽ˡ⁻¹⁾ᵥ and U⁽ˡ⁻¹⁾ᵥ = 0.2 (low uncertainty).

2. **Confidence calculation**: The original connection Aᵥᵤ = 1 (exists), so confidence = e^(-U⁽ˡ⁻¹⁾ᵥ) = e^(-0.2) ≈ 0.8187.

3. **Weight calculation**: The target node u has learnable tolerance εᵤ = 0.3. 
   - x = confidence - εᵤ = 0.8187 - 0.3 = 0.5187
   - Since x ≥ 0, weight = γ * sigmoid(x)
   - γ is dynamically adjusted to 0.7 (based on median uncertainty U⁽ˡ⁻¹⁾ₘₑ𝒹ᵢₐₙ = 0.45)
   - sigmoid(0.5187) ≈ 0.625
   - Final weight = 0.7 * 0.625 = 0.4375

4. **Aggregation**: The standard GNN would use weight 1.0 for this connection, but SIEVE uses 0.4375. This suppresses information flow from high-uncertainty neighbours (u) while emphasizing reliable connections (v). In the ablation study (Table 3), removing this uncertainty-aware aggregation (w/o UAA) reduces Android Hit@10 from 0.1532 to 0.1418, demonstrating its practical impact.

See Key Technical Contributions for how this precise mechanism enables SIEVE's 11.6% relative improvement on Christianity dataset.

## References

- **Code:** https://github.com/HeyWeCome/BuzzBloom
- Weikang He, Yunpeng Xiao, Mengyang Huang, Xuemei Mou, Rong Wang, Qian Li, "Directing Uncertainty-Aware Information Flow for Robust Diffusion Prediction", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37000

Tags: #social-networks #information-diffusion #graph-neural-networks #uncertainty-aware-modelling
