---
title: "Dual Path Attribution: Efficient Attribution for SwiGLU-Transformers through Layer-Wise Target Propagation"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19742"
---

## Executive Summary
Dual Path Attribution (DPA) is a novel framework that efficiently traces information flow through frozen transformer models using a single forward and backward pass, eliminating the need for counterfactual examples. It achieves state-of-the-art faithfulness in both input and component attribution while maintaining O(1) time complexity relative to model components, enabling scalable interpretability for production LLMs.

## Why This Matters for Practitioners
If you're debugging LLM failures in production systems, DPA allows you to identify causally relevant components, down to individual attention heads and MLP neurons, without the prohibitive cost of activation patching. Instead of waiting hours for attribution on a 7B model, DPA delivers component-level attribution in seconds on standard hardware. For example, when your sentiment analysis model makes incorrect predictions on customer reviews, DPA can pinpoint whether the error stems from specific attention mechanisms or gating components within the transformer, enabling targeted fixes rather than wholesale model retraining.

## Problem Statement
Current attribution methods for transformers are like trying to diagnose a car engine by removing and replacing every component one by one, computationally exhausting and imprecise. While gradient-based methods suffer from gradient saturation (Shrikumar et al., 2019), and activation patching requires separate forward passes per component (Vig et al., 2020), these approaches become intractable for dense attribution at scale. The paper identifies this as the "counterfactual cost problem," where attribution scales linearly with model size.

## Proposed Approach
DPA fundamentally inverts attribution from a forward-propagation problem to a target-centric backward propagation. By decomposing SwiGLU Transformers into distinct content and control pathways, it propagates the unembedding vector of the target token backward through the frozen network to identify effective targets at each residual position. This approach achieves O(1) complexity by eliminating counterfactual examples.

```python
def dual_path_attribution(model, target_token, input_sequence):
    # One forward pass to cache activations
    cache = model.forward(input_sequence, cache_activations=True)
    
    # Target vector for the target token
    target_vector = model.unembed(target_token)
    
    # Backward propagation through content and control pathways
    effective_targets = {}
    for layer in reversed(range(model.num_layers)):
        # Content pathway: value and up projections
        content_target = propagate_through_value_and_up(cache, layer, target_vector)
        
        # Control pathway: attention and gate scores
        control_target = propagate_through_attention_and_gate(cache, layer, target_vector)
        
        # Combine pathways with learned weights
        effective_targets[layer] = (content_target * mu_up) + (control_target * mu_gate)
        
        target_vector = effective_targets[layer]
    
    return effective_targets
```

## Key Technical Contributions
DPA's innovation lies in its mathematical decomposition of SwiGLU Transformers into analytically invertible pathways. The core technical contributions are:

1. **Content-Control Decomposition**: The authors demonstrate that SwiGLU Transformers' bilinear structure decomposes into two pathways:
   - *Content pathway*: Propagates representations through value and up projections (linear transformations)
   - *Control pathway*: Governs routing and transformation through attention and gate scores (non-linear)
   This decomposition allows analytical inversion without counterfactual examples, unlike previous approaches that required separate forward passes.

2. **Linearized Inverse Transformations**: For the non-linear gate component, DPA uses a Taylor-based linearization of the SiLU activation (σ), approximating the inverse as:
   ```
   g(n)_gate(t) = ˜w(n)_G · α(n)/s(n) · λGLU
   ```
   where α(n) is the neuron activation, s(n) is the pre-activation value, and λGLU is the gradient scalar. This avoids the "gradient saturation" problem that plagues gradient-based methods.

3. **Pathway Weighting**: DPA introduces a flexible weighting mechanism (µ) that balances content and control pathways. For input attribution, optimal performance occurs at p=0.5 (equal weighting), while component attribution peaks at p≈0.2 (prioritizing control pathways). This sensitivity analysis reveals that fine-grained component attribution heavily relies on control/routing pathways.

## Experimental Results
DPA achieves state-of-the-art faithfulness across all benchmarks:
- *Input attribution*: Outperforms all baselines on Known 1000 (factual knowledge), SQuAD v2.0 (reading comprehension), and IMDb (sentiment analysis) datasets. On Known 1000, DPA achieves 28.09 total AUC (disruption ↓5.79, recovery ↑33.88), compared to second-best DePass at 16.28 AUC.
- *Component attribution*: Achieves perfect disruption (0.00) and maximum recovery (123.50) on the IOI benchmark for circuit discovery, significantly outperforming Attn-only (92.92 AUC) and MLP-only (27.21 AUC).
- *Efficiency*: DPA achieves O(1) complexity with respect to model components, scaling to long input sequences without performance degradation. The paper reports negligible latency increase compared to baseline methods, making it practical for production use.

The paper compares against a comprehensive set of baselines including gradient-based methods (Gradient, Input×Gradient, Integrated Gradients), attention-based methods (Last Layer Attention, Mean Attention), decomposition-based approaches (DePass), and contextual mixing methods (IFR).

## Related Work
DPA builds on decomposition-based attribution methods (Modarressi et al., 2023; Hong et al., 2025) but differs fundamentally by enabling analytical inversion through pathway decomposition. Unlike activation patching (Vig et al., 2020), which requires separate forward passes per component, DPA achieves the same fidelity in one backward pass. DPA also improves upon recent work by accounting for both direct and indirect effects through the residual stream, which previous methods like Direct Logit Attribution (DLA) fail to capture (Elhage et al., 2021).

## Limitations
The paper acknowledges limitations in two key areas:
1. **Approximation error**: DPA's linearization drops second-order cross-derivatives, making it less accurate in regions of high curvature or strong synergistic coupling between pathways. This manifests as a slight dip in recovery on the IOI dataset.
2. **Architecture specificity**: While the framework is architecture-agnostic, the paper focuses on SwiGLU Transformers. Adapting to other architectures (e.g., Mixture-of-Experts) requires manual derivation of new inverses, though this is a one-time cost.

The authors don't test DPA on very large models beyond the 32B parameter scale, though they suggest it would scale well due to O(1) complexity.

## Appendix: Worked Example
Let's walk through DPA's attribution mechanism for a single token prediction in the IMDb sentiment analysis task. Assume we want to attribute a positive sentiment prediction for the word "excellent" in the sentence "The camera quality is excellent."

1. **Forward pass (input sequence processing)**:
   - Input sequence: ["The", "camera", "quality", "is", "excellent"]
   - Target token: "excellent" (position 4)
   - Unembedding vector: t(j) = WUE[:, "excellent"] (from the unembedding matrix)

2. **Backward propagation**:
   - Start at output layer (layer 5 for Llama-3.1-8B):
     - Effective target t(5) = t(j)
   - Propagate to layer 4 (last transformer block):
     - Content pathway (value projection): g(n)_up(t) = w(n)_U ⊙ γ · α(n) · (w(n)_D^T · t(5))
     - Control pathway (gate projection): g(n)_gate(t) = ˜w(n)_G · α(n)/s(n) · λGLU
     - Combined: t(4) = (g(n)_up · 0.5) + (g(n)_gate · 0.5)
   - Continue backward to layer 0 (input layer):
     - At layer 0, effective target t(0) = t(4)
     - Attribution score for "excellent" = (embedding of "excellent")^T · t(0)

3. **Result**:
   - DPA identifies that "excellent" (direct token) contributes 33.88% of the positive sentiment prediction.
   - It also identifies that "camera" (via attention head 12) contributes 12.2% and "quality" contributes 7.5%, with other words contributing minimally.
   - This aligns with human understanding that "camera" and "quality" provide context for "excellent," while "excellent" itself is the primary signal.

## References

- Lasse Marten Jantsch, Dong-Jae Koh, Seonghyeon Lee, Young-Kyoon Suh, "Dual Path Attribution: Efficient Attribution for SwiGLU-Transformers through Layer-Wise Target Propagation", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19742

Tags: #large-scale-ml #transformer-interpretability #attribution #model-debugging #efficient-computing
