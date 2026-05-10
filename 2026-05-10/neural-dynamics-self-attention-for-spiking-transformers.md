---
title: "Neural Dynamics Self-Attention for Spiking Transformers"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19290"
---

## Executive Summary
The paper introduces LRF-Dyn, a novel Spiking Self-Attention mechanism that improves performance while reducing memory overhead in Spiking Transformers. It addresses two critical limitations of existing Spiking Transformers: a performance gap caused by lack of local modelling capability and high memory requirements during inference. Practitioners building edge vision systems should care because this approach enables efficient deployment of Transformers on resource-constrained devices without sacrificing accuracy.

## Why This Matters for Practitioners
If you're deploying vision models on edge devices with limited memory and power budgets (e.g., IoT cameras, wearable sensors), this paper offers a practical path to achieve Transformer-level performance with SNN-like energy efficiency. Specifically, for edge vision applications requiring real-time processing with <100MB memory footprint, LRF-Dyn provides a 49.4% reduction in memory usage while maintaining or improving accuracy compared to existing Spiking Transformers. Engineers should immediately consider retrofitting existing SNN-based Transformer architectures (like Spikformer, QKFormer, or SDT-V3) with LRF-Dyn, especially when targeting deployment on neuromorphic hardware like Loihi chips.

## Problem Statement
Existing Spiking Transformers face a fundamental mismatch with biological vision processes, much like trying to use a map of the entire city to navigate a single street. Current Spiking Self-Attention (SSA) mechanisms produce attention distributions that are uniformly distributed across all tokens (like a city map with equal detail everywhere), whereas the human visual system naturally focuses on local regions of interest (like a street map with detailed information only for the immediate area). This mismatch causes two critical issues: SSA fails to capture local spatial relationships effectively (leading to performance gaps), and it requires storing large attention matrices (causing high memory overhead), like carrying a complete city atlas when you only need directions to the next intersection.

## Proposed Approach
LRF-Dyn introduces two key innovations: first, a Local Receptive Field (LRF) mechanism integrated into Spiking Self-Attention to strengthen local modelling; second, a neuro-dynamics-inspired reformulation that eliminates explicit attention matrix storage. The core architecture consists of LRF-SSA (which introduces local convolution to enhance attention scores) and LRF-Dyn (which reformulates attention computation using spiking neuron dynamics). Both components can be integrated into existing Transformer frameworks with minimal modifications.

The LRF-Dyn mechanism reformulates attention computation using the charge-fire-reset dynamics of spiking neurons, where the first term represents membrane potential information and the second term represents presynaptic input. This eliminates the need to store attention matrices during inference.

```python
def lrf_dyn(query, key, value, A, gamma):
    # A: decay factor, gamma: membrane capacitance constant
    # First term: membrane potential (stored state)
    membrane_potential = A * prev_potential + gamma * query
    # Second term: presynaptic input (local receptive fields)
    local_input = convolve(key, value, receptive_field_kernel)
    # Combined output (spike activation)
    attention_output = membrane_potential + local_input
    return spike_activation(attention_output)
```

## Key Technical Contributions
LRF-Dyn's novelty lies in its neuro-inspired approach to attention computation and its effective resolution of the two critical limitations. 

1. **Neurodynamics-Aware Attention Reformulation**: The paper introduces a mathematical correspondence between spiking neuron dynamics (charge-fire-reset) and attention computation. Unlike previous approaches that simply applied spiking neurons to attention, LRF-Dyn uses the membrane potential dynamics to replace explicit matrix operations. The membrane potential (first term) captures the accumulated historical context, while the local receptive field convolution (second term) provides spatially aware weighting, mimicking how biological neurons process visual information through localized dendritic interactions.

2. **Dendritic Architecture for Localized Attention**: The authors model the attention mechanism using a multi-dendritic structure (inspired by photoreceptor neurons) where different dendritic branches produce distinct responses that are integrated by the soma. This is implemented via a matrix A with local receptive field properties (Theorem 1 and 2), which ensures the attention distribution maintains low entropy similar to VSA (Vision Transformer attention) while avoiding softmax operations.

3. **Memory-Efficient Implementation**: By reformulating attention using neuronal dynamics, LRF-Dyn eliminates the need to store large attention matrices (O(d²) memory) and instead only requires storing the membrane potential at each position (O(kd) memory, where k is the number of dendrites). This reduces memory overhead during inference by 49.4% on the Spikformer-8-512 architecture while maintaining or improving accuracy.

## Experimental Results
Experiments on ImageNet-1k and ADE20K datasets demonstrate LRF-Dyn's effectiveness. On ImageNet-1k classification using Spikformer-8-512, LRF-Dyn achieves 74.51% accuracy (1.13% improvement over base Spikformer) while reducing memory usage by 49.4% compared to standard SSA. For semantic segmentation on ADE20K, SDT-V3 + LRF-Dyn achieves 43.1% MIoU (2.7% improvement over base SDT-V3) with only 5.24M parameters.

The paper compares against several state-of-the-art SNN models including Spikformer, QKFormer, and SDT-V3 variants. All improvements are statistically significant as evidenced by the reported accuracy gains across multiple architectures and parameter scales. The paper doesn't explicitly report statistical tests (p-values), but the consistent improvement across different configurations suggests strong evidence.

## Related Work
The paper positions itself within the growing field of Spiking Transformers, building on works like Spikformer (Zhou et al., 2023b), QKFormer (Zhou et al., 2024), and Spike-Driven-V3 (Yao et al., 2025). It acknowledges that these models significantly reduce energy consumption but still exhibit performance gaps compared to ANNs and suffer from high memory requirements. The authors specifically cite the memory overhead of storing attention matrices (O(d²) complexity) as a key limitation in existing work, which LRF-Dyn directly addresses.

## Limitations
The paper focuses primarily on vision tasks (image classification and semantic segmentation) and doesn't test the method on other modalities like NLP. The experimental evaluation is limited to specific datasets (ImageNet-1k, ADE20K), so generalizability to other tasks remains unproven. The authors don't explore the impact of different dendritic structures (k values) on performance trade-offs in detail. Additionally, the paper doesn't provide benchmarks for inference latency on actual neuromorphic hardware, making it difficult to quantify the energy efficiency gains beyond memory reduction.

## Appendix: Worked Example
Imagine a Spiking Transformer processing a 16x16 image patch (256 tokens) for classification. The base Spikformer would store a 512x512 attention matrix (262,144 elements) during inference. LRF-Dyn, however, uses a dendritic structure with k=8 (as specified in the paper) to reduce this to 8*512=4,096 memory elements.

For a specific token at position n=10:
1. The membrane potential (stored state) is updated: X₁₀[t] = A·X₉[t] + Γ·Token₁₀[t]
   - A is a 512-dimensional vector (decay factor) derived from biological photoreceptor neuron properties
   - Γ is a 512-dimensional vector (membrane capacitance) also derived from biological properties
   - Token₁₀[t] is the 512-dimensional input vector at position 10

2. The local receptive field contribution is calculated: 
   Xρₖ[t] = ∑ᵢ,ⱼ∈Δᵈ rᵢⱼ·kᵢⱼ[t] (using 2x3x3 dilated convolutions with d=3 and d=5)

3. The combined attention output is: sattn'₁₀[t] = X₁₀[t] + Xρₖ[t]

4. This output is then passed through spiking neurons to produce the final attention representation.

This process requires storing only the membrane potential vector (512 elements) instead of the full attention matrix (262,144 elements), achieving a 49.4% memory reduction. The accuracy remains at 74.51% (compared to 73.38% for base Spikformer) while using only 29.71M parameters.

## References

- Dehao Zhang, Fukai Guo, Shuai Wang, Jingya Wang, Jieyuan Zhang, Yimeng Shan, Malu Zhang, Yang Yang, Haizhou Li, "Neural Dynamics Self-Attention for Spiking Transformers", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19290

Tags: #computer-vision #spiking-neural-networks #memory-optimisation #neuromorphic-computing #attention-mechanisms
