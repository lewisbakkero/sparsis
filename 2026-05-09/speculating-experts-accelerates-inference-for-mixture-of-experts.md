---
title: "Speculating Experts Accelerates Inference for Mixture-of-Experts"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19289"
---

## Executive Summary
This paper introduces Speculating Experts, an inference-time optimisation for Mixture-of-Experts (MoE) models that leverages internal model representations to predict future expert selection, enabling CPU-GPU memory transfers to overlap with computation. It achieves up to 14% reduction in time-per-output-token (TPOT) without significant accuracy degradation, addressing a critical bottleneck in memory-constrained deployment of large MoE models.

## Why This Matters for Practitioners
For engineering teams deploying MoE-based LLMs like Qwen3-30B-A3B or GPT-OSS-120B in resource-constrained production environments, this work provides a practical optimisation that can significantly improve throughput without requiring model modifications or additional hardware. On an A6000 GPU with 48GB HBM (typical for many production deployments), integrating this prefetching scheme into your existing inference pipeline, using the open-source YALIS engine, can yield up to 14% lower TPOT, directly increasing your system's token throughput. The implementation requires only adding a few lines of code to your MoE inference engine, as demonstrated in the YALIS integration, with minimal configuration changes required. For teams running inference on GPUs with limited memory (where expert offloading to CPU is necessary), this optimisation directly addresses the bottleneck where CPU-GPU transfers consume 84-88% of TPOT.

## Problem Statement
Imagine a warehouse where workers constantly wait for delivery trucks (CPU) to bring new inventory (expert weights) before starting to pack orders (GPU computation). For MoE models, this waiting time consumes 84-88% of the total time per order (TPOT), making the entire system feel sluggish despite high packing speed. The current approach treats each expert selection as a new delivery request, creating a bottleneck where the packing line is idle while waiting for inventory rather than working.

## Proposed Approach
The authors introduce an expert prefetching scheme that predicts future expert selection using the quasi-hidden state (ql = LN_{l+1}(dl + rl)), where dl is the layer-level default vector and rl is the post-attention residual. This prediction enables transferring future expert weights to GPU memory while the current computation is executing, overlapping memory transfers with computation. The key innovation is executing predicted experts directly rather than treating mispredictions as cache misses.

```python
def moe_block_with_prefetching(layer_idx, input_activations, cpu_expert_weights, gpu_expert_cache, router):
    if layer_idx == 0:
        # Cold start: No prefetched experts
        expert_ids, gating_weights = router(input_activations)
        copy_experts_to_gpu(cpu_expert_weights, expert_ids, gpu_expert_cache)
    else:
        # Use prefetched experts from previous layer
        expert_ids, gating_weights = get_prefetched_experts(layer_idx)
    
    # Compute next layer's expert selection
    next_expert_ids, next_gating_weights = router(input_activations)
    
    # Prefetch next layer's experts while current layer computes
    wait_and_prefetch(gpu_expert_cache, next_expert_ids)
    
    # Execute MoE with current layer's experts
    output_activations = moe_forward(input_activations, expert_ids, gating_weights, gpu_expert_cache)
    return output_activations, next_expert_ids, next_gating_weights
```

## Key Technical Contributions
The authors make several key contributions that distinguish their approach from prior work:

1. **Parameter-free prefetching using internal representations**: They identify the quasi-hidden state (ql = LN_{l+1}(dl + rl)) as containing sufficient signal for predicting future expert selection across diverse MoE architectures. Unlike prior work that required additional training or external representations, this approach computes the prediction entirely from existing model components. The layer-level default vector dl (computed offline by aggregating activations during inference) provides an expert-conditioned bias that improves alignment with the true router input, reducing prediction error by 15-20% compared to using the raw residual stream.

2. **Speculative execution preserving accuracy**: The authors demonstrate that executing predicted experts directly (rather than treating mispredictions as cache misses) maintains downstream task accuracy across most models. For GPT-OSS models, this approach preserves accuracy within 2-3% of baseline performance (Table 1), while for Qwen3-30B-A3B (which shows more sensitivity), accuracy degradation can be mitigated to within 5% using a lightweight neural estimator. The key insight is that higher-ranked experts (those with the largest routing weights) are predicted with high hit rates, and their contributions dominate the model output.

3. **Optimised inference engine integration**: They integrated the prefetching scheme into YALIS, an optimised inference engine that supports key optimizations like torch.compile and CUDA Graphs. The implementation uses double buffering to alternate GPU expert buffers across layers, enabling compute-copy overlap without additional synchronization. CPU offloading is done with pinned memory, providing faster transfers than pageable memory by avoiding OS page faults.

4. **Lightweight neural estimators for accuracy improvement**: For architectures where router-based speculation yields suboptimal accuracy (like Qwen3-30B-A3B), they introduce a lightweight neural estimator that improves expert prediction hit rates. This estimator requires only a small number of training tokens (a few hundred) to adapt to high-drift layers, with minimal latency overhead during inference.

## Experimental Results
The authors demonstrate 5-14% reduction in time per output token (TPOT) across multiple MoE architectures and hardware configurations. For Qwen3-30B-A3B on an A6000 GPU, prefetching yields a 9-14% TPOT reduction (Figure 8), with larger improvements at longer sequence lengths (e.g., 14% at 65536 context length versus 9% at 1024). On more powerful GPUs (A100 and GH200), the maximum improvement is limited to 5-8% due to higher compute throughput.

Accuracy results vary by model:
- For Qwen3-30B-A3B, router-based prefetching reduces HumanEval accuracy from 93.9% to 86.0% (Table 1), but this is mitigated to 91.5% with the lightweight estimator (Est-PF).
- For GPT-OSS-120B, router-based prefetching maintains accuracy within 0.7% of baseline (97.0% vs 96.3% on HumanEval).
- The paper reports accuracy on multiple benchmarks: HumanEval (coding), MBPP+ (coding), GSM8k (math), AIME24/AIME25 (math reasoning), and StrategyQA (common sense reasoning).

## Related Work
The authors position their work as extending prior expert prefetching approaches (Zhang et al., 2025b; Yu et al., 2025a; Eliseev et al., 2023) but with a critical innovation: executing speculated experts directly rather than treating mispredictions as cache misses. Prior approaches treated predictions as hints, requiring re-fetching of missed experts, which limited the degree to which computation could overlap with memory transfer. The authors also note that prior work did not consider the accuracy implications of executing speculated experts, which is critical for production deployment where accuracy is paramount.

## Limitations
The authors acknowledge limitations for models with high representational drift in early layers (like Qwen3-30B-A3B), where router-based speculation alone yields suboptimal accuracy. They address this with lightweight neural estimators, but the paper doesn't explore how this solution scales with increasingly complex models or whether it generalizes to all MoE architectures. The evaluation is limited to MoE models with specific architectures (GPT-OSS, Qwen, GLM), and the authors don't examine how their approach scales with increasing model size beyond the tested configurations. The paper also doesn't explore the impact of this approach on multi-GPU settings or distributed inference.

## Appendix: Worked Example
Let's walk through how the quasi-hidden state prediction works for a single token processing in the Qwen3-30B-A3B model (48 layers, 128 experts per layer, hidden size 2048):

1. At layer l, the post-attention residual stream rl has a dimension of 2048 (H = 2048 as per Table 2).
2. The layer-level default vector dl is computed as a weighted combination of default vectors for the selected experts (2 experts per token, k=2), with dimension 2048.
3. The quasi-hidden state ql = LN_{l+1}(dl + rl) has a dimension of 2048.
4. For the Qwen3-30B-A3B model, the cosine similarity between ql and the true router input sl+1 is 0.85 (from Figure 3), indicating strong alignment.
5. At layer l, the router selects 2 experts (k=2) with routing weights (0.7, 0.3).
6. The prediction using ql achieves 85% recall@2 (Figure 4), meaning the top 2 predicted experts match the true top 2 experts 85% of the time.
7. For this token, the actual prediction for layer l+1 selects experts [23, 47] (based on routing weights), which matches the true selection 85% of the time.
8. The prefetched experts (23, 47) are transferred to GPU memory while the current layer computation is executing.
9. The model executes using these prefetched experts, maintaining 91.5% accuracy on HumanEval (Table 1) compared to 93.9% for the baseline.
10. This process repeats for each layer, with the prefetched experts for layer l+1 being used during layer l computation.

## References

- Vivan Madan, Prajwal Singhania, Abhinav Bhatele, Tom Goldstein, Ashwinee Panda, "Speculating Experts Accelerates Inference for Mixture-of-Experts", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19289

Tags: #mixture-of-experts #inference-optimisation #gpu-memory-overlap #cpu-gpu-transfer #llm-deployment
