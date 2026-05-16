---
title: "The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19664"
---

## Executive Summary
This paper proves the key-value (KV) cache in transformer inference is entirely redundant, as keys and values are deterministic projections of the residual stream. The authors introduce KV-Direct, a bounded-memory inference scheme that stores residual vectors (5 KB per token for Gemma 3-4B) instead of KV pairs (136 KB), achieving 2.5× peak memory reduction without token-level accuracy loss over 20 conversation turns.

## Why This Matters for Practitioners
If you're operating large language model inference systems at scale (e.g., 10k+ concurrent users with long context windows), this paper directly impacts your memory budgeting. KV-Direct eliminates the need for complex cache eviction policies (H2O, StreamingLLM, etc.) that degrade output fidelity to 5, 28% match at moderate budgets. Instead, you can replace your KV cache implementation with residual checkpointing: on Gemma 3-4B, this reduces peak memory from 103 MB to 42 MB over 20 turns while maintaining 100% token match. For every production system where context length grows linearly with user sessions (e.g., chatbots, long-document QA), this means fewer memory spikes, simpler deployment, and no quality trade-offs.

## Problem Statement
Current inference systems treat the KV cache as essential state, like assuming a car's steering wheel is irreplaceable because it’s the only input mechanism. But the authors prove it’s just a redundant copy: the steering wheel (KV cache) is derived from the driver’s hand position (residual stream), not a distinct information source. In practice, this means every cache eviction strategy (H2O, TOVA, etc.) is unnecessarily sacrificing output quality to save memory, when the memory saving could be achieved without degradation.

## Proposed Approach
KV-Direct replaces the standard KV cache with residual checkpointing: instead of storing keys/values for every token at every layer, it stores one residual vector per token (shared across layers) and recomputes KV on demand. This leverages the residual stream’s Markov property, future outputs depend solely on current residuals, not historical KV. The system maintains a fixed-size cache for recent tokens and recomputes evicted entries from residuals. 

```python
def KV_DIRECT(x_t, residual_checkpoints, cache_budget):
    # Compute residual from input token
    h = RMSNorm(embed(x_t))
    
    # Recompute KV from checkpoints for evicted tokens
    K_old, V_old = reconstruct_KV(residual_checkpoints, layer)
    
    # Assemble full KV sequence for attention
    K_all = concatenate(K_old, cache_keys, current_K)
    V_all = concatenate(V_old, cache_values, current_V)
    
    # Proceed with standard transformer computation
    out = attention(Q, K_all, V_all)
    return softmax(out)
```

## Key Technical Contributions
The paper's core insight is structural, not heuristic. Each contribution specifically addresses why prior cache optimisation approaches failed to achieve lossless inference:

1. **Exact reconstruction via algebraic identity**: Keys/values are not approximated but *exactly recomputed* from residuals using frozen weight matrices (Equations 13, 14). For full-attention layers, this is *bit-identical* (max |ΔK| = max |ΔV| = 0 across all models), not approximate. This differs from prior work like MiniCache or Multi-head Latent Attention, which introduced approximation error by compressing KV.

2. **Sliding-window boundary specification**: For models using window-relative RoPE (e.g., Gemma 3-4B), key reconstruction requires the local position index (not absolute position), but value reconstruction remains universal. This precise boundary (verified across 29 sliding-window layers) explains why prior eviction strategies failed *at the mechanism level*.

3. **Per-token memory ratio formalisation**: KV-Direct’s memory advantage scales with model architecture (Equation 25: *ρ* = 2*L*n_kv*d_head / d_hidden). For Gemma 3-4B, *ρ* = 27.2× (5 KB vs 136 KB per token), while Qwen3-0.6B achieves *ρ* = 56×. This quantifies *why* memory savings grow with model complexity, unlike heuristic approaches like quantization, which offer fixed per-entry savings.

## Experimental Results
The paper verifies exact reconstruction across six models (135M, 4B parameters, four architecture families) with 0 reconstruction error (Table 2). For token-identical generation (RQ2), all models produced 30/30 identical tokens under KV-Direct vs. full cache (Table 3). Over 20 conversation turns, KV-Direct held peak memory at 42 MB vs. 103 MB for standard cache (2.5× reduction). Against five eviction baselines (H2O, StreamingLLM, SnapKV, TOVA, window-only), KV-Direct maintained 100% token match at all budgets while baselines degraded to 5, 28% match (with KL divergences of 7, 14). Recomputation latency was 0.2, 0.3× the cache-read time at moderate batch sizes (500 tokens), confirming recomputation is faster than cache reads when memory bandwidth is the bottleneck (Section 5.8).

## Related Work
KV-Direct challenges the entire field’s assumption that KV cache stores essential information. Prior work (e.g., H2O, StreamingLLM) treated eviction as permanent loss (Section 2), while KV-Direct proves it’s *not* lossy. It builds on mechanistic interpretability (circuits framework [36], induction head analysis [37]) but *redirects* it from explanation to optimisation: instead of *why* residuals carry information, it *uses* them to eliminate redundancy. Unlike KVPR [31] or HybridServe [32], KV-Direct achieves exact output fidelity without approximation.

## Limitations
The paper only verifies greedy (argmax) decoding; beam search or sampling may behave differently. The sliding-window boundary (Section 5.1) requires position tracking for keys, adding minimal overhead but not addressed in the current implementation. The experiments focus on inference; training with residual checkpointing is untested. The authors note that "the residual stream satisfies a Markov property" *only* for pre-norm architectures (RMSNorm), which excludes some post-norm models.

## Appendix: Worked Example
Consider a single token in a Gemma 3-4B conversation. The residual vector *h* (2560 elements, bfloat16 = 5 KB) replaces its KV cache (136 KB). To compute attention for this token:
1. **Reconstruct keys**: *K_recon = RoPE(h, p) × W_k*  
   (Uses absolute position *p*; for sliding-window layers, *p* is adjusted to local window index)
2. **Reconstruct values**: *V_recon = h × W_v*  
   (No position encoding needed)
3. **Compare to cache**: *|K_recon - K_cache| = 0* (bit-identical) and *|V_recon - V_cache| = 0* (Table 2).

Over 20 turns, this replaces 103 MB of cache with 42 MB of residuals (2.5× reduction). For 1000 concurrent users with 20-turn contexts, this saves **61 GB of memory** (103 MB, 42 MB = 61 MB per user × 1000 users) without affecting output quality.

## References

- **Code:** https://github.com/Kaleemullahqasim/KV-Direct.
- Kaleem Ullah Qasim, Jiashu Zhang, Muhammad Kafeel Shaheen, Razan Alharith, Heying Zhang, "The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19664

Tags: #machine-learning #memory-efficiency #transformer-optimisation
