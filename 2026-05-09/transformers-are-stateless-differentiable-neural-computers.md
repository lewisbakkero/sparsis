---
title: "Transformers are Stateless Differentiable Neural Computers"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19272"
---

## Executive Summary
This paper formally proves that a causal Transformer layer is mathematically equivalent to a stateless Differentiable Neural Computer (sDNC), where attention mechanisms implement content-based reads from a write-once memory of value vectors. This unifies Transformers under a memory-centric framework without requiring recurrent states or dynamic memory updates, offering a new lens for understanding and enhancing production systems.

## Why This Matters for Practitioners
When building long-context applications (e.g., legal document analysis or medical record summarisation), you're implicitly facing the quadratic memory growth inherent in Transformers' write-once value matrix. This paper reveals that *every* Transformer layer is effectively a memory system where value vectors are appended but never modified, meaning a 10,000-token sequence requires 10,000× memory storage for values. For production systems, this implies:
- **Avoid over-engineering context windows**: Instead of naively increasing sequence lengths (which causes memory explosions), implement selective write mechanisms (like those in DNCs) to dynamically update memory. The paper’s equivalence shows that adding *controlled* memory updates (e.g., discarding low-value tokens) would reduce memory use without sacrificing performance.
- **Debug memory bottlenecks faster**: If your model fails on long sequences, check whether attention weights are over-relying on distant tokens (indicating memory overflow). Use the sDNC framework to verify if value vectors are growing unbounded (e.g., via `memory.shape[0]` checks during inference).
- **Replace ad-hoc caching with principled memory management**: The paper’s cross-attention analysis shows encoder memory behaves as a fixed, write-once storage, providing a blueprint to formalise retrieval augmentation systems instead of patching with key-value caches.

## Problem Statement
Current Transformer implementations treat attention as a "relational black box," obscuring how information is stored. It’s like managing a library where you can only add books (append-only) but never reorganise shelves (no overwriting) or remove outdated volumes (no erasure). This makes scaling past 2,000 tokens inefficient, your system consumes memory proportional to sequence length squared, not linearly, causing crashes in production when processing long documents.

## Proposed Approach
The authors introduce the stateless DNC (sDNC): a memory-augmented architecture where:
- The controller is feedforward (no recurrent state)
- Memory grows via append-only writes (value vectors)
- Content-based reads match attention
- Multi-head attention = parallel read heads

This matches Transformers perfectly: keys/values from linear projections → memory → attention reads. For encoder-decoder models, cross-attention uses two memories (encoder: fixed; decoder: causal).

```python
def sDNC_read(memory: torch.Tensor, key: torch.Tensor, d_k: int) -> torch.Tensor:
    """Equivalent to a single Transformer attention head.
    
    Args:
        memory: Write-once matrix of value vectors [t, W]
        key: Query vector for current token [W]
        d_k: Key dimension for scaling
    
    Returns:
        Read vector: weighted sum over memory [W]
    """
    weights = torch.softmax(memory @ key / torch.sqrt(torch.tensor(d_k)), dim=0)
    return (memory.T @ weights).squeeze()
```

## Key Technical Contributions
The paper’s core insight is that Transformers *are* memory systems, not just relational models. This reshapes how we engineer them:

1. **Write-once memory as the value matrix**: The Transformer’s `value` vectors form a *fixed, append-only memory*, each token’s representation is stored once and never modified. This explains why attention weights decay for distant tokens: the system must traverse a growing memory, not a dynamic state. Unlike DNCs (which allow overwriting), this design simplifies training but causes quadratic memory growth.

2. **Multi-head attention as parallel read heads**: Each head implements a *separate content-based lookup* over the same memory. The paper proves this isn’t an analogy, it’s a direct equivalence where `H` heads correspond to `H` independent read heads. This clarifies why multi-head attention improves performance: it enables multiple parallel "searches" for context within the memory, rather than aggregating all context into one query.

3. **Cross-attention as two-memory retrieval**: Encoder memory (fixed, populated by encoder values) acts as a *read-only knowledge base*, while decoder memory (causal, write-once) stores *progressive state*. This mirrors how humans process information: the encoder "memorises" the input document, and the decoder "retrieves relevant facts" while building the output. The paper shows this structure is *exactly* an sDNC with two memories, eliminating ambiguity in how encoder-decoder models interact.

## Experimental Results
This is a theoretical paper with no empirical results. The authors provide formal proofs (Section 4) and examples (Section 3), but no benchmarks, datasets, or performance metrics. The equivalence is mathematically derived from Transformer and DNC architectures, with no training or evaluation on real-world tasks.

## Related Work
The paper builds on prior observations that Transformers behave like memory systems [3, 5, 9] but makes the correspondence *exact*. Unlike Katharopoulos et al. [5] (which restricted to limited settings), this work generalises to full Transformer layers and cross-attention. It also clarifies misconceptions: while [3] called Transformers "soft reasoning systems," this paper proves they are *memory systems* with fixed write semantics. The authors explicitly contrast their work with DNCs [4] (which add dynamic memory control) and Neural Turing Machines [4] (which use recurrence).

## Limitations
The paper’s equivalence holds only for standard Transformers without recurrence or dynamic memory. For architectures like Recurrent Memory Transformers [2], the sDNC framework doesn’t apply directly. Crucially, the authors acknowledge Transformers *lack* DNC capabilities like memory erasure or temporal linking, hence the need for auxiliary mechanisms (e.g., retrieval augmentation [1]) to extend memory horizons. Production engineers should note this: the sDNC view explains *why* systems like `key-value caching` are necessary (to mimic DNC-style memory updates), but doesn’t provide the implementation.

## Appendix: Worked Example
Consider a 3-token sequence processed by a single Transformer layer with `d_k = 64`:

- **Input tokens**: `x₁ = [0.2, 0.5]`, `x₂ = [-0.1, 0.3]`, `x₃ = [0.4, -0.2]` (2-dimensional for simplicity)
- **Linear projections**: `W_K = [[1, 0], [0, 1]]` (identity), `W_V = [[1, 0], [0, 1]]` (identity)
- **Step 1 (t=1)**:
  - `k₁ = W_Kᵀ x₁ = [0.2, 0.5]`
  - `v₁ = W_Vᵀ x₁ = [0.2, 0.5]`
  - Memory `M₁ = [[0.2, 0.5]]` (1×2 matrix)
  - Attention weights: `w₁ = softmax([ (0.2*0.2 + 0.5*0.5)/8 ]) = softmax([0.34/8]) ≈ [0.52]`
  - Output: `z₁ = M₁ᵀ w₁ ≈ [0.10, 0.26]` (matches `v₁` since memory is empty)
- **Step 2 (t=2)**:
  - `k₂ = [ -0.1, 0.3 ]`
  - `v₂ = [ -0.1, 0.3 ]`
  - Memory `M₂ = [[0.2, 0.5], [-0.1, 0.3]]` (2×2)
  - Weights: `w₂ = softmax( [ (0.2*-0.1 + 0.5*0.3)/8, (-0.1*-0.1 + 0.3*0.3)/8 ] ) = softmax([0.13/8, 0.10/8]) ≈ [0.57, 0.43]`
  - Output: `z₂ = (M₂ᵀ @ w₂) ≈ [0.05, 0.40]`
- **Step 3 (t=3)**:
  - `k₃ = [0.4, -0.2]`
  - `v₃ = [0.4, -0.2]`
  - Memory `M₃ = [[0.2, 0.5], [-0.1, 0.3], [0.4, -0.2]]` (3×2)
  - Weights: `w₃ = softmax( [ (0.2*0.4 + 0.5*-0.2)/8, ... ] )` → *computed similarly*
  - Output: `z₃ = M₃ᵀ @ w₃` (weighted sum of all three values)

*Note: The sDNC perspective clarifies why `z₃` depends on all prior tokens, the memory is write-once and grows with each token.*

## References

- Bo Tang, Weiwei Xie, "Transformers are Stateless Differentiable Neural Computers", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19272

Tags: #machine-learning #memory-augmented #transformer-memory
