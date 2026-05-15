---
title: "TrustFlow: Topic-Aware Vector Reputation Propagation for Multi-Agent Ecosystems"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19452"
---

## Executive Summary
TrustFlow introduces a reputation propagation algorithm that assigns each software agent a multi-dimensional reputation vector rather than a scalar score. It uses topic-gated transfer operators to modulate trust propagation based on interaction content, achieving up to 98% multi-label Precision@5 on dense graphs while resisting sybil attacks with minimal precision impact (≤4 percentage points). This enables more nuanced and robust reputation systems for multi-agent ecosystems that scale beyond traditional scalar approaches.

## Why This Matters for Practitioners
If you're building a platform with autonomous agents that interact, delegate tasks, and transact, such as a marketplace for AI services, a multi-agent simulation environment, or a distributed task coordination system, TrustFlow provides a robust reputation foundation that scales beyond simple star ratings. Instead of relying on brittle scalar scores that get gamed, you can implement vector reputation that naturally captures multi-domain expertise and resists common attack vectors. For instance, when implementing a service marketplace, use the continuous formulation with projection operators to ensure reputation propagates only along topic-relevant paths, reducing the risk of reputation laundering by 100% (as shown in the paper's attack resistance experiments) while maintaining top performance on sparse graphs with 78% Precision@5.

## Problem Statement
Imagine trying to trust a network of thousands of autonomous agents in a marketplace where each agent serves specific domains (medicine, law, finance, coding), but they can be bribed to give each other inflated ratings, create fake identities, or form coordinated groups to manipulate rankings. Traditional reputation systems, like star ratings or PageRank, fail because they treat all interactions as equally relevant, making them vulnerable to simple attacks: a sybil ring of agents can artificially boost each other's scores by creating numerous interconnected interactions, while agents can launder reputation through intermediaries to mask malicious activity. As the paper notes, "Traditional approaches (star ratings, manual curation, call counts) fail at scale because they are easily gamed, domain-agnostic, and cannot propagate transitive trust."

## Proposed Approach
TrustFlow generalises PageRank along two axes: from scalar to vector reputation, and from topic-independent to topic-aware transfer. Each agent's reputation exists as a vector in the same embedding space as interaction content, with reputation propagation modulated by the topic of each interaction. The core iteration starts with reputation vectors and iteratively updates them based on interaction embeddings, damping factors, teleportation priors, and exogenous authority injection.

```python
def trustflow_update(R, M, e, T, C, alpha=0.85):
    R_new = np.zeros_like(R)
    for j in range(N):  # For each agent
        for i in range(N):  # For each interaction from agent i to j
            weight = row_normalize(M[i,j])
            # Topic-gated transfer using projection operator (example)
            f = np.maximum(R[i] @ e[i,j], 0) * e[i,j]
            R_new[j] += alpha * weight * f
        R_new[j] += (1 - alpha) * T[j] + C[j]
    return R_new
```

## Key Technical Contributions
TrustFlow's core innovation lies in how it handles topic-aware vector reputation propagation at implementation level:

1. **Topic-gated transfer operators**: Unlike PageRank or Topic-Sensitive PageRank, TrustFlow modulates each edge's reputation transfer by the interaction's content embedding. The projection operator (`f(R, e) = σ(R·e)·e`) provides maximum cross-domain isolation by confining reputation transfer strictly to the interaction topic direction, which explains why it achieved 78% P@5 on combined graphs (70 labelled + 612 blind edges) compared to 72% with squared gating.

2. **Lipschitz-1 operators with convergence guarantees**: TrustFlow constructs a family of transfer operators (projection, squared gating, scalar-gated, Hadamard relu, hybrid) that all satisfy a Lipschitz-1 bound, guaranteeing convergence via the contraction mapping theorem. For squared gating, the proof relies on the property that `∥(R1 − R2) ⊙ e²∥ ≤ ∥R1 − R2∥ · ∥e²∥∞ ≤ ∥R1 − R2∥` since `∥e∥₂ = 1` implies `|eₖ| ≤ 1`.

3. **Blind edge handling with proxy embeddings**: For interactions without inspectable content (e.g., encrypted API calls), TrustFlow uses a mean embedding proxy (`eᵢⱼ = avg(pᵢ, pⱼ)`) rather than discarding the edge. This preserves directional information (achieving 74% directional preservation for projection on the proxy vs. 0.004 on uniform embeddings), which explains why projection could still achieve 78% P@5 on combined graphs.

4. **Negative trust edges for moderation**: TrustFlow extends its core algorithm to include negative trust edges (for spam, harmful content, etc.) with convergence guarantees. The iteration becomes `Rnew[:, d] = α(Mᵀₚₒₛ,𝒹R[:, d] − βMᵀₙₑg,𝒹R[:, d]) + (1 − α)T[:, d] + C[:, d]`, converging when `α(1 + β) < 1`. With `α = 0.85` and `β = 0.15`, this gives `0.85 × 1.15 = 0.9775 < 1`, satisfying the condition for convergence.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
TrustFlow was evaluated on a benchmark of 50 agents across 8 domains (medicine, law, finance, coding, cybersecurity, education, creative, data science) with 6 cross-domain specialists. Key results:

- **Accuracy**: 78% multi-label Precision@5 on sparse graphs (70 labelled edges + 612 blind edges) with the projection operator, compared to 74% for the discrete formulation.
- **Attack resistance**: TrustFlow resisted sybil attacks with at most 4 percentage-point precision impact: cross-domain sybil (78.0% vs baseline 78.0%), same-domain sybil (74.0% vs 78.0% baseline), reputation laundering (78.0%), and vote rings (80.0%).
- **Transfer operators**: Projection achieved the highest combined-graph P@5 (78%) due to its rank-1 output structure, while squared gating converged 3× faster than projection in unnormalized mode (4 vs 12 iterations).
- **KL-divergence gating**: With λ = 1.0, self-alignment increased by 24.5% with a P@5 improvement of 8pp (68% → 76%) on labelled-only graphs.

The paper does not report statistical significance tests for the accuracy improvements, though it does report the exact numbers and metrics.

## Related Work
TrustFlow builds on PageRank (Brin and Page 1998) and Topic-Sensitive PageRank (TSPR, Haveliwala 2002) but fundamentally generalises them in two directions: vector reputation (instead of scalar) and topic-gated transfer (instead of topic-biased teleportation). Unlike TSPR, which uses the same transition matrix for all topics and only varies the teleportation vector, TrustFlow makes every edge's transfer depend on the interaction's semantic content. This allows TrustFlow to distinguish "A links to B in a medical context" from "A links to B in a coding context" without requiring pre-defined topic categories.

## Limitations
The paper does not test TrustFlow on extremely large-scale systems (beyond 50 agents) or with more complex attack strategies beyond the four tested. It also notes that cross-domain specialists face challenges in discrete formulations (must split reputation across domains), though the continuous formulation handles this naturally. The evaluation also doesn't explore the impact of different embedding spaces beyond E5-small (384 dimensions), though the paper mentions mean-centering embeddings as a key preprocessing step.

## Appendix: Worked Example
Consider a simple interaction between two agents: a medical agent (A) and a data science agent (B), where agent A delegates a medical task to agent B. Agent A has reputation vector R[A] = [0.9, 0.1] (strong medical expertise, weak data science), and the interaction embedding e[A→B] = [0.8, 0.2] (medical topic).

Using the projection operator:
1. Compute cosine similarity: R[A] · e[A→B] = (0.9×0.8) + (0.1×0.2) = 0.72 + 0.02 = 0.74
2. Apply ReLU: σ(0.74) = 0.74
3. Scale the embedding: f(R[A], e[A→B]) = 0.74 × e[A→B] = [0.592, 0.148]

This output vector [0.592, 0.148] is aligned with the interaction topic (medical), preserving the medical direction while reducing the data science component. After 12 iterations (for projection in unnormalized mode), the reputation vector for agent B would be updated to incorporate this medical expertise, while maintaining separation from unrelated domains.

## References

- Volodymyr Seliuchenko, "TrustFlow: Topic-Aware Vector Reputation Propagation for Multi-Agent Ecosystems", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19452

Tags: #multi-agent #reputation-system #graph-algorithms #trust-propagation #vector-representation
