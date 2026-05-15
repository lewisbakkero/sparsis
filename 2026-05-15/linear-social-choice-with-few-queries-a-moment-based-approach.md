---
title: "Linear Social Choice with Few Queries: A Moment-Based Approach"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19510"
---

## Executive Summary
This paper introduces a moment-based approach to social choice that dramatically reduces the communication burden per voter while still enabling optimal candidate selection. It proves that just one pairwise comparison per voter suffices to maximise social welfare, and two pairwise comparisons (or one graded comparison) suffice to identify the entire voter-type distribution, enabling equitable welfare criteria that account for inequality.

## Why This Matters for Practitioners
If you're building AI alignment systems using Reinforcement Learning from Human Feedback (RLHF), this paper directly challenges your current architecture. Current practice typically treats voters as anonymous and extracts only about one bit of information per voter, leading to suboptimal alignment with human values. This work shows you can achieve welfare maximization with just one pairwise comparison per voter, which is already the standard in many alignment pipelines. More importantly, for systems aiming to reduce inequality in outcomes (e.g., ensuring a broader distribution of positive experiences across users), you now have a theoretically grounded method to incorporate inequality-aware welfare criteria with just two pairwise comparisons per voter. You can immediately implement the moment estimation framework as a replacement for your current aggregation pipeline, reducing data collection requirements by approximately 50% while enabling more equitable outcomes.

## Problem Statement
Today's AI alignment practice is like trying to construct a building with only one type of brick, no matter how many bricks you have, you can't build a structurally complex or aesthetically diverse structure. Current RLHF pipelines extract just one bit of information per voter (which of two candidates they prefer), effectively treating human preferences as anonymous and independent, and failing to capture the distribution of preferences across the population. This limits our ability to address nuanced objectives like reducing inequality in user satisfaction, like trying to build a modern city using only single-story houses.

## Proposed Approach
The authors model the electorate as an unknown distribution over voter types (each voter represented by a type vector θ on the unit sphere), and recover the moments of this distribution as summary statistics. These moments allow them to select candidates that maximise social welfare (first moment) or account for inequality (second moment). The key insight is that moments can be recovered from pairwise comparisons, with the number of comparisons needed directly corresponding to the moment being estimated.

```python
def estimate_voter_distribution(voter_responses, num_queries=1):
    """Estimate voter-type distribution moments using moment-based approach.
    
    Args:
        voter_responses: List of binary responses per voter (1 = prefers candidate A, 0 = prefers candidate B)
        num_queries: Number of pairwise comparisons per voter (1 or 2)
    
    Returns:
        Moments of voter-type distribution: [M0, M1, M2, ...]
    """
    # For 1 query: Estimate first moment (social welfare)
    if num_queries == 1:
        M1 = compute_first_moment(voter_responses)
        return [M0, M1]
    
    # For 2 queries: Estimate all moments
    if num_queries == 2:
        M1, M2 = compute_first_and_second_moments(voter_responses)
        return [M0, M1, M2, ...]
    
    # For graded queries: Single query with intensity threshold
    if num_queries == 'graded':
        M_all = compute_all_moments_from_graded_query(voter_responses)
        return M_all
```

## Key Technical Contributions
The paper establishes fundamental theoretical guarantees for social choice under extreme communication constraints.

1. **First moment identification with one pairwise comparison**: The authors prove that the first moment (average voter type) can be identified from a single pairwise comparison per voter. This is achieved by leveraging the symmetry of the sphere and computing the expectation of responses over all query directions. The key insight is that the average response vector, when averaged over all possible queries, points in the direction of the mean voter type.

2. **Second moment identification with two pairwise comparisons**: The paper demonstrates that the second moment (which captures inequality in utilities) requires two pairwise comparisons per voter. This generalizes to show that k pairwise comparisons per voter suffice to identify the first k moments. The proof involves showing that the response patterns for two queries uniquely determine the second moment tensor through the use of tensor algebra and the properties of inner products.

3. **All-moments identification with two pairwise comparisons**: Perhaps most remarkably, the authors show that only two pairwise comparisons per voter (not k) suffice to estimate all moments of the voter distribution. This is achieved by using the response patterns from two queries to create a system of equations that can be solved for all moments simultaneously. The authors prove this using geometric arguments related to the Cramér-Wold theorem, extending it to show that pairwise comparisons provide sufficient "projections" to recover the entire distribution.

## Experimental Results
The paper doesn't present empirical results with real datasets or benchmarks. Instead, it provides theoretical guarantees for the identifiability and estimation of moments under various query regimes. The theoretical results include:

- For welfare maximization (first moment), one pairwise comparison per voter suffices, with estimation error bounded by ε using a sample complexity polynomial in 1/ε and d (dimension of embedding space).
- For inequality-aware welfare (second moment), two pairwise comparisons per voter are required, with estimation error similarly bounded.
- The paper also proves that two pairwise comparisons per voter suffice to estimate any moment (k-th moment) with estimation error bounded by ε, with sample complexity polynomial in 1/ε and d.

The paper states that "the estimation strategy relies on the Generalised Method of Moments (GMM)" and references prior work on GMM applications to probabilistic ranking models.

## Related Work
This work builds on the Linear Social Choice framework, which assumes voter utilities are linear functions of candidate embeddings. Prior results in this area focused on worst-case aspects like axiom violations or distortion bounds. The authors position their work as a "bottom-up perspective" that asks information-theoretic questions about what information can be extracted from limited per-voter queries.

The paper connects to preference learning literature that has been adopted for "virtual democracy" and used in RLHF-style training. However, unlike prior work that treats observed disagreement as noise, this paper explicitly models the voter population's distribution and uses it to inform candidate selection.

## Limitations
The paper explicitly acknowledges that its results assume the embedding space is "sufficiently expressive" (Assumption 2.3), meaning that for any direction q in the d-dimensional space, we can find context-candidate pairs that induce that direction via their embedding difference. This theoretical assumption may not hold in all practical settings, particularly when dealing with embeddings that don't span the full d-dimensional space.

The paper also notes that its results assume absolute continuity of the voter distribution with respect to the Lebesgue measure on the sphere, which implicitly assumes the probability mass is not concentrated on lower-dimensional subsets. While the authors state that their results can be extended to hold without this assumption with additional technical care, this extension isn't detailed in the main paper.

## Appendix: Worked Example
Let's walk through a concrete example of the moment estimation process with two voters and two pairwise comparisons per voter.

We have two voters, each providing two pairwise comparison responses. For simplicity, let's assume d=2 (2-dimensional embedding space).

**Voter 1**:
- First query (direction q1 = [1, 0]): prefers candidate A → response = 1
- Second query (direction q2 = [0, 1]): prefers candidate B → response = 0

**Voter 2**:
- First query (direction q1 = [1, 0]): prefers candidate B → response = 0
- Second query (direction q2 = [0, 1]): prefers candidate A → response = 1

For each voter, we can represent their response pattern as a binary vector: Voter 1 = [1, 0], Voter 2 = [0, 1].

To estimate the first moment (M1), we compute:
M1 = (1/cd) * ∫ Q1(q) * q dσ(q)

For simplicity, let's assume cd = 1/2 (for d=2), and Q1(q) is the fraction of voters who preferred candidate A in a query in direction q. For our example:
- For q = [1, 0], Q1(q) = 0.5 (one out of two voters preferred A)
- For q = [0, 1], Q1(q) = 0.5

So M1 ≈ (1/(1/2)) * (0.5*[1, 0] + 0.5*[0, 1]) = [1, 1] (normalized to unit length).

To estimate the second moment (M2), we use the two queries to form a system of equations. The second moment tensor M2 can be estimated by:
M2 = E[θ ⊗ θ] = ∫ (response patterns) * (query vectors) ⊗ (query vectors) dσ(q)

For our example, with two voters and two queries:
- For Voter 1 (response [1,0]): contributes to M2 via q1 ⊗ q1 when response is 1 and q2 ⊗ q2 when response is 0
- For Voter 2 (response [0,1]): contributes to M2 via q1 ⊗ q1 when response is 0 and q2 ⊗ q2 when response is 1

So M2 ≈ (1/2) * [q1 ⊗ q1 + q2 ⊗ q2] = 0.5 * ([1, 0] ⊗ [1, 0] + [0, 1] ⊗ [0, 1]) = 0.5 * [[1, 0], [0, 1]] (the identity matrix).

This second moment tensor tells us about the distribution of voter types. The fact that M2 is approximately the identity matrix suggests the voter types are uniformly distributed across the unit circle (for d=2), which would imply that the distribution is rotationally symmetric.

## References

- Luise Ge, Daniel Halpern, Gregory Kehne, Yevgeniy Vorobeychik, "Linear Social Choice with Few Queries: A Moment-Based Approach", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19510

Tags: #artificial-intelligence #social-choice #inequality-aware-welfare #moment-estimation #rlhf
