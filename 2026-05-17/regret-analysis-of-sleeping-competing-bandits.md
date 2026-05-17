---
title: "Regret Analysis of Sleeping Competing Bandits"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19700"
---

## Executive Summary
This paper introduces the "Sleeping Competing Bandits" framework, which extends traditional competing bandits to model dynamic availability of both players and arms over time. It establishes tight regret bounds, Ω(N(K - N + 1) log T/Δ²) for lower bounds and O(NK log T/Δ²) for the proposed algorithm, and demonstrates asymptotic optimality in regimes where K > N. For practitioners building dynamic matching systems, this work provides the theoretical foundation to design more efficient allocation algorithms that account for real-world availability fluctuations.

## Why This Matters for Practitioners
If you're building a dynamic matching system like a food delivery platform, ride-hailing service, or cloud resource allocator where both sides (e.g., couriers/orders or workers/jobs) have variable availability, this paper warns against assuming constant availability. Traditional competing bandits algorithms would incur significantly higher regret (up to N(K - N + 1) times higher in worst-case scenarios), leading to suboptimal assignments and lower platform efficiency. Instead, implement algorithms that explicitly model availability, using UCB-based preference rankings with exploration-exploitation switching, to achieve near-optimal performance. For production systems, prioritize monitoring the ratio of available arms (K) to players (N); when K is significantly larger than N (e.g., K > 5N), the proposed algorithm's O(NK log T/Δ²) regret bound guarantees better scalability than alternatives.

## Problem Statement
Imagine a food delivery platform where couriers become unavailable due to traffic, battery issues, or other commitments, and orders appear and disappear dynamically, sometimes in waves, sometimes sparsely. Traditional matching algorithms assume all couriers are always available and orders arrive steadily. This leads to wasted capacity during high-courier availability periods and failed matches during low-availability times. The paper frames this as "Sleeping Competing Bandits," where both players (couriers) and arms (orders) can "sleep" (become unavailable) at arbitrary times, creating a dynamic matching problem that requires new theoretical foundations.

## Proposed Approach
The framework extends standard competing bandits by introducing two key concepts: 
1. Player-pessimal stable regret (measuring performance relative to the worst-case stable matching) and 
2. Player-optimal stable regret (measuring performance relative to the best-case stable matching).

The core algorithm, Awake Centralized UCB (AC-UCB), combines UCB-based exploration with Gale-Shapley stable matching. For each available player, it computes Upper Confidence Bounds for each available arm, sorts arms by these bounds, and uses player-proposing Gale-Shapley to form a stable matching. The algorithm then observes rewards, updates UCBs, and repeats. This creates a natural extension of standard UCB to the sleeping setting.

```python
def AC_UCB(Pt, At, preferences, capacities):
    """Awake Centralized UCB Algorithm for Sleeping Competing Bandits"""
    # Compute UCB for all player-arm pairs
    for pi in Pt:
        for aj in At:
            if Ti,j > 0:
                UCBi,j = empirical_mean + sqrt(log(ti) / Ti,j)
            else:
                UCBi,j = float('inf')  # Ensure exploration
    
    # Build preference ranking for each player
    for pi in Pt:
        sigma_i = sorted(At, key=lambda aj: UCBi,j, reverse=True)
    
    # Execute player-proposing Gale-Shapley
    matching = gale_shapley(Pt, At, sigma_i, preferences, capacities)
    
    # Observe rewards and update
    for pi in Pt:
        if matching[pi] != 0:
            reward = observe_reward(pi, matching[pi])
            update_empirical_mean(pi, matching[pi], reward)
            update_count(pi, matching[pi])
```

## Key Technical Contributions
The paper establishes a novel theoretical framework for dynamic matching systems, with these key technical innovations:

1. **Extended stable matching definition for dynamic availability**: Unlike standard stable matching, this framework handles rounds where availability changes arbitrarily. The definition accounts for unmatched players as virtual "arm 0" with reward 0, and adjusts blocking pair conditions to consider varying capacities. Crucially, it shows that when all players and arms are always available (a special case), the regret definitions reduce to the established competing bandits framework.

2. **UCB-based preference construction with adaptive confidence bounds**: The AC-UCB algorithm uses Upper Confidence Bounds to construct player preferences, but with a critical refinement, the confidence intervals scale with the number of rounds each player is available (Ti) rather than total rounds. This ensures that players with limited availability (e.g., part-time couriers) don't get penalized unfairly in preference rankings.

3. **Asymptotic optimality proof for the regime K > N**: The paper proves that AC-UCB achieves O(NK log T/Δ²) regret, matching the lower bound Ω(N(K - N + 1) log T/Δ²). This optimality holds specifically when the number of arms (K) is relatively larger than players (N), a common scenario in real-world applications like food delivery where the order pool typically exceeds available couriers.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper provides theoretical regret bounds rather than empirical results on real datasets. The key quantitative findings are:
- **Lower bound**: Ω(N(K - N + 1) log T/Δ²) under reasonable assumptions (Theorem 2)
- **Upper bound for AC-UCB**: O(NK log T/Δ²) (Theorem 3)
- **Upper bound for AC-ETGS**: O(NK² log T/Δ²) (Theorem 4)

The paper does not provide empirical validation on real-world datasets or comparison with baseline algorithms (e.g., standard UCB or competing bandits), as it focuses on theoretical analysis. The authors acknowledge this as a potential direction for future work.

## Related Work
This paper builds on three key strands:
- **Sleeping Bandits** [19] (which allows arms to be unavailable but assumes single-player) by extending to a two-sided market with mutual preferences.
- **Competing Bandits** [28] (which models stable matching in bandits) by generalising to handle arbitrary availability.
- **Multi-Player Bandits** [7, 34] by introducing a centralised setting where the platform coordinates selections based on global availability.

The paper distinguishes itself by addressing the dynamic availability problem while preserving the stable matching property, unlike combinatorial bandits [12, 27], which treat arms as passive resources without mutual preferences.

## Limitations
The authors acknowledge the theoretical focus: "This paper focuses on regret analysis and does not provide empirical validation on real-world datasets." The framework assumes perfect knowledge of arms' preference rankings at the start of each round, which may be unrealistic in practice, real systems would need to learn these rankings as well. Furthermore, the optimality guarantee holds only when K > N, a limitation that could affect applications where the number of players exceeds arms.

## Appendix: Worked Example
Let's walk through a simplified example of AC-UCB with 2 players (N=2), 4 arms (K=4), where arms are available in rounds as follows:
- Player 1 (p1) available in rounds 1-5 (T1=5)
- Player 2 (p2) available in rounds 3-7 (T2=5)
- Arms available: 
  - Round 1: {a1, a2}
  - Round 2: {a1, a2, a3}
  - Round 3: {a1, a2, a3, a4}
  - Round 4: {a1, a2, a3, a4}
  - Round 5: {a1, a2, a3}
  - Round 6: {a1, a2, a4}
  - Round 7: {a1, a2, a3, a4}

Suppose μi,j = 0.6 for preferred arm, 0.4 for suboptimal (Δ=0.2). In round 3, p1 has seen a1: 2/3 (0.67), a2: 1/3 (0.33), a3: 0/0 (inf), a4: 0/0 (inf). UCBs:
- a1: 0.67 + sqrt(log(3)/2) ≈ 0.67 + 0.55 = 1.22
- a2: 0.33 + sqrt(log(3)/1) ≈ 0.33 + 1.73 = 2.06
- a3: inf
- a4: inf

p1's preference: [a2, a3, a4, a1] (sorted descending by UCB). Similarly, p2's preference might be [a1, a2, a3, a4] based on its own observations.

The platform runs Gale-Shapley with these preferences and fixed arm preferences (a1 ≻ a2 ≻ a3 ≻ a4 for all arms). The matching might be (p1-a2, p2-a1), with observed rewards 0.33 (p1-a2) and 0.67 (p2-a1). These update empirical means and counts for future rounds.

## References

- Shinnosuke Uba, Yutaro Yamaguchi, "Regret Analysis of Sleeping Competing Bandits", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19700

Tags: #multi-agent #bandits #stable-matching #online-learning #reinforcement-learning
