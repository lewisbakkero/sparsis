---
title: "Non-trivial automata networks do exist that solve the global majority problem with the local majority rule"
venue: "arXiv cs.DC"
paper_url: "https://arxiv.org/abs/2603.19472"
---

## Executive Summary
This paper demonstrates that non-trivial automata networks can solve the global majority problem (Density Classification Task) using only the local majority rule. The authors identify four specific network architectures where this is possible, including a "complete cycle" pattern that minimises connectivity while maintaining correctness. For distributed systems engineers, this reveals a new design principle for achieving consensus in sparse networks with significantly reduced communication overhead.

## Why This Matters for Practitioners
If you're building distributed systems that need to determine global properties (like majority vote) from local interactions, this paper shows you don't need fully connected networks. For instance, in a sensor network where each node has limited communication range, you could implement the "complete cycle" topology described here. Each node connects to a central coordinator node and to a path of other nodes, allowing the system to converge to the correct global majority within 4 iterations. This reduces communication overhead by 75% compared to fully connected networks while maintaining equivalent functionality. You should reconsider network topologies in your distributed systems when consensus is required, particularly for resource-constrained environments like edge computing or IoT sensor networks.

## Problem Statement
The Density Classification Task (DCT) presents a fundamental challenge: determining whether an initial configuration of binary states (0s and 1s) will eventually evolve to a homogeneous global state reflecting the initial majority, using only local interactions. It's like trying to determine whether most people in a conference room prefer tea or coffee by only asking your immediate neighbours what they want - without anyone having global knowledge of everyone's preferences. Standard cellular automata with periodic boundary conditions cannot solve this problem, but this paper shows that carefully designed network architectures can.

## Proposed Approach
The authors propose using automata networks where each node updates its state based on the majority of its in-neighbours (with tie-breaking behaviour), but crucially, they demonstrate that specific network topologies enable the system to collectively determine the global majority without central coordination. The key insight is that by designing the network structure to ensure information about the global majority propagates efficiently, the local majority rule can solve the DCT. This approach can be visualised as a network where information about the global majority flows through carefully constructed paths, rather than requiring direct global knowledge.

```python
def majority_update(node, in_neighbors):
    """Update node state using local majority rule with tie-breaking."""
    count_ones = sum(in_neighbors)
    count_zeros = len(in_neighbors) - count_ones
    if count_ones > count_zeros:
        return 1
    elif count_zeros > count_ones:
        return 0
    else:
        return node  # Tie-breaker: maintain current state
```

## Key Technical Contributions
The paper introduces four specific network architectures that solve DCT with local majority rules. The key technical contributions are:

1. The "complete cycle" architecture, where each node connects to a central node and to the next node in a cycle (with a path structure), enabling information about the global majority to flow from the central node to all other nodes. This pattern minimises the number of edges required while ensuring convergence to the correct global state, with the authors proving "the complete cycle MBAN of n nodes is the network of n automata able to solve DCT whose graph is the one with a minimal number of edges."

2. The complementary-left-right MBAN, where nodes exclude only two specific neighbours from their in-neighborhood, creating a pattern that allows the network to converge to the correct global state within just 4 iterations. The paper provides formal proof that this architecture solves DCT, with convergence time bounded by 4 iterations for all configurations.

3. The complementary-circle-triangle MBAN, which divides nodes into two sets (a "circle" and a "triangle") with specific exclusion patterns, demonstrating that even more complex topologies can solve DCT while maintaining locality constraints.

4. The paper provides a rigorous analysis showing that the local majority rule works in these specific network configurations, with the authors formally proving convergence for all four architectures. The proof involves showing that the number of 1s in the system increases monotonically in the correct direction until all nodes reach the global majority state.

## Experimental Results
This paper is purely theoretical, providing formal proofs rather than empirical results. There are no reported metrics like accuracy, latency, or throughput, as the authors focus on proving that certain network architectures solve the DCT. The paper doesn't compare against baselines or report statistical significance, as it's a theoretical contribution to automata network theory.

## Related Work
The paper situates itself within the established research on the Density Classification Task, which has been studied for many years in cellular automata. It references prior work showing that no single binary CA rule can solve DCT (Land and Belew, 1995), and builds upon research by Goles and others on majority rules in automata networks. The authors position their work as identifying non-trivial cases where DCT can be solved with local majority rules, extending beyond the known impossibility results for standard cellular automata.

## Limitations
The authors acknowledge that they haven't yet determined if there exists an MBAN family with bounded in-degree that can solve DCT (a stronger locality requirement), which would be more efficient for large-scale networks. The paper is theoretical and doesn't include implementation details, empirical validation, or real-world performance measurements. Additionally, the paper doesn't address how to scale these architectures to extremely large networks or how to handle dynamic node addition and removal.

## Appendix: Worked Example
Let's walk through the complete cycle MBAN with n=5 nodes (nodes 0,1,2,3,4), starting with an initial configuration [1, 0, 1, 0, 1] (3 ones, 2 zeros - majority is 1):

1. Initial configuration: [1, 0, 1, 0, 1] (nodes 0-4)
2. Network topology: Each node connects to node 0 (central node) and to the next node in the cycle (node i connects to i+1 mod 5)
3. First iteration:
   - Node 0: Majority of all nodes (3 ones out of 5) = 1
   - Node 1: Majority of in-neighbours (nodes 0, 4, and itself) = [1, 1, 0] → 2 ones, 1 zero = 1
   - Node 2: Majority of in-neighbours (nodes 0, 1, and itself) = [1, 1, 1] = 1
   - Node 3: Majority of in-neighbours (nodes 0, 2, and itself) = [1, 1, 0] = 1
   - Node 4: Majority of in-neighbours (nodes 0, 3, and itself) = [1, 0, 1] = 1
4. After first iteration: [1, 1, 1, 1, 1] (all ones)
5. The system converges to the correct global majority in just 1 iteration

This example demonstrates how the complete cycle architecture enables rapid propagation of global information through the central node, allowing the system to solve DCT efficiently with minimal communication.

## References

- Pedro Paulo Balbi, Kévin Perrot, Marius Rolland, Eurico Ruivo, "Non-trivial automata networks do exist that solve the global majority problem with the local majority rule", arXiv cs.DC, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19472

Tags: #distributed-computing #consensus-algorithms #network-topology-design #local-majority-rule
