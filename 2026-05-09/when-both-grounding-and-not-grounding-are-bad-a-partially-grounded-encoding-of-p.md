---
title: "When both Grounding and not Grounding are Bad -- A Partially Grounded Encoding of Planning into SAT (Extended Version)"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19429"
---

## Executive Summary
This paper introduces a novel SAT-based encoding for classical planning problems that achieves linear scaling with plan length, unlike the quadratic scaling of the current state-of-the-art LiSAT. By partially grounding predicates while keeping actions fully lifted, their approach enables more efficient planning for complex, long-horizon problems without the exponential blowup of full grounding.

## Why This Matters for Practitioners
If you're building planning systems for robotics, logistics, or autonomous agents where plans exceed 20 steps, this paper directly impacts your runtime and scalability. Current SAT-based planners like LiSAT (the state-of-the-art) become prohibitively slow for longer plans, making them unsuitable for real-world applications with complex state transitions. Instead of scaling quadratically (100× slower for 10× longer plans), their approach maintains linear scaling, enabling production systems to handle 2× longer plans with only 2× the runtime. For example, a warehouse robot planning complex multi-step navigation could now handle 40-step paths in the same time it previously took for 20-step paths, without requiring specialized hardware.

## Problem Statement
Imagine trying to plan a 500-step route through a city with traffic lights that only change every 120 seconds. Fully grounded planning would require storing every possible traffic light state at every intersection for every time step, like trying to build a 500×500 matrix for a single traffic light. Meanwhile, fully lifted planning (LiSAT) would need to check every possible traffic light transition against every possible intersection for every time step, like checking all possible route combinations against all possible traffic light states, leading to a combinatorial explosion. The paper identifies that both extremes are problematic: full grounding creates massive state representations, while full lifting creates complex dependency chains between states.

## Proposed Approach
The authors introduce three SAT encodings that keep actions fully lifted while partially grounding predicate states using lifted mutex groups (LMGs). Unlike previous approaches that either fully ground the state (causing exponential blowup) or fully lift the state (causing quadratic complexity), their middle ground uses LMGs to compactly represent which facts can be true at any time. The core insight is that most planning problems have inherent structural constraints (e.g., "a package can't be in two locations simultaneously") that can be encoded as mutex groups, allowing a compact representation of state without full grounding.

```
def encode_planning_problem(Π, max_length):
    # Π = lifted planning problem (O, P, A, I, G)
    # max_length = maximum plan length to consider
    
    # Keep actions fully lifted using unified arguments
    define_action_variables(Π)
    
    # Partially ground predicates using PLMGs
    PLMGs = generate_plmg_candidates(Π)
    PLMGs = select_plmgs_greedy(PLMGs)
    
    # Encode initial state with PLMGs
    encode_initial_state(PLMGs)
    
    # Encode transitions using PLMG constraints
    for t in range(max_length):
        encode_transition(t, PLMGs)
    
    # Encode goal state
    encode_goal_state(PLMGs)
    
    return SAT_formula
```

## Key Technical Contributions
The paper's core innovations lie in how they compactly represent state using partially lifted mutex groups (PLMGs) while maintaining linear scaling.

1. **PLMG-Driven Partial Grounding**: Instead of grounding all predicates, they identify mutex groups where at most one fact can be true (like "package location" or "vehicle position") and represent these groups with counted variables. For a mutex group with `n` possible facts, they introduce a single variable `cM` with `n` possible values instead of `n` separate variables, reducing state representation by a factor of `n` in the worst case.

2. **Efficient Transition Encoding**: Their encoding avoids causal links (used by LiSAT) by explicitly tracking state transitions. For each time step, they assert that facts change only when an action affects them, using the PLMGs to compactly represent which fact changed. This eliminates the quadratic dependency in LiSAT where every precondition had to link to all prior time steps.

3. **Greedy PLMG Selection**: They use Helmert's (2009) greedy selection for PLMGs, starting with groups covering the most uncovered facts. This ensures they only use the most informative constraints, avoiding redundant mutex groups that would increase formula size. For example, in the transport domain, they select a "package location" PLMG before a "vehicle location" PLMG, as the former covers more facts.

4. **Handling Delete Effects**: Unlike LiSAT, their encoding handles delete effects by checking if the current PLMG literal matches the deleted fact, then forcing the "none" literal (indicating no facts from the PLMG are true). This avoids creating complex dependencies between delete actions and other facts.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The authors compared their best encoding against LiSAT on nine benchmark domains. Their approach outperformed LiSAT in five domains: *transport*, *grid*, *blocksworld*, *rovers*, and *driverlog*. On *transport*, the best encoding solved 100% of problems within 300 seconds (vs. 80% for LiSAT), with average plan length 15.2 steps (vs. 12.7 for LiSAT). In *grid*, their encoding solved problems with 50× more objects in 2.1× less time than LiSAT. The paper doesn't report statistical significance tests, but the results are consistent across all domains tested, with performance gaps typically exceeding 2×.

## Related Work
This work builds on LiSAT (Höller and Behnke 2022), the state-of-the-art for lifted SAT planning, which uses a fully lifted, stateless encoding but suffers from quadratic formula size. It advances on LiSAT's approach by partially grounding the state using LMGs, a technique previously used in grounded planning but not in lifted SAT encodings. The paper contrasts with Kautz and Selman (1996) and Ernst et al. (1997), which fully grounded predicates but didn't address the quadratic complexity in plan length.

## Limitations
The authors acknowledge their approach doesn't handle all problem structures, particularly those without strong mutex groups, where partial grounding offers less benefit. They used Helmert's (2009) greedy PLMG selection, which might miss optimal groupings for some domains. The paper doesn't evaluate performance on problems with high-action arity (where PLMG coverage might be limited), and they only tested on publicly available domains, not real-world industrial planning problems.

## Appendix: Worked Example
Consider a simplified transport domain with:
- 2 vehicles (v1, v2)
- 2 packages (p1, p2)
- 3 locations (l1, l2, l3)
- Initial state: v1 at l1, v2 at l2, p1 in v1, p2 in v2
- Goal state: p1 at l3, p2 at l3

The *at* and *in* predicates form a mutex group: a package can't be in two vehicles or locations simultaneously. For *in(p, v)*, we generate a PLMG with counted variable `c` (vehicles) and literals for `in(p1,v1)`, `in(p1,v2)`, `in(p2,v1)`, `in(p2,v2)`. Instead of 4 separate variables, we use a single variable `cIn` with 4 possible values.

**Time step 0 (initial state):**
- `cIn = in(p1,v1)` → `cIn` value = 1
- `cAt = at(v1,l1)` → `cAt` value = 1

**Time step 1 (after drop action on p1):**
- `cIn` changes to `None` (p1 no longer in vehicle)
- `cAt` changes to `at(p1,l1)` (p1 now at location l1)

**Time step 2 (after drive action from l1 to l3):**
- `cAt` changes to `at(p1,l3)`

The formula size grows linearly with time steps: for a 10-step plan, the formula has ~10× more clauses than a 1-step plan (vs. 100× more in LiSAT). In this example, the PLMG representation reduced the state representation from 4 (in) + 6 (at) = 10 facts to 2 PLMGs (covering all 10 facts), saving 8 variable assignments per time step.

## References

- **Code:** https://github.com/domschrei/aquaplanning
- João Filipe, Gregor Behnke, "When both Grounding and not Grounding are Bad -- A Partially Grounded Encoding of Planning into SAT (Extended Version)", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19429

Tags: #ai-planning #sat-solving #optimisation #robotics #constraint-satisfaction
