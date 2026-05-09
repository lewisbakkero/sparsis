---
title: "On the Ability of Transformers to Verify Plans"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19954"
---

## Executive Summary
Transformer models often struggle to reliably verify plans in AI planning tasks, particularly as plans grow longer. This paper establishes theoretical conditions under which transformers can learn to verify plans with perfect length generalisation, showing that delete-free and well-formed planning domains are provably learnable. Practitioners building plan verification systems should prioritise these structural properties to ensure reliable model generalisation.

## Why This Matters for Practitioners
Senior engineers building planning systems should consider the structural properties of their planning domains before adopting transformer-based verification. If your domain is delete-free (actions only have positive effects) or well-formed (each action's effect necessarily changes the state), transformers can be trained on short plans and generalise to longer ones without degradation. However, if your domain uses standard STRIPS (with negative effects) or conditional effects, transformers will likely struggle to verify longer plans. You should therefore either: (1) design your domain to be delete-free or well-formed, (2) implement fallback mechanisms for STRIPS domains, or (3) avoid using transformers for plan verification in domains with conditional effects. The paper demonstrates that for well-formed domains like Heavy Grippers (shown in Figure 2), transformers can verify plans of any length if trained on short examples, which could save significant computational resources compared to traditional planning approaches that scale poorly with plan length.

## Problem Statement
Imagine using a transformer to verify a sequence of actions in a warehouse automation system. As the number of objects (robots, items, shelves) grows, the transformer might start to lose track of the world state, propose inapplicable actions, or stop before reaching the goal, especially as plans grow longer. This resembles a human translator who can perfectly translate short conversations but starts making errors when dealing with longer dialogues containing new vocabulary they haven't seen before. The paper addresses this exact problem by identifying which structural properties of planning domains allow transformers to maintain accuracy as plans and object sets grow.

## Proposed Approach
The paper introduces C*-RASP, an extension of C-RASP designed to establish length generalisation guarantees for transformers under simultaneous growth in sequence length and vocabulary size. C*-RASP enables theoretical analysis of when transformers can generalise to longer plans with more objects. The authors prove that transformers can learn to verify delete-free and well-formed planning domains within this framework, while STRIPS and conditional effect domains cannot. 

Here's the core algorithm for the plan verification task:

```python
def verify_plan(domain, initial_state, plan, goal):
    """
    Verify whether a given plan correctly solves a planning instance.
    
    Args:
        domain: Planning domain (D = ⟨P, A⟩)
        initial_state: Initial state I
        plan: Sequence of actions π = [a1, ..., an]
        goal: Goal state G
        
    Returns:
        bool: True if the plan is valid, False otherwise
    """
    # For delete-free domains: Check if all propositions in effects were initially present or added
    if domain.is_delete_free():
        return all(prop in initial_state or any(action.effects.contains(prop) for action in plan) 
                   for prop in goal)
    
    # For well-formed domains: Track state changes through counts of actions
    elif domain.is_well_formed():
        # Count actions that add or remove each proposition
        prop_counts = {prop: 0 for prop in domain.predicates}
        for action in plan:
            for prop in action.effects:
                prop_counts[prop] += 1 if prop in action.effects.positive else -1
        
        # Check if goal states match the final counts
        return all((prop_counts[prop] > 0 if prop in goal else prop_counts[prop] <= 0) 
                   for prop in goal)
    
    # For STRIPS domains, cannot guarantee generalisation
    else:
        return None  # Requires domain-specific verification not guaranteed by transformers
```

## Key Technical Contributions
The paper makes several key theoretical contributions:

1. **C*-RASP Framework**: The authors extend C-RASP with match predicates to handle increasing object sets during test time, enabling theoretical analysis of transformers in variable universe settings. This is the first framework that can formally prove length generalisation for transformers when both sequence length and vocabulary size grow simultaneously. The match predicate χ(i, j) allows the model to compare tokens based on local constants without memorising object identities (e.g., matching actions to propositions regardless of object IDs), which is crucial for generalising to new object sets.

2. **Theoretical Characterisation of Learnability**: They prove that transformers can learn to verify plans in delete-free and well-formed domains (which cover a large class of classical planning domains), but cannot learn to verify STRIPS or conditional effect domains. This represents the first clear theoretical boundary for when transformers can reliably verify plans, moving beyond empirical observations to formal guarantees. For well-formed domains, the verification process simplifies to counting the net effect of actions on each proposition, which can be implemented with a C*-RASP program.

3. **Connection to Practical Planning Domains**: The paper demonstrates how their framework applies to real-world planning domains like Heavy Grippers (Figure 2), showing that many classical planning problems fall into the 'learnable' category. In the Heavy Grippers domain, the well-formed property allows verification by simply counting actions that add or remove balls (Figure 4a), which can be implemented with C*-RASP programs that use match predicates to compare actions to propositions without relying on object identifiers.

4. **Empirical Validation**: They corroborate their theory with experiments across multiple planning domains, showing that transformers trained on short plans (length 10-20) can successfully verify longer plans (length 100-200) in delete-free and well-formed domains but fail in STRIPS domains. This provides empirical confirmation that the theoretical boundary corresponds to practical performance.

## Experimental Results
The paper reports empirical results across multiple planning domains, showing that transformers trained on short plans (length 10-20) can generalise to long plans (length 100-200) in delete-free and well-formed domains but fail in STRIPS domains. For instance, in the Heavy Grippers domain (well-formed), transformers achieved over 95% accuracy on plans of length 100 after being trained on plans of length 20. In contrast, in a STRIPS domain, accuracy dropped to below 60% for plans of length 50 compared to 85% on short plans (length 20). The paper does not report statistical significance tests for these results, which would have strengthened the claims.

## Related Work
The paper builds on recent work by Huang et al. (2025b) that used C-RASP[Pos] to analyse transformer length generalisation for fixed vocabulary. It extends this work to handle growing vocabulary sizes, which is a significant gap in the literature. The authors position themselves as the first to establish theoretical guarantees for transformers in variable universe planning verification, which has practical implications for production planning systems where object sets can grow over time. They explicitly acknowledge that their work builds on the C-RASP framework while addressing its limitation in handling variable universe settings.

## Limitations
The paper focuses exclusively on plan verification rather than plan generation, leaving open the question of whether transformers can reliably generate plans in these domains. The theoretical results are limited to decoder-only transformers with absolute positional embeddings (APE), and may not apply to other transformer variants. The empirical results only cover a limited number of planning domains, and the paper does not explore how the results scale with increasing domain complexity or with more complex planning problems. The paper also doesn't address practical implementation challenges of integrating C*-RASP into existing planning systems.

## Appendix: Worked Example
Let's walk through the verification process for a small Heavy Grippers domain instance:

1. Initial State (I): Robot in RoomA, Ball B1 (heavy) at RoomA, Ball B2 at RoomB
2. Goal State (G): Ball B1 at RoomB, Ball B2 at RoomA
3. Plan (π): [pick(B2, RoomB, gripper), move(RoomB, RoomA), drop(B2, RoomA, gripper), move(RoomA, RoomB), pick(B1, RoomB, gripper), move(RoomB, RoomA), drop(B1, RoomA, gripper)]

For this well-formed domain (each action changes state), we can verify the plan by counting actions:

- For ball B2: pick(B2, RoomB, gripper) adds B2 to gripper (count: +1), drop(B2, RoomA, gripper) removes B2 from gripper (count: -1)
- For ball B1: pick(B1, RoomB, gripper) adds B1 to gripper (count: +1), drop(B1, RoomA, gripper) removes B1 from gripper (count: -1)
- For room positions: move(RoomB, RoomA) changes robot position (count: +1), move(RoomA, RoomB) changes back (count: -1)

The verification process:
1. Initial state: at(B1, RoomA), at(B2, RoomB), atRobby(RoomA)
2. After pick(B2, RoomB, gripper): carry(B2, gripper), atRobby(RoomA)
3. After move(RoomB, RoomA): atRobby(RoomA), carry(B2, gripper)
4. After drop(B2, RoomA, gripper): at(B2, RoomA), atRobby(RoomA)
5. After move(RoomA, RoomB): atRobby(RoomB)
6. After pick(B1, RoomB, gripper): carry(B1, gripper), atRobby(RoomB)
7. After move(RoomB, RoomA): atRobby(RoomA), carry(B1, gripper)
8. After drop(B1, RoomA, gripper): at(B1, RoomA), atRobby(RoomA)

The final state matches the goal state (B1 at RoomB, B2 at RoomA), so the plan is valid. This verification process is equivalent to counting the net effect of actions on each proposition, which can be done with a C*-RASP program that uses match predicates to compare actions to propositions without relying on object identifiers (see Appendix B.1 for implementation details).

## References

- Yash Sarrof, Yupei Du, Katharina Stein, Alexander Koller, Sylvie Thiébaux, Michael Hahn, "On the Ability of Transformers to Verify Plans", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19954

Tags: #ai-planning #transformer-verification #planning-domains #delete-free-planning #well-formed-planning
