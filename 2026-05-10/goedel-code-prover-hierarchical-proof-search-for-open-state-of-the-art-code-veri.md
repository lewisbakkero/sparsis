---
title: "Goedel-Code-Prover: Hierarchical Proof Search for Open State-of-the-Art Code Verification"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19329"
---

## Executive Summary
Goedel-Code-Prover introduces a hierarchical proof search framework for automated code verification in Lean 4, decomposing complex verification goals into structurally simpler subgoals before attempting tactic-level proving. The system achieves a 62.0% prove success rate on 427 verification tasks, a 2.6× improvement over the strongest baseline while surpassing neural provers up to 84× larger. This approach addresses the fundamental gap between LLM-generated code and provably correct implementations in safety-critical systems.

## Why This Matters for Practitioners
If your team is using LLMs for code generation in safety-critical applications like automotive control systems or financial transaction processing, this paper suggests you should prioritise formal verification over traditional testing. The authors demonstrate that LLM-generated code often contains subtle logical errors that unit tests miss, and their hierarchical approach enables automated verification that scales to real-world complexity. Practitioners should integrate Lean 4 verification into their CI/CD pipelines by building a library of reusable lemmas for common patterns (like list operations or graph traversals) that can be invoked during verification. Additionally, when selecting LLMs for code generation, favour those with training on formal verification data rather than general code corpora, as this directly addresses the "ungrounded decomposition" challenge identified in the paper.

## Problem Statement
Most current LLM-based code generation systems operate as "black boxes" that produce code that appears to work but cannot be verified for correctness, much like a chef who serves food without ever tasting it, risking food poisoning. Just as a chef would need a rigorous recipe checking system to verify all ingredients are safe and combined correctly, code generation requires formal verification to ensure safety-critical properties like boundary condition handling and specification adherence. The current gap is that LLMs lack the pretraining on formal verification reasoning to decompose verification goals effectively, leading to "ungrounded decomposition" where proposed subgoals are frequently semantically invalid or no simpler than the original goal.

## Proposed Approach
Goedel-Code-Prover operates in a planning-and-proving loop where an LLM interacts with Lean 4's kernel to decompose verification goals into simpler subgoals before attempting to prove them. The system uses a single unified policy that: (1) decomposes goals hierarchically using a principled score, (2) proves decomposed lemmas through iterative tactic generation with feedback. The core innovation is that the same decomposition score serves as both the training reward and inference-time ranking criterion, ensuring alignment between optimisation and deployment.

```python
def hierarchical_proof_search(goal):
    # Stage 1: Recursive Lemma Decomposition
    while not goal.provable:
        # Generate candidate decompositions using LLM
        candidates = llm.generate_decompositions(goal)
        # Evaluate each candidate using decomposition score
        ranked_candidates = sort_by_score(candidates, goal)
        # Select best candidate for decomposition
        best_decomposition = ranked_candidates[0]
        # Verify proposed subgoals with proof reconstruction and quickcheck
        if all_subgoals_valid(best_decomposition):
            goal = best_decomposition
        else:
            # Discard invalid decomposition and try again
            continue
    
    # Stage 2: Lemma Completion
    while not all_subgoals_proven:
        # Prove each subgoal iteratively
        for subgoal in goal.subgoals:
            tactic_sequence = llm.generate_tactics(subgoal)
            if lean_kernel.verify(tactic_sequence):
                mark_subgoal_proven(subgoal)
            else:
                # Retry with feedback
                continue
```

## Key Technical Contributions
The paper introduces a novel framework that addresses the three key challenges in code verification with specific mechanism-level innovations:

1. **Constructive justification through proof reconstruction**: The framework requires that a proposed decomposition must include a proof reconstruction that Lean can verify as logically entailing the parent theorem. This prevents the search from pursuing invalid decomposition branches, unlike previous approaches that relied solely on the LLM's ability to generate meaningful subgoals. The system generates a proof reconstruction term as part of the decomposition output, ensuring each proposed subgoal is not just a syntactic list but a constructively justified reduction of the original goal.

2. **Structural reduction using operator footprint**: To quantify how much simpler subgoals are relative to the original, the system computes the operator footprint - the total number of logical and domain-specific operator occurrences in the abstract syntax tree of a verification goal. This metric captures reasoning cost because each operator corresponds to a specific class of proof obligations. The structural reduction ratio r = max(1 - (d(Li)/d(G)), 0) is used to rank decompositions, with the system favouring those that reduce the footprint of the hardest subgoal.

3. **Hybrid reinforcement learning pipeline**: The training pipeline combines supervised initialization with a novel hybrid reinforcement learning approach. The continuous decomposition reward drives planning exploration during training, while supervised replay of high-quality proof trajectories stabilises proof generation. This resolves the reward mismatch problem where decomposition benefits from a dense continuous score, but completion yields only sparse binary signals (proof accepted or rejected).

4. **Unified policy for decomposition and completion**: Unlike previous frameworks that trained separate models for decomposition and completion, Goedel-Code-Prover uses a single 8B-parameter policy for both tasks. This eliminates the need for separate training objectives and model coordination, while the system's ability to use the same decomposition score for both training and inference ensures strict alignment between the two phases.

## Experimental Results
The authors evaluated Goedel-Code-Prover on three Lean 4 code verification benchmarks: Verina (68.8% success rate), Clever (54.0%), and AlgoVeri (62.3%), comprising 427 tasks in total. The 8B-parameter model achieved a 62.0% prove success rate overall, a 2.6× improvement over the strongest baseline (which used a 55B-parameter model) and surpassed neural provers up to 84× larger. Verified proofs averaged 8, 17 decomposed lemmas and over 130 lines of proof code, with the most complex exceeding 680 lines, demonstrating the system's ability to sustain deep structured reasoning over non-trivial verification tasks. The authors observed consistent inference-time scaling: success rates improved monotonically with search iterations and sampling budget, indicating unsaturated scaling potential.

## Related Work
Goedel-Code-Prover distinguishes itself from prior work in two key ways. Unlike previous LLM-based theorem provers that focused on mathematical domains (e.g., Hubert et al., 2025; Ren et al., 2025), this system addresses the specific challenges of code verification. Earlier work on mathematical proof generation (e.g., Ren et al., 2025) relies on a rich corpus of natural language descriptions of mathematical proofs that enables effective decomposition, but such a corpus does not exist for program correctness. The authors also build on the "decompose-and-prove" paradigm used in mathematical theorem provers (Li et al., 2024), but adapt it for code verification by addressing the "ungrounded decomposition" issue with their principled decomposition score. Unlike previous approaches that trained separate models for decomposition and completion, this work uses a unified policy that leverages the same decomposition score for both training and inference.

## Limitations
The authors acknowledge that their current system requires substantial computational resources, with training a single 8B-parameter model requiring significant GPU time. They note that the system's performance is currently limited to Lean 4 verifications and may not directly transfer to other theorem provers without substantial adaptation. The authors also point out that their framework currently requires some initial manual lemma generation for new program domains, though they propose future work to automate this. From a practical perspective, the current system is not yet ready for production use in safety-critical systems due to the resource requirements and the need for further evaluation on larger, more complex codebases. The authors don't specify whether their decomposition score would work for program types beyond those covered by their benchmarks, which might limit generalisability to novel programming paradigms.

## Appendix: Worked Example
Let's walk through the hierarchical proof search process for the FindSingleNumber verification task from Figure 1, using the specific metrics from the paper.

Start with the top-level goal "FindSingleNumber_spec" which has an operator footprint d(G) = 18 (as detailed in Figure 1). The decomposition score S = v({Li}; G) · r({Li}; G) guides the search.

1. **Initial decomposition proposal**: The LLM proposes two subgoals:
   - L1: "filterlist on the unique element returns a singleton list" (d(L1) = 7)
   - L2: "every other element appears in filterlist exactly twice" (d(L2) = 8)

2. **Constructive justification check**:
   - Proof reconstruction: The LLM produces a proof that (L1 ∧ L2) ⇒ G, which Lean accepts as valid (1proof = 1).
   - Quickcheck: The system randomly samples inputs and checks L1 and L2. Both pass (1qc(L1) = 1, 1qc(L2) = 1).
   - Validity gate v = 1 × 1 × 1 = 1.

3. **Structural reduction calculation**:
   - Average footprint of subgoals = (7 + 8)/2 = 7.5
   - Structural reduction ratio r = 1 - (7.5/18) = 0.583
   - Decomposition score S = 1 × 0.583 = 0.583

4. **Inference-time ranking**: This decomposition score is used to rank candidate decompositions. The system would select this decomposition as it scores higher than alternatives that either fail the validity check or yield a lower structural reduction.

5. **Lemma completion**: Each subgoal is then proved individually:
   - L1: Proved using Lean tactics like "grind" and "native_decide" for list reasoning (13 lines of proof code)
   - L2: Proved using "inductive" tactics for reasoning about list elements (21 lines of proof code)

The final proof integrates these subgoals, requiring a total of 85 lines of proof code (13 + 21 + 51), demonstrating how the hierarchical structure reduces the complexity of the verification task from a single 18-operator goal to two simpler subgoals.

## References

- Zenan Li, Ziran Yang, Deyuan, Haoyu Zhao, Andrew Zhao, Shange Tang, Kaiyu Yang, Aarti Gupta, Zhendong Su, Chi Jin, "Goedel-Code-Prover: Hierarchical Proof Search for Open State-of-the-Art Code Verification", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19329

Tags: #software-verification #hierarchical-proof-search #decomposition-score #hybrid-reinforcement-learning
