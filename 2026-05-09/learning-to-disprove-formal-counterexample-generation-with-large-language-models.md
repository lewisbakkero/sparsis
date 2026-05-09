---
title: "Learning to Disprove: Formal Counterexample Generation with Large Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19514"
---

## Executive Summary
This paper introduces a method for training large language models (LLMs) to generate formal counterexamples using a symbolic mutation strategy and multi-reward expert iteration. Unlike most AI approaches to mathematics that focus on proof construction, their framework addresses the critical but neglected task of counterexample generation, which enables LLMs to self-verify their reasoning. This work demonstrates a 47-74% relative improvement in counterexample generation over existing models, with applications in mathematical research and LLM reasoning reliability.

## Why This Matters for Practitioners
If you're building AI systems that require robust mathematical reasoning for verification or debugging, this paper suggests a new approach to enhance your model's reliability. Specifically, for systems that need to validate mathematical claims (e.g., financial models, safety-critical algorithms, or formal verification systems), the ability to generate and verify counterexamples can prevent costly errors by identifying flaws before deployment. Practitioners should consider integrating this counterexample generation capability into their validation pipelines, as it directly addresses the "black box" problem in mathematical reasoning by providing concrete evidence of why a mathematical claim fails.

## Problem Statement
Mathematical reasoning in AI is like building a house with only one type of tool: you can construct perfect walls (proofs), but you can't find the rotten ones (counterexamples) that would break the foundation. Currently, AI systems excel at proving mathematical statements but lack the ability to systematically find counterexamples that disprove false claims. This creates a dangerous blind spot: a system might confidently "prove" something that's actually false because it never verified with specific counterexamples. The paper compares this gap to a car safety system that can prove your seatbelt is secure but can't test what happens when it's not buckled.

## Proposed Approach
The authors propose a two-stage framework for training LLMs to generate formal counterexamples: (1) counterexample problem synthesis through symbolic mutation of theorems, and (2) multi-reward guided training. The framework uses Lean 4 theorem prover for automatic verification of counterexamples. The core innovation is their mutation strategy that discards hypotheses from provable theorems to create counterexample problems, combined with a multi-reward system that provides continuous training signal even when LLMs fail on complex problems.

```python
def integrated_workflow(seed_theorems):
    # Phase 1: Dataset Mutation
    mutated_problems = []
    for theorem in seed_theorems:
        mutated_version = mutate_theorem(theorem, drop_hypothesis=True)
        dropped_hypothesis = get_dropped_hypothesis(theorem)
        mutated_problems.append((mutated_version, dropped_hypothesis))
    
    # Phase 2: Expert Iteration
    for iteration in range(max_iterations):
        # Generate counterexamples and formal proofs
        for (mutated, dropped) in mutated_problems:
            counterexample = generate_counterexample(mutated)
            proof_mutated = generate_proof(mutated, counterexample)
            proof_dropped = generate_proof(dropped, counterexample)
            
            # Compute rewards
            reward_mutated = verify_proof(proof_mutated)
            reward_dropped = verify_proof(proof_dropped)
            total_reward = alpha * reward_mutated + (1 - alpha) * reward_dropped
            
            # Update models using weighted dataset
            update_counterexample_model(mutated, counterexample, total_reward)
            update_proof_model(mutated, proof_mutated, reward_mutated)
            update_proof_model(dropped, proof_dropped, reward_dropped)
```

## Key Technical Contributions
The framework introduces novel mechanisms that overcome two critical challenges in counterexample generation: data scarcity and sparse reward signals.

1. **Symbolic mutation strategy for data synthesis:** Instead of relying on limited datasets like CounterMath (1,216 examples), they extract theorems from formal libraries (Mathlib, Leanworkbook) and automatically mutate them by discarding essential hypotheses. This method produced 575,039 counterexample problems by applying mutation to 321,929 seed theorems from diverse mathematical domains. The key innovation is using Lean 4 to verify the necessity of discarded hypotheses before creating counterexample problems, ensuring each problem is meaningful and verifiable.

2. **Multi-reward training for continuous learning:** They introduce a double-reward mechanism where the model receives two separate rewards: one for correctly proving the mutated theorem (H₂(x) → C(x)) and another for proving the dropped hypothesis (¬H₁(x)). The authors set α=0.8 to balance these rewards, ensuring the model receives continuous training signals even for difficult problems. This prevents the "sparse reward" problem that plagues standard reinforcement learning approaches.

3. **Integrated formal verification pipeline:** The entire process is built around Lean 4 theorem prover, allowing automatic verification of both counterexamples and formal proofs. This creates a closed-loop system where every counterexample can be verified and used to improve the model, eliminating the need for human verification. The framework handles both "guess" (counterexample proposal) and "check" (proof verification) phases simultaneously, a significant departure from previous informal-to-formal reasoning approaches.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The authors conducted experiments on three benchmarks:
1. **FOR-COUNTER:** 1,058 formal counterexample problems generated from CounterMath (constructed from four classic counterexample textbooks)
2. **VERI-REASON:** 3,000 problems assessing errors in reasoning steps (from DSP+ applied to FormalMath)
3. **VERI-FORMALIZE:** 3,000 problems assessing errors in autoformalized results (from FormalMath)

The model outperformed all baselines, with the strongest improvements on FOR-COUNTER:
- **Pass@1:** Solved 222 problems, 95 more than the strongest baseline (DeepSeek-Prover-v2 with 127)
- **Relative improvement:** 74% (74% higher than strongest baseline) for Pass@1 on FOR-COUNTER
- **Multi-reward training:** Achieved 49% Pass@1 vs. 43% for single-reward baseline (p<0.05)
- **Dataset size:** 575K counterexample training examples synthesized from 321K seed theorems

The performance gap was consistent across all metrics (Pass@1, Pass@4, Pass@9) across all three benchmarks, with the largest improvements (56% for Pass@1 on VERI-FORMALIZE) on the more challenging verification tasks.

## Related Work
This work builds on recent advances in formal reasoning LLMs (Seed prover, Kimina prover, Goedel prover) but addresses an underexplored aspect of mathematical reasoning. Unlike previous approaches that focus solely on proof construction, this paper directly tackles counterexample generation, which requires a different reasoning paradigm (guess-and-check rather than logical deduction). The authors position their work as filling a critical gap in mathematical AI research, noting that "counterexamples play a vital role in theory development, conjecture refinement, and educational enhancement."

## Limitations
The authors acknowledge two main limitations: the framework currently focuses on mathematical counterexamples and hasn't been tested on other domains like programming or natural language understanding. They also note that the mutation strategy relies on Lean 4 for verification, which limits applicability to domains without formal verification tools. The experiments were conducted on a single epoch with 56 iterations, suggesting more training could potentially yield further improvements.

## Appendix: Worked Example
Let's walk through a specific counterexample generation example based on the paper's methodology. We'll use a simple mathematical theorem from Mathlib as our starting point:

**Original Theorem (provable):** 
```
theorem original_version (x : ℕ) : x > 0 → x + 1 > 1 := by
  simp [add_one]
```

**Step 1: Hypothesis removal (symbolic mutation):**
The authors' Lean 4 tactic `mutate` discards the hypothesis `x > 0`, creating a new theorem:
```
theorem mutated_version (x : ℕ) : x + 1 > 1 := by
  sorry
```
This mutated theorem is invalid for x=0 (0+1=1 is not >1), establishing a counterexample.

**Step 2: Counterexample generation:**
The LLM is prompted to find a counterexample for `mutated_version`. It proposes x=0 as a counterexample.

**Step 3: Formal proof generation:**
The model generates two formal proofs:
1. For the mutated theorem with counterexample x=0:
   ```
   theorem mutated_version : ∃ x : ℕ, x + 1 > 1 := by
     use 0
     simp
   ```
   This proof is invalid (proves a false statement), so it receives 0 reward.

2. For the dropped hypothesis (¬(x > 0)):
   ```
   theorem dropped_hypothesis : ∃ x : ℕ, ¬ (x > 0) := by
     use 0
     simp
   ```
   This proof is valid (0 is not >0), so it receives 1 reward.

**Step 4: Reward calculation:**
With α=0.8, the total reward = (0.8 × 0) + (0.2 × 1) = 0.2, which provides a small but meaningful signal for training.

**Step 5: Model update:**
The counterexample model is updated with a weight of 0.2 based on this example, while the proof model receives weights of 0 for the mutated proof and 0.2 for the dropped hypothesis proof.

This example shows how the multi-reward system provides continuous training signals even when the primary counterexample is incorrect, addressing the sparse reward problem.

## References

- Zenan Li, Zhaoyu Li, Kaiyu Yang, Xiaoxing Ma, Zhendong Su, "Learning to Disprove: Formal Counterexample Generation with Large Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19514

Tags: #mathematical-reasoning #formal-verification #counterexample-generation #llm-training #lean-theorem-prover
