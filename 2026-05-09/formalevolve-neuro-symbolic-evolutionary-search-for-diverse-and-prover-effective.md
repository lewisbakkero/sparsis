---
title: "FormalEvolve: Neuro-Symbolic Evolutionary Search for Diverse and Prover-Effective Autoformalization"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19828"
---

## Executive Summary
FormalEvolve introduces a neuro-symbolic evolutionary search framework that constructs diverse, semantically consistent Lean 4 formalizations within a strict generator-call budget. It addresses the critical gap between semantic consistency and prover effectiveness in autoformalization systems. For production theorem-proving pipelines, this means engineers can reduce wasted compute on intractable formalizations while increasing the likelihood of successful proof completion.

## Why This Matters for Practitioners
If you're building theorem-proving infrastructure for mathematical libraries or formal verification systems, FormalEvolve offers a practical solution to a pervasive production problem: semantic consistency doesn't guarantee proof success. The paper demonstrates that without diversity-aware search, semantic successes concentrate on a small subset of easy problems (Gini coefficient 0.813 on CombiBench), wasting 37% of the generator budget on problems that would never be provable. Instead of optimising for single-output semantic accuracy, you should implement repertoire-based search within your existing LLM-based autoformalization pipeline. Start by instrumenting your generator to track compilation success and semantic consistency, then add bounded repair mechanisms and archive-based diversity operators. For teams using Lean 4, this means modifying your existing autoformalization code to maintain a compilation-feasible archive, implement usage-penalized selection, and apply EvolAST-style structural diversity operators without consuming additional generator calls.

## Problem Statement
Imagine building a recipe book for a professional chef where every recipe is technically accurate (uses correct ingredients in correct proportions) but some require 30-minute prep while others need 8 hours to cook. The chef can't simply select the most accurate recipe; they need a diverse repertoire of recipes that fit within their time budget. Similarly, in autoformalization, a semantically correct Lean 4 statement can require 100 proof attempts to verify (like a complex recipe) or succeed on the first try (like a simple recipe). Current systems collapse this diversity into single-output decisions, wasting compute on problems that would never prove successfully within budget constraints.

## Proposed Approach
FormalEvolve constructs a diverse repertoire of semantically consistent Lean 4 formalizations within a strict generator-call budget. It maintains a compilation-feasible archive where candidates are stored with semantic scores, uses usage-penalized selection to avoid mode collapse, and applies LLM-driven mutation with bounded repair. For structural diversity without additional generator calls, it uses a conservative AST rewrite operator (EvolAST). The framework operates under two constraints: all generator calls (seeds, patches, repairs) are debited against a fixed budget (T=100 calls per problem), and compilation serves as a hard feasibility gate.

```python
def formal_evolve(informal_statement, T=100):
    # Initialize archive with compilation-feasible seed candidates
    archive = initialize_archive(informal_statement, T)
    
    # While budget remains
    while T > 0:
        # Sample from current archive with usage penalties
        parent = sample_parent(archive)
        
        # Propose candidate via LLM with bounded repair
        candidate = propose_candidate(parent, informal_statement)
        T -= 1
        
        # Apply bounded compilation repair if needed
        if not compile(candidate):
            candidate = repair_candidate(candidate, T)
            T = max(0, T - 1)
        
        # Evaluate candidate
        if semantic_judge(candidate):
            archive.add(candidate)
        
        # Apply EvolAST fallback for structural diversity
        if not candidate:
            candidate = evolast_rewrite(parent)
            archive.add(candidate)
    
    # Return semantically consistent repertoire
    return deduplicate(archive, semantic_consistent=True)
```

## Key Technical Contributions
FormalEvolve introduces three key innovations that differentiate it from prior approaches:

1. **Compilation-Gated Repertoire Search**: Unlike previous systems that optimise for single-output semantic accuracy, FormalEvolve explicitly constructs a diverse repertoire of compilation-feasible candidates within a fixed generator budget. It maintains a compilation-feasible archive (C=1) where candidates are stored regardless of semantic consistency (J=0), then filters to a semantically consistent repertoire (C=1 ∧ J=1) for downstream proving. This separates statement generation (coverage and concentration) from proving performance, addressing the core problem that semantic consistency does not imply prover effectiveness.

2. **Usage-Penalized Selection**: To prevent mode collapse where semantic successes concentrate on easy problems (Gini coefficient 0.813 for Kimina Compile+Semantic Repair), FormalEvolve implements a robust scoring mechanism that factors in both compilation success (C) and semantic consistency (J), then applies a parent-usage penalty term. The score s(c) = C(c) · (1 + J(c)) ranges from 0 to 2, with the selection mechanism using median absolute deviation (MAD) and logistic transformation to weight candidates while penalizing frequently-selected templates.

3. **EvolAST Structural Diversity Fallback**: For structural diversity without additional generator calls, FormalEvolve applies an EvolAST-style rewrite operator that conservatively rewrites only within binder types and goal types. This produces symmetry/structure variants (e.g., swapping argument order in a theorem statement) that preserve semantic meaning while changing proof-search behaviour. The paper shows this improves cross-problem uniformity without increasing generator calls, reducing the Gini coefficient from 0.813 to 0.759 on CombiBench.

## Experimental Results
On CombiBench (N=100), FormalEvolve achieves SH@100=58.0% (vs 46.0% for Kimina Compile+Semantic Repair), increasing semantic coverage by 12 percentage points within the same generator-call budget of T=100. It also reduces cross-problem concentration: the Gini coefficient decreases from 0.813 (Kimina) to 0.759 (FormalEvolve), and the top-10% share of semantic successes drops from 60.9% to 53.1%. On ProofNet (N=186), SH@100 increases from 78.0% to 84.9% with Gini decreasing from 0.555 to 0.443.

Downstream proving performance improves significantly on CombiBench: FormalEvolve achieves 13/100 theorem-complete@64 versus 8/100 for Kimina Compile+Semantic Repair. Crucially, this improvement comes from broader semantic coverage (58/100 problems with semantically consistent repertoire vs 46/100) rather than higher success rates on individual problems (22.8% for FormalEvolve vs 17.4% for Kimina). On ProofNet, performance remains comparable (45/186 vs 46/186 theorem-complete@64), consistent with ProofNet being closer to the training distribution of existing autoformalizers.

## Related Work
FormalEvolve builds upon autoformalization systems like Kimina (Wu et al., 2022), which focuses on semantic fidelity but doesn't address prover effectiveness. It differs from systems like ReForm (Chen et al., 2025) that use iterative refinement and tool feedback, as FormalEvolve optimizes repertoire-level reliability within a strict generator-call budget rather than refining trajectories. Unlike test-time search approaches (e.g., Novikov et al., 2025), FormalEvolve uses compilation as a hard feasibility gate and applies a conservative, zero-call diversity operator (EvolAST) for structural diversity. It advances beyond EvolProver (Tian et al., 2025), which focuses on augmenting training data for provers, by enabling online diversity during test-time search without additional generator calls.

## Limitations
The paper acknowledges that performance on ProofNet (which is closer to the training distribution of competition-style autoformalizers) shows less improvement, suggesting that FormalEvolve's benefits may be most pronounced on out-of-distribution problems like CombiBench. The evaluation uses a fixed prover (Goedel-Prover-V2-32B), so results may vary with different provers. The authors don't test how FormalEvolve scales beyond T=100 calls, though Figure 3 shows coverage continues to improve with increased budget. The paper doesn't address how to select which problems should receive more generator calls within a fixed overall budget.

## Appendix: Worked Example
Let's walk through the generation of a single statement on CombiBench using FormalEvolve's approach:

1. **Initial seed bank**: The seed model (Mseed) generates 16 initial candidates for a problem about graph theory. Only 11 compile successfully (C=1), which are stored in the archive.

2. **Archive sampling**: The system samples a parent from the archive using usage-penalized selection. The parent has a score s(c) = 1 (C=1, J=0), and its parent-usage count n=2.

3. **Proposal and repair**: The patch model proposes a new candidate (diff patching) based on the parent. Compilation fails (C=0), so the system applies bounded compilation repair (max 3 attempts). After 2 repair attempts (consuming 2 generator calls), the candidate compiles (C=1).

4. **Semantic evaluation**: The semantic judge evaluates the candidate (J=0), so it's stored in the archive with C=1 but J=0. The system applies bounded semantic repair (1 attempt), and the candidate now achieves J=1.

5. **Diversity check**: The archive contains 12 compilation-feasible candidates. The system applies EvolAST (conservative AST rewrite) to the latest candidate, creating a structural variant that preserves meaning but changes proof-search behaviour (e.g., swapping theorem statement arguments).

6. **Repertoire formation**: After 100 generator calls, the system has 58 semantically consistent candidates. The archive contains candidates covering 58 problems, with a Gini coefficient of 0.759 (vs 0.813 for Kimina), indicating more uniform coverage.

See Appendix for a step-by-step worked example with concrete numbers.

## References

- Haijian Lu, Wei Wang, Jing Liu, "FormalEvolve: Neuro-Symbolic Evolutionary Search for Diverse and Prover-Effective Autoformalization", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19828

Tags: #formal-methods #theorem-proving #neural-symbolic-systems #evolutionary-algorithms #prover-effectiveness
