---
title: "Failure Localization in Multi-Agent Code Generation via Knowledge-Guided and Transferable Reasoning"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36993"
---

## Executive Summary
FLKR (Failure Localization via Knowledge-guided Reasoning) is a self-supervised framework that accurately pinpoints responsible agents in multi-agent code generation systems. It tackles the challenge of solution-path multiplicity, where different valid coding strategies produce structurally distinct but equally correct implementations, by aligning agent behaviour with domain knowledge rather than surface-level code comparison. For practitioners building production multi-agent systems, this means debugging failures without requiring expensive human-annotated failure data.

## Why This Matters for Practitioners
If you're maintaining a production multi-agent code generation system (like those using MetaGPT), failures often stem from inter-agent dependencies rather than individual agent flaws, making debugging arduous. FLKR shows you can automate failure localization without human-annotation, improving debugging efficiency by 50% or more. For example, when a multi-agent system fails to generate correct code for a Codeforces problem, FLKR can pinpoint whether the failure came from the architect (designing wrong algorithm) or the engineer (implementing correctly but with flawed logic), as seen in the human evaluation where FLKR's suggestions resolved 85% of failures (vs. 60% for prompting baselines). Practically, this means you can focus debugging efforts on the specific agents causing issues rather than manually reviewing all agent logs, reducing debugging time significantly.

## Problem Statement
Imagine a multi-agent team building a complex software feature: the product manager defines requirements, the architect designs the structure, the engineer implements code, and the QA engineer tests it. When a bug appears in the final product, the team often argues over who's responsible, was it the architect for a flawed design, or the engineer for a coding mistake? In multi-agent code generation systems, this problem is amplified because the same functionality can be implemented with vastly different code paths (e.g., using a greedy algorithm vs. dynamic programming), making it impossible to rely on surface-level code comparison. The challenge is that failures often stem from conceptual misunderstandings along a specific path, yet appear structurally different from reference solutions.

## Proposed Approach
FLKR localizes responsible agents in multi-agent code generation by encoding agent behaviour, aligning it with domain knowledge, and scoring deviations from both knowledge and context. It consists of four key components:

1. A semantic encoder (CodeBERT) that transforms agent interactions into vectors
2. A knowledge projection module that aligns behaviour with algorithmic strategies
3. A consistency scorer measuring deviation from knowledge and context
4. An unsupervised responsibility estimator combining these signals

The framework operates entirely without human supervision, learning from unlabeled logs via contrastive learning and self-distillation.

```python
def flkr(problem, agent_log):
    # Encode each agent's decision using CodeBERT
    agent_vectors = [semantic_encoder(decision) for decision in agent_log]
    
    # Retrieve knowledge strategies for the problem
    knowledge_strategies = knowledge_base.retrieve(problem)
    
    # Project into shared latent space
    agent_projections = [knowledge_projection(vec) for vec in agent_vectors]
    strategy_projections = [knowledge_projection(strategy) for strategy in knowledge_strategies]
    
    # Compute alignment scores
    alignment_scores = [
        softmax(dot(agent_proj, strategy_proj)) 
        for agent_proj in agent_projections
    ]
    
    # Compute consistency scores
    knowledge_divergence = [
        1 - max(alignment_score) 
        for alignment_score in alignment_scores
    ]
    context_divergence = [
        compute_contextual_divergence(vec, neighbours) 
        for vec, neighbours in zip(agent_vectors, agent_log)
    ]
    consistency_score = [0.7 * k + 0.3 * c for k, c in zip(knowledge_divergence, context_divergence)]
    
    # Estimate responsibility scores
    responsibility_scores = [
        mlp([vec, score]) 
        for vec, score in zip(agent_vectors, consistency_score)
    ]
    normalized_scores = softmax(responsibility_scores)
    
    return normalized_scores
```

## Key Technical Contributions
FLKR's core innovation is its solution-path invariant localization, which doesn't rely on reference code comparison but on aligning behaviour with domain knowledge. Here's how it works at the implementation level:

1. **Domain knowledge alignment**: Instead of comparing agent code to a reference (which fails when different valid strategies are used), FLKR aligns each agent's behaviour with a set of canonical algorithmic strategies retrieved from a knowledge base. For a problem requiring a sorting solution, it aligns with strategies like "merge sort" or "quick sort" rather than comparing to a specific implementation. The knowledge base uses Codeforces tags and hand-labelled algorithmic templates to retrieve these strategies, enabling solution-path invariance.

2. **Consistency scoring mechanism**: FLKR computes two divergence scores: knowledge divergence (how far an agent's behaviour is from any canonical strategy) and contextual divergence (how much an agent's behaviour deviates from surrounding agents' behaviour). These are combined using learnable weights (λ1 = 0.7, λ2 = 0.3) to form a consistency signal, which is critical for handling ambiguity and enabling robust attribution without reference alignment.

3. **Self-supervised training approach**: FLKR is trained without human annotations using a contrastive learning objective (InfoNCE loss) that maximizes alignment between agent behaviour and correct strategies, and a self-distillation loss that uses consistency scores as pseudo-labels. This allows learning from unlabeled logs, which is crucial because expert annotations for failure localization are expensive and rare.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
FLKR was evaluated on COFL (Code Oriented Failure Localization), the first benchmark with expert-annotated failure localizations, containing 2520 unlabeled Codeforces cases for training and 142 expert-annotated cases for evaluation.

On COFL, FLKR outperforms prompting-based baselines by 14 points in Fault Localisation Accuracy (0.89 vs. 0.75 for Binary Search Prompting) and 45 points in Top-1 Localisation Accuracy (78% vs. 33% for Binary Search Prompting).

On divergent-solution cases (84 cases where agents pursued valid but non-canonical strategies), FLKR outperforms baselines by 14 points in Fault Localisation Accuracy (0.80 vs. 0.66) and 21 points in Top-1 Localisation Accuracy (62% vs. 41%).

For out-of-domain generalisation, FLKR was tested on 70 cases from the SoftwareDev dataset, achieving Fault Localisation Accuracy of 0.81 (vs. 0.65 for Binary Search Prompting) and Top-1 Localisation Accuracy of 76% (vs. 30% for Binary Search Prompting).

Human evaluation showed FLKR's suggestions had higher Correctness (4.3), Specificity (4.1), and Actionability (4.2) on a 5-point scale compared to baselines.

## Related Work
The paper positions itself against existing failure localization approaches that rely on prompting techniques (Zhang et al. 2025) for identifying responsible agents. While these methods work for natural language dialogue, they fail for code generation due to the divergent structures of code data and the high accuracy requirements of code. The paper also references root-cause analysis in traditional systems (using structural causal models) but argues these are inadequate for high-dimensional generative workflows with complex agent interactions.

## Limitations
The paper acknowledges several limitations:
- The knowledge base used for strategy alignment is limited to Codeforces tags and hand-labelled algorithmic templates, which may not cover all programming domains.
- The COFL benchmark has only 142 expert-annotated cases, which may not represent the full diversity of real-world failures.
- The method focuses on agent-level failure localization but doesn't address how to resolve the failure once localized.

My assessment: The paper's focus on code generation from a multi-agent perspective is valuable, but the limited scale of the expert-annotated dataset (142 cases) means the results may not generalise to all real-world failure scenarios. Additionally, the paper doesn't provide a clear method for updating the knowledge base as new problem types emerge.

## Appendix: Worked Example
Let's walk through a specific case from the paper's evaluation for a "find the maximum subarray sum" problem:

1. **Input**: The problem statement describes finding the maximum contiguous subarray in an integer array (e.g., [-2, 1, -3, 4, -1, 2, 1, -5, 4]) with expected output 6.

2. **Agent Log**:
   - Architect: "We'll use Kadane's algorithm for this problem" (aligned with strategy)
   - Engineer: "I implemented Kadane's algorithm with a modification" (flawed implementation)
   - QA: "The code fails for input [-2, 1, -3, 4, -1, 2, 1, -5, 4] with output 5 instead of 6" (identifies failure)
   - Product Manager: No relevant contribution

3. **Semantic Encoding**:
   - Each agent's contribution is encoded using CodeBERT (768-dimensional vectors).
   - Architect's decision → Vector z1 (768-d)
   - Engineer's decision → Vector z2 (768-d)
   - QA's decision → Vector z3 (768-d)

4. **Knowledge Strategy Alignment**:
   - Knowledge base contains two strategies:
     - K1: "Use Kadane's algorithm" (aligned with architect)
     - K2: "Use dynamic programming approach"
   - Alignment scores for engineer (z2):
     - α2,1 = 0.4 (aligns with K1 but with deviations)
     - α2,2 = 0.6 (aligns better with K2)
   - Knowledge divergence = 1 - max(α2,1, α2,2) = 0.4

5. **Contextual Divergence**:
   - Engineer's vector (z2) compared to architect (z1) and QA (z3)
   - Contextual divergence = ||z2 - (z1 + z3)/2|| = 0.15 (768-d space)

6. **Consistency Score**:
   - d2 = (0.7 * knowledge divergence) + (0.3 * context divergence) = 0.315

7. **Responsibility Estimation**:
   - MLP([z2; d2]) outputs raw score s2 = 2.1
   - Normalised responsibility score = exp(2.1) / sum(exp(all scores))
   - Engineer: 78%, Architect: 20%, QA: 1%, Product Manager: 1%

This means 78% of the failure is attributed to the engineer (who implemented the flawed version), 20% to the architect (who proposed the flawed strategy), and 1% each to the other agents. This matches the actual reason for the failure (a flawed implementation of Kadane's algorithm).

## References

- **Code:** https://github.com/gmy2013/FLRK
- Mingyang Geng1∗, Shanzhi Gu, Zhipeng Liu, Chuanfu Xu, Zhaoyang Qu, Haotian Wang, "Failure Localization in Multi-Agent Code Generation via Knowledge-Guided and Transferable Reasoning", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36993

Tags: #multi-agent #code-generation #failure-localisation #knowledge-guided #self-supervised
