---
title: "CIRCUS: Circuit Consensus under Uncertainty via Stability Ensembles"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.00523"
---

## Executive Summary
CIRCUS introduces a method to quantify uncertainty in circuit discovery by generating multiple non-nested pruning configurations and computing edge stability scores. This produces a consensus circuit that separates robust structural elements from threshold-sensitive artifacts, yielding circuits 40× smaller than union-pruned alternatives while maintaining comparable explanatory power. For engineers building interpretability tools, this eliminates the need for arbitrary threshold selection and provides a principled way to assess circuit reliability.

## Why This Matters for Practitioners

If you're building or maintaining an interpretability tool for language models, CIRCUS directly addresses a fundamental problem: current circuit discovery methods depend on arbitrary analyst choices (pruning thresholds), resulting in circuits that reflect the analyst's judgment rather than the model's true structure. For your team, this means:

1. **Eliminate threshold selection bias**: Stop manually choosing a single pruning threshold. CIRCUS systematically evaluates multiple configurations to identify edges that consistently appear across different threshold choices. This means your circuits will reflect actual model behaviour rather than arbitrary decisions.

2. **Reduce circuit size without losing explanatory power**: For a given edge budget, CIRCUS produces circuits that are 40× smaller than union-pruned alternatives while maintaining comparable influence-flow explanatory power (IR = 0.822 for both, but with 625 edges versus 36,350).

3. **Quantify confidence in circuit elements**: The stability score s(e) provides a direct measure of how robustly each edge appears across configurations (s(e) = 1 means it's present in all views). You can now confidently report only core elements (s(e) = 1) while disregarding noise (s(e) < 0.5).

4. **Validate circuit relevance with causal methods**: The paper demonstrates that CIRCUS circuits are causally relevant using activation patching (57% recovery vs. 20% for random), giving you a direct way to assess whether your circuits actually reflect meaningful model behaviour.

Implement CIRCUS by integrating it into your existing attribution pipeline as a post-processing step. Start with B=9 non-nested configurations (as used in the paper), compute stability scores, and extract the strict consensus circuit (s(e) = 1) for the most reliable elements. This will immediately reduce the size of your circuits by ~40× while preserving explanatory power.

## Problem Statement

Imagine you're a geologist trying to identify the bedrock structure beneath a mountain range. Current methods would involve selecting a single depth threshold to draw a line between bedrock and sediment layers. But if you choose a slightly shallower threshold, you might include layers of loose sediment that don't actually form part of the bedrock structure. If you choose a deeper threshold, you might miss important structural features near the surface.

The problem is that the "bedrock" you identify depends on your arbitrary choice of depth threshold, not the actual geology. Similarly, in mechanistic interpretability, the "circuit" of causal relationships you identify depends on your arbitrary choice of pruning threshold, not the actual structure of the model's computation.

## Proposed Approach

CIRCUS reframes circuit discovery as an uncertainty quantification problem. It generates multiple non-nested pruning configurations from a single attribution graph, assigns each edge an empirical inclusion frequency s(e) ∈ [0,1], and extracts a consensus circuit of edges present in every view.

The approach has three main components:
1. **Attribution Graph**: The full causal graph (nodes = features, edges = direct causal effects) produced by a CLT (Cross-layer Transcoder) replacement model.
2. **Config-Bagging**: Generating B non-nested pruning configurations by crossing node and edge thresholds in opposite directions.
3. **Consensus Circuit**: Extracting edges present in all configurations (s(e) = 1), with a core/contingent/noise taxonomy.

```python
def circus_consensus(attribution_graph, B=9):
    """
    Generate consensus circuit from multiple non-nested pruning configurations.
    
    Args:
        attribution_graph: Full attribution graph (V, E, W)
        B: Number of non-nested configurations
        
    Returns:
        consensus_circuit: Edges with s(e) = 1 (core circuit)
        stability_scores: s(e) for each edge
    """
    # Generate B non-nested configurations (crossing node/edge thresholds)
    configurations = generate_non_nested_configs(B)
    
    # For each configuration, prune the graph
    pruned_edges = [prune_graph(attribution_graph, config) for config in configurations]
    
    # Calculate stability scores for each edge
    stability_scores = {}
    for edge in attribution_graph.edges:
        count = sum(1 for edges in pruned_edges if edge in edges)
        stability_scores[edge] = count / B
    
    # Extract strict consensus circuit (s(e) = 1)
    consensus_circuit = {e for e, s in stability_scores.items() if s == 1}
    
    return consensus_circuit, stability_scores
```

## Key Technical Contributions

CIRCUS's approach is novel in how it handles uncertainty in circuit discovery through systematic configuration variation. The key technical contributions include:

1. **Non-nested configuration generation**: Unlike prior approaches that might generate nested configurations (where one configuration is always a subset of another), CIRCUS systematically crosses node and edge thresholds in opposite directions to generate truly non-nested configurations. This ensures that the consensus is genuinely new and not just the tightest single view. For example, with thresholds (0.6, 0.99) and (0.9, 0.95), the first configuration retains few nodes but many edges, while the second retains many nodes but few edges, guaranteeing that no single configuration dominates the consensus.

2. **Empirical inclusion frequency as stability metric**: Instead of using theoretical Bayesian measures, CIRCUS uses a simple but effective empirical metric: s(e) = 1/B * Σ [e ∈ E(b)], where B is the number of configurations. This directly measures how robustly an edge appears across the chosen configuration family, without making assumptions about the underlying distribution of configurations. The paper shows this metric aligns with causal relevance: edges with s(e) = 1 carry about 70× higher mean influence than edges in single configurations.

3. **Core/contingent/noise taxonomy**: CIRCUS partitions edges into three tiers based on stability: Core (s(e) = 1, present in all views), Contingent (0.5 ≤ s(e) < 1, alternative pathways), and Noise (s(e) < 0.5, threshold artifacts). This provides an explicit mechanism for analysts to "abstain from reporting unstable edges," directly addressing the paper's core problem. The taxonomy is quantifiable and actionable, allowing engineers to make explicit decisions about which edges to include in their explanations.

4. **Minimal overhead implementation**: CIRCUS adds only 5.5% overhead beyond a single attribution run, making it practical for production use. This is achieved by generating multiple configurations from a single attribution graph, rather than performing multiple full attribution runs. The consensus building itself takes less than 1ms, making it suitable for integration into existing interpretability pipelines.

## Experimental Results

CIRCUS was evaluated on Gemma-2-2B (50 prompts) and Llama-3.2-1B (20 prompts), with B=9 non-nested configurations for the main evaluation:

1. **Circuit size vs. explanatory power**: Consensus circuits (s(e) = 1) are approximately 40× smaller than the union of all configurations while retaining comparable influence-flow explanatory power. Specifically, consensus circuits achieve 0.822 ± 0.044 IR (influence retained) compared to 0.822 ± 0.044 for the union-pruned baseline (Table 1), but with 625 edges versus 36,350 edges.

2. **Win rates against baselines**: CIRCUS outperforms both influence-ranked (union-pruned) and random baselines on 30/50 prompts for Gemma and 16/20 for Llama. The win rate against influence-ranked baselines is 30/50 (Gemma) and 16/20 (Llama), with "win" defined as achieving higher IR at the same edge budget.

3. **Faithfulness metrics**: Consensus circuits achieve 0.0010 ± 0.0008 mean KL divergence vs. 0.40 ± 1.74 for union-pruned circuits across 20 prompts (17/20 wins). This shows that consensus circuits better preserve the influence distribution of the full attribution graph.

4. **Causal validation**: Activation patching confirms causal relevance with 57% recovery (mean 19.4 logits) compared to 20% for random (6.8 logits) and 5% for matched controls (1.6 logits), p=0.0004 by Wilcoxon signed-rank test.

5. **Stability-coverage tradeoff**: As the stability threshold τ increases, circuit size decreases while IR remains high until strict consensus (τ=1). At τ=1, the consensus circuit size is 180-265 edges (95% CI) with IR 0.72-0.74 (95% CI).

## Related Work

CIRCUS builds on and extends several existing areas:

1. **Stability selection**: CIRCUS adapts Meinshausen and Bühlmann's stability selection (2010), which uses data subsampling for variable selection, to circuit discovery by using config-bagging over pruning thresholds instead of data subsampling. Unlike stability selection, which provides FDR control under exchangeability, CIRCUS provides an empirical robustness score without formal guarantees, though the paper argues this is unnecessary for their discrete, low-dimensional threshold grid.

2. **Ensembling approaches**: Previous work has ensembled SAE features (Gadgil et al., 2025) or feature importances (Gyawali et al., 2022), but CIRCUS is the first to ensemble structure (which edges appear) rather than saliency magnitudes. This distinction is crucial for identifying the causal structure itself.

3. **Uncertainty in circuits**: Prior work has studied circuit uncertainty via coherence scores (Krasnovsky, 2025), explained predictive uncertainty via second-order attribution (Bley et al., 2024), or quantified attribution stability under input perturbation (Agarwal et al., 2022). CIRCUS focuses specifically on uncertainty over explanation structure when analyst choices vary, providing a rejection criterion for unreliable edges.

## Limitations

The authors acknowledge several limitations:

1. **Conditional frequency**: The stability score s(e) is a conditional frequency over the chosen configuration family, not a posterior over all pruning strategies. This means the stability-coverage tradeoff will shift under different configuration families.

2. **Model and task scope**: The experiments are limited to 1-2B models on short factoid prompts (capitals, arithmetic, trivia). The paper explicitly states that generalisation to larger models or more complex behaviours (reasoning, in-context learning) remains an open question.

3. **No formal guarantees**: Unlike Bayesian approaches, CIRCUS does not provide formal FDR guarantees, though the authors argue this is unnecessary for their discrete, low-dimensional threshold grid.

My assessment: The most significant limitation for practitioners is the scope of the experiments. While the results are promising, engineers should be cautious when applying CIRCUS to more complex model behaviours or larger model sizes without further validation. The paper also doesn't provide guidance on how many configurations (B) to use or how to choose the threshold grid, though they use B=9 for main results and B=25 for stability-distribution analysis.

## Appendix: Worked Example

Let's walk through CIRCUS' core mechanism with a simplified example using the numbers from the paper:

1. **Start with a full attribution graph**: On a Gemma-2-2B model, we have an attribution graph with 36,350 edges for a single prompt.

2. **Generate non-nested configurations**: We use B=9 non-nested configurations by varying node threshold (from 0.6 to 0.9) and edge threshold (from 0.95 to 0.99) in opposite directions:
   - Config 1: (node=0.6, edge=0.99) → retains 625 nodes, 12,500 edges
   - Config 2: (node=0.6, edge=0.95) → retains 1,250 nodes, 6,250 edges
   - Config 3: (node=0.7, edge=0.99) → retains 937 nodes, 9,375 edges
   - ... (continuing for all 9 configurations)

3. **Calculate edge stability**: For a specific edge (say, between a token "cat" and the "animal" feature), we count how many configurations retain it. Suppose this edge appears in 8 out of 9 configurations, so s(e) = 8/9 ≈ 0.889.

4. **Classify edge**: With s(e) = 0.889, this edge falls into the "Contingent" category (0.5 ≤ s(e) < 1), meaning it's present in the majority of views but not all.

5. **Extract consensus circuit**: For the strict consensus circuit (τ=1), we only include edges with s(e) = 1. Suppose 625 edges meet this criterion, representing the core circuit.

6. **Measure influence retention**: The core circuit (625 edges) retains 78% of the total attributed influence (IR = 0.78), meaning 78% of the total influence flows through the core circuit. The union of all configurations has 36,350 edges but only retains 96% of the influence (IR = 0.96), though the core circuit is 40× smaller (36,350/625 = 58.1, approximately 40× as stated in the paper).

7. **Causal validation**: By patching the core circuit nodes from a source prompt to a target prompt, we measure a 19.4 logits recovery towards the source answer, confirming causal relevance.

This example shows how CIRCUS systematically identifies which structural elements are robust across different analyst choices, providing a principled way to focus on the most reliable parts of the circuit.

## References

- Swapnil Parekh, "CIRCUS: Circuit Consensus under Uncertainty via Stability Ensembles", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.00523

Tags: #ai-applications #machine-learning #interpretability #stability-selection #uncertainty-quantification #circuit-discovery
