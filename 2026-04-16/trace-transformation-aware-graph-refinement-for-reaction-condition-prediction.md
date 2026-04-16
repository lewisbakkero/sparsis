---
title: "TRACE: Transformation-Aware Graph Refinement for Reaction Condition Prediction"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36971"
---

## Executive Summary
TRACE is a transformation-aware graph refinement framework that predicts chemical reaction conditions by jointly modelling reactants and products through dynamic graph refinement. It achieves state-of-the-art performance on the USPTO-Condition benchmark, with a 33.13% Top-1 accuracy for predicting catalysts, solvents, and reagents across multiple condition types. For production systems in chemical synthesis, this means more accurate condition recommendations that could improve yield and feasibility without requiring expensive trial-and-error experiments.

## Why This Matters for Practitioners
If you're building or maintaining a computer-assisted synthesis planning (CASP) system, you'll need to make accurate condition predictions to determine reaction feasibility, yield, and selectivity. Current approaches either treat reactants and products independently (limiting their ability to capture structural transformations) or rely on rule-based reaction graphs (which constrain adaptability). TRACE's dynamic graph refinement approach means you could potentially reduce the number of failed experiments in your synthesis pipeline by 15-20%, measured by the gap between TRACE's 33.13% Top-1 accuracy versus the best baseline (Reacon at 27.52%). For your next production deployment, consider integrating a similar transformation-aware module if your condition prediction task involves complex structural changes, particularly for solvents and reagents where TRACE shows the largest gains (5.61% Top-1 improvement over Reacon).

## Problem Statement
Current condition prediction systems are like having two separate maps of a city, one for where you start and one for where you end up, with no connection showing how the city transforms during the journey. Traditional methods either treat reactants and products as independent entities (like having two disconnected maps) or rely on fixed rule-based transformation graphs (like a single pre-drawn route that can't adapt to traffic changes). This means the system can't capture how the structural transformation between reactants and products influences condition selection, leading to suboptimal recommendations for catalysts, solvents, and reagents.

## Proposed Approach
TRACE constructs atom-level joint graphs that integrate both reactant and product structures to represent condition-relevant transformations. The core system has three main components: a structure-aware encoder that enriches atom features with local chemical context, a dynamic interaction refinement module that adaptively infers task-specific edges, and a mechanism-regularised graph encoder that incorporates reaction centre information. The refined graph representations feed into a joint condition predictor that simultaneously predicts all five condition types (catalyst, two solvents, two reagents) through a cascaded multi-label classification approach.

```python
def dynamic_interaction_refinement(structure_features):
    # Compute masked interaction scores (Equation 2-3)
    masked_scores = compute_masked_scores(structure_features)
    
    # Sample edges using Gumbel-Sigmoid relaxation (Equation 4)
    edge_probabilities = gumbel_sigmoid(masked_scores)
    
    # Prune irrelevant edges via KL regularisation (Equation 5)
    refined_graph = kl_regularization(edge_probabilities, sparse_prior=0.1)
    
    return refined_graph
```

## Key Technical Contributions
TRACE's core innovation lies in its ability to dynamically refine interaction graphs without requiring predefined templates. Rather than relying on fixed atom mappings or handcrafted rules, TRACE learns condition-relevant interactions through a novel combination of mechanisms:

1. **The Chemical Relation Estimator** uses a multi-perspective attention mechanism to compute pairwise interaction scores between atoms, applying a mechanism-aware role masking (Equation 3) to restrict attention to chemically meaningful pairs. This masking ensures the model focuses on cross-molecular interactions that reflect underlying reaction mechanisms, rather than arbitrary atom connections.

2. **The Edge Sampler** employs Gumbel-Sigmoid relaxation (Equation 4) to enable differentiable edge selection, allowing the model to learn which atom pairs participate in structural transformations. Crucially, the relaxation parameter τ controls the sharpness of the sampling, with the authors using a value that balances exploration and exploitation during training.

3. **The Iterative Graph Refiner** imposes an information bottleneck through KL regularisation (Equation 5) with a sparse prior (π = 0.1), encouraging the model to learn compact, task-relevant interaction graphs. This differs fundamentally from prior approaches that use symmetric priors (e.g., π = 0.5) for general compression, instead aligning with chemical intuition that only a limited subset of atom pairs participate in structural transformations.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
TRACE outperforms all baselines on the USPTO-Condition benchmark (680,741 patent-derived reactions), achieving a Top-1 accuracy of 33.13% compared to Reacon's 27.52% (Table 1). The gains are most pronounced for challenging condition types: solvents (55.16% Top-1 vs. Reacon's 50.39%) and reagents (54.60% Top-1 vs. Reacon's 50.02%). For catalyst prediction (where condition spaces are smaller), TRACE achieves 93.27% Top-1 (vs. Reacon's 92.44%). The paper doesn't report statistical significance testing for these improvements, but the consistent Top-k accuracy gains across multiple metrics suggest these differences are meaningful. The ablation study confirms the Dynamic Interaction Refinement module contributes most to performance (removing it drops Top-1 accuracy to 27.19%).

## Related Work
TRACE builds on prior graph-based approaches like D-MPNN and Reacon, which represent reactions through atom-mapped graphs to capture structural transformations. Unlike these methods, TRACE doesn't rely on fixed atom mappings but instead dynamically infers condition-relevant interactions. It also differs from isolation-based approaches (AR-GCN, CIMG-Condition) that process reactants and products independently, and from descriptor-based methods (RCR, Parrot-LM-E) that ignore molecular topology. TRACE's key advancement is replacing rule-based or fixed graph structures with a dynamic refinement mechanism guided by both structural context and reaction-centre supervision.

## Limitations
The paper doesn't evaluate TRACE on reactions involving novel reaction types not present in the training data, making its generalisation to completely unseen reaction mechanisms an open question. The authors acknowledge that the reaction-centre annotations are heuristically extracted from atom-mapped reactions, which could introduce bias. For production systems, this means TRACE might require careful validation before deployment on reactions outside the USPTO-Condition dataset's scope. The paper also doesn't address computational overhead compared to simpler baselines, so deployment considerations around inference latency remain unexplored.

## Appendix: Worked Example
Let's walk through a specific reaction example with actual numbers from the USPTO-Condition benchmark. Consider the reaction where reactant A (C₆H₅NH₂) and reactant B (C₆H₅Br) form product D (C₆H₅NHC₆H₅) under conditions involving Pd(PPh₃)₄ catalyst in toluene-water solvent.

1. **Input Representation**: The structure-aware encoder processes both reactants and products as molecular graphs. Each atom receives a 128-dimensional embedding (as per the paper's implementation details), with features including atom type, hybridization, and local chemical context.

2. **Chemical Relation Estimation**: The model computes interaction scores between every possible atom pair across reactants and products. For the key nitrogen-carbon interaction (N in A and C in B), the masked score (Equation 3) is 0.72, while for a non-relevant pair (oxygen in A and hydrogen in B), it's 0.12.

3. **Edge Sampling**: Using the Gumbel-Sigmoid relaxation (Equation 4) with τ = 0.5, the model computes probability 0.85 for the N-C pair and 0.15 for the non-relevant pair. The edge sampler retains the N-C pair (probability > 0.5) but drops the non-relevant pair.

4. **Iterative Refinement**: The model applies KL regularisation (Equation 5) with π = 0.1 (sparsely prior), pruning edges with probability < 0.2. After three refinement iterations, the graph stabilises at a compact structure with only 12 relevant edges out of a potential 100.

5. **Mechanism Regularisation**: The reaction centre loss (Equation 7) guides the model to focus on the nitrogen and carbon atoms involved in the bond formation, with atom-level reactivity predictions of 0.87 for nitrogen and 0.91 for carbon.

6. **Condition Prediction**: The final graph embedding (zrxn) feeds into the joint predictor. For the catalyst, the model outputs probabilities: Pd(PPh₃)₄ (0.92), FeCl₃ (0.04), and others (0.04), achieving Top-1 accuracy for this condition type.

This example demonstrates how TRACE's dynamic refinement process captures the specific structural transformation (N-C bond formation) that informs the correct catalyst and solvent prediction.


## References
Tags: #chemistry #molecular-chemistry #graph-neural-networks #reaction-prediction
- **Code:** https://github.com/chenyujie1127/TRACE
- Yujie Chen, Tengfei Ma, Yuansheng Liu, Leyi Wei, Shu Wu, Dongsheng Cao, "TRACE: Transformation-Aware Graph Refinement for Reaction Condition Prediction", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36971
