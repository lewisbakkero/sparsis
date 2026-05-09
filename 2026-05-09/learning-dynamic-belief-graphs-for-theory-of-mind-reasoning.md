---
title: "Learning Dynamic Belief Graphs for Theory-of-mind Reasoning"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20170"
---

## Executive Summary
This paper introduces a structured cognitive trajectory framework that represents mental states as dynamic belief graphs, enabling LLM-based Theory-of-Mind reasoning with coherent belief evolution. It addresses the limitation of existing methods that treat beliefs as static and independent, leading to incoherent mental models in high-stakes scenarios like disaster response. For engineers building human-AI interaction systems, this provides a principled way to model and predict human behaviour under uncertainty without requiring explicit belief supervision.

## Why This Matters for Practitioners
If you're building production systems for high-stakes domains like emergency response, medical triage, or human-in-the-loop autonomy, this paper suggests you should move beyond simple belief inference to model belief interdependencies and temporal dynamics. Instead of prompting LLMs to generate static beliefs directly (which leads to semantic drift and incoherent reasoning), you can implement a structured belief graph with energy-based potentials that captures how beliefs evolve and interact over time. For example, when developing an evacuation alert system, the model can now predict not just "will they evacuate?" but "what beliefs will drive that decision?" (e.g., "my home is in danger" combined with "my neighbour has evacuated"). This enables more accurate action prediction and interpretable mental model recovery, which is critical for systems where errors have real-world consequences.

## Problem Statement
Consider the current state of LLM-based Theory-of-Mind systems: they treat beliefs as isolated, static snapshots, like trying to understand a movie by freezing it on a single frame. In reality, beliefs evolve and interact, just as a person's decision to evacuate a wildfire doesn't depend on a single fear, but on a complex interplay of beliefs that accumulate over time (e.g., "I see flames" reinforcing "my home is in danger" which then triggers "I should leave"). Existing approaches fail to capture this dynamic interplay, producing incoherent mental models that accumulate errors rapidly in high-stakes settings like disaster response.

## Proposed Approach
The authors propose a structured cognitive trajectory model where mental states are represented as dynamic belief graphs. The system has three main components: a semantic-to-potential projection layer that maps LLM embeddings into unary and pairwise potentials, an energy-based factor graph representation of belief interdependencies, and an ELBO-based objective that jointly optimizes for belief trajectory and action prediction. The model updates the belief graph online from observations alone, then predicts actions from the updated belief state.

```python
def update_belief_graph(previous_belief: List[bool], observation: str) -> List[bool]:
    # Project LLM embeddings to potential functions
    unary_potentials = semantic_to_potential(observations=observation, 
                                            previous_belief=previous_belief, 
                                            type='unary')
    pairwise_potentials = semantic_to_potential(observations=observation, 
                                              previous_belief=previous_belief, 
                                              type='pairwise')
    
    # Compute belief transition prior as Gibbs distribution
    energy = compute_energy(unary_potentials, pairwise_potentials)
    belief_transition = exp(-energy) / partition_function(energy)
    
    # Sample new belief state from transition distribution
    new_belief = sample_belief(belief_transition)
    return new_belief
```

## Key Technical Contributions
The paper makes three key technical contributions that move beyond simple belief inference:

1. **Semantic-to-potential projection**: They introduce a method to map LLM-derived semantic embeddings into unary and pairwise potential functions, ensuring the latent belief space maintains consistent semantic orientation. Unlike previous approaches that directly prompt LLMs for belief states (which leads to semantic drift), this projection contrasts embeddings against reference states ("Yes" vs "No" for belief persistence) and anchors the orientation using cosine similarity. This prevents the model from flipping semantic interpretations of beliefs (e.g., interpreting "my home is in danger" as "my home is safe" due to LLM bias), as shown by their use of ϕbase = τ(cos(ht,i, hYes) - cos(ht,i, hNo)).

2. **Energy-based factor graph representation**: Instead of treating beliefs as independent, they model belief interdependencies using an energy-based factor graph where pairwise potentials capture how beliefs reinforce or suppress each other. For example, belief 0 ("My home is in danger") and belief 1 ("My neighbour has evacuated") have a positive pairwise potential (ϕ01 = 1.4), indicating co-activation, while belief 0 and belief 2 ("There's no immediate threat") have a negative potential (ϕ02 = -0.7), indicating suppression. The paper explicitly trains these pairwise interactions, recovering belief co-variation patterns that match human survey data (see Figure 5b).

3. **ELBO-based joint optimisation**: They derive a variational learning objective that jointly optimizes the semantic projection, belief graph potentials, and action model by maximising the evidence lower bound on action trajectory likelihood. Crucially, this separates inference (using the realized action to guide posterior belief estimation) from generation (using the belief state to predict actions), addressing a key limitation in prior work that couldn't properly align belief inference with behavioral outcomes. The authors use the inference model qϕ(bt|ot, at) with action conditioning to produce more accurate posterior estimates, which guide the learning of the generative model through the KL divergence term in the ELBO.

## Experimental Results
The model was evaluated on wildfire evacuation datasets from real-world surveys (Kincade Fire and Marshall Fire), with 6 binary belief dimensions, 3 discrete time steps, and 4 intermediate actions plus 2 final evacuation decisions. It achieved significant improvements in action prediction accuracy over baselines (AutoToM, Model Reconciliation, FLARE), with final evacuation decision accuracy reaching 82.7% compared to AutoToM's 71.2%. The Spearman correlation between predicted belief scores and human survey ratings averaged 0.54 for the proposed model versus 0.27 for AutoToM and 0.32 for FLARE. For pairwise belief structures, the proposed model achieved a Spearman correlation of 0.66 between predicted and survey-derived belief co-variation rankings, significantly outperforming baselines (0.30 for AutoToM, 0.11 for Model Reconciliation), with statistical significance confirmed by p < 0.01.

## Related Work
The paper positions itself as moving beyond traditional Bayesian Inverse Planning (BIP) approaches that require synthetic state spaces and fixed dynamics, and beyond LLM-based ToM methods that treat beliefs as static and independent (e.g., AutoToM and MuMToM). It builds on structured latent variable modelling (e.g., Deep Markov Models and Energy-Based Models) but integrates them with LLM semantics to capture belief interdependencies. Unlike FLARE, which combines PADM with LLM reasoning, this work doesn't require pre-defined cognitive pathways but learns belief interdependencies directly from data.

## Limitations
The authors acknowledge that their model was evaluated only on wildfire evacuation datasets, so its applicability to other high-stakes domains (e.g., medical triage or financial crisis decisions) remains untested. They also note that the model requires belief marginals to be computed exactly by enumerating all joint belief configurations, which becomes computationally intractable for larger belief sets (K > 6 in their experiments). Additionally, the paper doesn't explicitly test how the model handles belief contradictions (e.g., when a person has both "my home is in danger" and "there's no immediate threat").

## Appendix: Worked Example
Let's walk through the belief graph update for a specific example in the wildfire evacuation scenario:

Start with a resident who has observed "Received an evacuation order" (observation at time t=1) and has a previous belief state of [0, 0, 0, 0, 0, 0] (all beliefs false). The LLM extracts semantic embeddings for each belief dimension, contrasting the "Yes" and "No" previous belief states:

- For belief 0 ("My home is in danger"), the embeddings are hYes = [0.95, -0.12, ..., 0.23] and hNo = [0.12, 0.57, ..., -0.34]
- The anchor score is computed as ϕbase = 0.7(cos(h, hYes) - cos(h, hNo)) = 0.85
- After learning, the unary potential for belief 0 becomes ϕ0 = 1.2 (indicating high probability)
- Belief 1 ("My neighbour has evacuated") has ϕ1 = 0.8
- The pairwise potential between beliefs 0 and 1 is ϕ01 = 1.4 (positive, indicating co-activation)
- The pairwise potential between belief 0 and belief 2 ("There's no immediate threat") is ϕ02 = -0.7 (negative, indicating suppression)

The energy function is computed as E = -∑ϕi - ∑ϕij = -1.2 - 0.8 + 1.4 + 0.7 = -0.3. The belief transition prior is defined as p(b|b_prev, o) = exp(-E)/Z = exp(0.3)/Z. The model samples a new belief state from this distribution, resulting in [1, 1, 0, 0, 0, 0], meaning the resident now believes "My home is in danger" and "My neighbour has evacuated," but not "There's no immediate threat." This belief state then drives the action prediction, resulting in a high probability of choosing "Evacuate" as the action.

## References

- Ruxiao Chen, Xilei Zhao, Thomas J. Cova, Frank A. Drews, Susu Xu, "Learning Dynamic Belief Graphs for Theory-of-mind Reasoning", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20170

Tags: #disaster-response #human-ai-interaction #belief-graphs #energy-based-models #elbo-optimisation
