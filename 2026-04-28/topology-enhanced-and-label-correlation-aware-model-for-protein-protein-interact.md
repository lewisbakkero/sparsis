---
title: "Topology-Enhanced and Label Correlation-Aware Model for Protein-Protein Interaction Prediction"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36980"
---

## Executive Summary
TELC-PPI addresses two fundamental limitations in current graph neural network (GNN) approaches to protein-protein interaction (PPI) prediction: it explicitly models functional relationships between proteins that existing networks overlook, and it captures dependencies between different interaction types. For engineers building biomedical prediction systems, this means significantly improved accuracy on multi-label PPI tasks without requiring additional training data.

## Why This Matters for Practitioners
If your team is implementing PPI prediction systems for drug discovery pipelines, this paper reveals that simply using standard GNNs on protein interaction networks will miss critical functional relationships between proteins. The topology enhancement mechanism they developed, identifying functionally similar proteins through their interaction patterns, can be directly integrated into your graph construction phase to improve representation learning without adding computational overhead during inference. You should also consider implementing label correlation modelling for your multi-label classification tasks; the authors show a consistent 0.5-1% F1-score improvement by incorporating co-occurrence statistics, which is significant in the high-accuracy biomedical domain. The provided GitHub implementation offers a production-ready template for these modifications.

## Problem Statement
Imagine building a recommendation system for a music streaming service where the graph only connects users who've listened to the same song, ignoring that users who've listened to similar songs might share preferences. This is exactly the problem with current PPI networks: they connect proteins that interact directly (like users who've listened to the same song), but miss the crucial connections between proteins that share similar interaction patterns with a common partner (like users who've listened to similar songs). This violates the homophily assumption central to graph neural networks, making it impossible for standard GNNs to learn meaningful representations for functionally similar proteins.

## Proposed Approach
TELC-PPI constructs an enhanced graph by adding "hidden" edges between functionally similar protein pairs based on their interaction patterns with common neighbours. It then models dependencies between different interaction types using co-occurrence statistics. The system has five main components: input module (proteins, edges, labels), topology enhancement (adding H2 edges), edge representation learning (updating node features), label modelling (learning label embeddings), and prediction module (computing interaction types).

```python
def telc_ppi(graph, labels):
    # Step 1: Construct H2 candidate edges
    adj_matrix = graph.adjacency_matrix()
    adj_sq = adj_matrix @ adj_matrix
    acand = max(adj_sq - adj_matrix, 0) * (1 - np.eye(n))
    acand = min_max_normalize(acand)
    
    # Step 2: Compute node label distributions
    node_labels = {}
    for node in graph.nodes:
        labels = sum(graph.edges[node].labels, start=0)
        node_labels[node] = normalize(labels)
    
    # Step 3: Build enhanced graph with H2 edges
    enhanced_graph = graph.copy()
    for i, j in np.nonzero(acand):
        sim = alpha * acand[i,j] + (1-alpha) * dot_product(node_labels[i], node_labels[j])
        if random() < sim / sum(sim for all pairs):
            enhanced_graph.add_edge(i, j, type="H2")
    
    # Step 4: Learn edge and label embeddings
    edge_embeddings = graph_conv(enhanced_graph)
    label_embeddings = learn_label_embeddings(labels, co_occurrence_matrix)
    
    # Step 5: Predict interaction types
    return dot_product(edge_embeddings, label_embeddings)
```

## Key Technical Contributions
The core innovation lies in how the model addresses the two fundamental limitations of existing approaches.

1. **H2 Principle for topology enhancement**: The model identifies functionally similar proteins by evaluating both the number of shared neighbours (topological strength) and similarity in interaction type distributions (semantic similarity). It doesn't just add edges between proteins with common neighbours, it computes a weighted score `sim(i,j) = α * Acand(i,j) + (1-α) * (pi^T pj)/(||pi||_2 ||pj||_2)`, where `α` balances topological and semantic information. This prevents noise from indiscriminately adding edges between proteins that happen to share neighbours but have no functional relationship.

2. **Label co-occurrence-aware embedding**: The model learns label embeddings by combining two constraints: a label co-occurrence regularisation loss `Lcoor = ΣΣ Mcoor(i,j) ||E_i - E_j||^2` and a maximum likelihood loss `Lml = Σ CE(ψ(E_c), y_c)`. The co-occurrence matrix `Mcoor` is constructed from empirical conditional probabilities `Mcoor(i,j) = 1/2 [p(li|lj) + p(lj|li)]`, which ensures that functionally related interaction types (like binding and catalysis) are represented close together in embedding space.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
TELC-PPI achieves state-of-the-art performance on three datasets:
- STRING (15,355 proteins, 593,397 PPIs): 96.94% micro-F1
- SHS148k (5,189 proteins, 44,488 PPIs): 92.63% micro-F1
- SHS27k (1,690 proteins, 7,624 PPIs): 89.37% micro-F1

Compared to the previous state-of-the-art (MAPE-PPI), TELC-PPI improves micro-F1 by 0.83% on STRING (Random split), 0.95% on SHS148k (DFS split), and 0.57% on SHS27k (BFS split). The paper reports these results as "significant" but doesn't specify statistical tests beyond five runs with different random seeds. The ablation study shows removing the topology enhancement module leads to the largest performance drop (0.18-0.44% F1 decrease), confirming its importance.

## Related Work
TELC-PPI builds on established GNN-based PPI prediction methods (GNN-PPI, SemiGNN-PPI, HIGH-PPI) but addresses two specific limitations they ignore: 1) the heterophily problem in PPI graphs that violates GNNs' homophily assumption, and 2) the failure to model dependencies between interaction types. Unlike standard multi-label approaches (CAMEL, CorGCN) that focus on node classification, TELC-PPI specifically targets multi-label edge classification for PPI prediction, which is a distinct challenge requiring different modelling approaches.

## Limitations
The authors acknowledge their model ranks second-best on the BFS split of SHS148k, possibly due to noise introduced during H2 edge construction. The label embedding for "expression" shows divergence between datasets due to limited samples in SHS27k (only 622 samples), highlighting that rare labels require more data for reliable representation. I note that the paper doesn't test their approach on non-human protein networks, which would be important for broader applicability in biomedicine.

## Appendix: Worked Example
Let's walk through the H2 Principle computation for a small example with three proteins: ARF5 (node 0), CYTH3 (node 1), and PSD4 (node 2), based on Figure 1(a) in the paper.

1. **Original adjacency matrix A**:
   ```
   A = [[0, 1, 0],  # ARF5 interacts with CYTH3
        [1, 0, 1],  # CYTH3 interacts with ARF5 and PSD4
        [0, 1, 0]]  # PSD4 interacts with CYTH3
   ```

2. **Compute A² (two-hop paths)**:
   ```
   A² = A @ A = [[1, 0, 1],
                 [0, 2, 0],
                 [1, 0, 1]]
   ```

3. **Compute candidate edges Acand = max(A² - A, 0) ⊙ (1 - I)**:
   ```
   A² - A = [[1, -1, 1],
             [-1, 2, -1],
             [1, -1, 1]]
   max(A²-A, 0) = [[1, 0, 1],
                   [0, 2, 0],
                   [1, 0, 1]]
   ⊙ (1 - I) = [[0, 0, 1],
                [0, 0, 0],
                [1, 0, 0]]
   So Acand = [[0, 0, 1],
               [0, 0, 0],
               [1, 0, 0]]
   ```

4. **Compute node label distributions (simplified)**:
   - ARF5 (node 0): Interacts as "activation" (2 times) and "binding" (1 time) → p0 = [0.67, 0.33, 0, 0, 0, 0, 0] (7 interaction types)
   - CYTH3 (node 1): Interacts as "binding" (2 times) and "catalysis" (1 time) → p1 = [0.67, 0, 0.33, 0, 0, 0, 0]
   - PSD4 (node 2): Interacts as "binding" (1 time) and "catalysis" (1 time) → p2 = [0.5, 0, 0.5, 0, 0, 0, 0]

5. **Compute similarity score for candidate edge (0,2)**:
   ```
   sim(0,2) = α * Acand[0,2] + (1-α) * (p0^T p2)/(||p0||_2 ||p2||_2)
   With α = 0.4 (chosen from sensitivity analysis):
   = 0.4 * 1 + 0.6 * (0.67*0.5 + 0.33*0.5) / (1 * 1)
   = 0.4 + 0.6 * 0.55
   = 0.73
   ```

6. **Sample H2 edges**:
   The similarity score for edge (0,2) is 0.73. With ρ = 0.4 (sampling ratio), the edge will be included with probability proportional to 0.73 (normalized across all candidates).

This example demonstrates how TELC-PPI identifies the functional relationship between CYTH3 and PSD4 through their shared interaction with ARF5, connecting them with a new H2 edge that reflects their shared Sec7 functional domain (as illustrated in Figure 1a).

## References

- **Code:** https://github.com/dengbin151/TELC-PPI
- Bin Deng, Huifang Ma, Ruijia Zhang, Meihuizi Jia, Rui Bing, "Topology-Enhanced and Label Correlation-Aware Model for Protein-Protein Interaction Prediction", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36980

Tags: #biomedicine #graph-neural-networks #multi-label-classification #label-correlation
