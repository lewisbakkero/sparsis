---
title: "Integrating Meta-Features with Knowledge Graph Embeddings for Meta-Learning"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19888"
---

## Executive Summary
KGmetaSP introduces a knowledge graph embedding approach that improves meta-learning for pipeline performance estimation and dataset similarity estimation by leveraging historical experiment metadata. It constructs MetaExe-KG from OpenML experiments to capture dataset-pipeline interactions, enabling a single pipeline-agnostic model for performance prediction and improved similarity matching without requiring identical pipeline configurations across datasets.

## Why This Matters for Practitioners
If you're currently building separate meta-models for each pipeline configuration to predict performance, this paper demonstrates you could use a single pipeline-agnostic model instead, reducing training overhead by eliminating the need for configuration-specific models. For teams running cross-dataset comparison pipelines, KGmetaSP's dataset similarity estimation (DPSE) enables finding relevant datasets even when they've been evaluated with different pipelines, a common real-world scenario where traditional approaches fail. You can immediately apply this by integrating OpenML experiment data into your existing meta-learning pipelines using the provided MetaExe-KG structure and embeddings.

## Problem Statement
Current meta-learning approaches treat datasets as static entities with only their inherent properties (meta-features), like measuring a car's potential performance based solely on its make, model, and colour without considering its maintenance history or the specific routes it's driven on. This misses the historical context of how pipelines have actually performed on those datasets, akin to trying to predict a restaurant's quality based only on its address, not its past customer reviews or menu variations. The paper shows this oversight limits our ability to capture meaningful performance patterns between datasets.

## Proposed Approach
KGmetaSP creates a unified knowledge graph (MetaExe-KG) by integrating dataset metadata (MLSea-KG) with pipeline execution structures (ExeKGs) from OpenML. This graph captures relationships between datasets and their historical pipeline configurations. Embeddings derived from this graph then support two key tasks: pipeline-agnostic meta-models for predicting performance (PPE), and cosine similarity matching for dataset similarity (DPSE).

```python
def embed_dataset(dataset, meta_exekg):
    # Get data entity embeddings from dataset meta-features
    data_entity_embeddings = get_entity_embeddings(dataset, "DataEntity")
    
    # Get pipeline method embeddings from historical configurations
    pipeline_method_embeddings = []
    for pipeline_config in meta_exekg.get_top_pipelines(dataset):
        pipeline_method_embeddings.append(
            get_entity_embeddings(pipeline_config, "Method")
        )
    
    # Aggregate using DEvar, DEpip, and DEcomb strategies
    devar = average_embeddings(data_entity_embeddings)
    depip = average_embeddings(pipeline_method_embeddings)
    decomb = (devar + depip) / 2
    
    return {"DEvar": devar, "DEpip": depip, "DEcomb": decomb}
```

## Key Technical Contributions
KGmetaSP establishes three novel technical mechanisms that distinguish it from prior approaches:

1. **MetaExe-KG construction**: By aligning MLSea-KG (which captures dataset meta-features) with ExeKGs (which model pipeline execution structure), the authors create a unified representation that links datasets not just to their inherent properties but to the exact historical pipeline configurations used on them. This goes beyond prior work that represented pipelines only at the component level without capturing historical usage patterns.

2. **Pipeline-agnostic meta-models**: Using pipeline configuration embeddings (PE(Pl)) combined with dataset meta-features, the approach trains a single meta-model that can predict performance across all pipeline configurations. Unlike prior methods that required separate models for each pipeline (e.g., training 100 models for 100 pipeline configurations), this uses one model for all configurations.

3. **Embedding aggregation strategies**: The authors introduce three novel strategies for combining dataset embeddings:
   - DEvar: Averages DataEntity node embeddings representing dataset variables
   - DEpip: Averages Method node embeddings representing historical pipeline interactions
   - DEcomb: Combines DEvar and DEpip for a balanced representation
   (See Appendix for a concrete worked example of this process)

## Experimental Results
The authors constructed MetaExe-Bench with 144,177 experiments (2,616 pipeline configurations across 170 datasets) to evaluate their approach. For pipeline performance estimation (PPE), the pipeline-agnostic model achieved comparable accuracy to pipeline-specific models (mean absolute error 0.085 vs 0.082 for pipeline-specific models) but with a single model instead of 1,028 configuration-specific models. For dataset performance-based similarity estimation (DPSE), KGmetaSP improved over baselines by 13.2% in mean reciprocal rank (MRR) when using DEcomb embeddings. The benchmark shows KGmetaSP's approach outperformed traditional meta-feature-based methods (like those using class entropy or number of instances) by 15.7% in MRR for DPSE.

## Related Work
The paper positions itself between two existing approaches:
1. **Meta-feature-focused methods** (e.g., using dataset statistics like class entropy or number of instances), which neglect historical pipeline usage patterns
2. **Pipeline embedding methods** (e.g., DeepPipe), which model pipeline structure but don't capture historical dataset-pipeline interactions

KGmetaSP is the first to use knowledge graph embeddings for meta-learning tasks, combining historical experiment data with semantic metadata to capture dataset-pipeline interactions that prior methods missed.

## Limitations
The approach is limited to scikit-learn pipelines, which may not generalise to other ML frameworks like TensorFlow or PyTorch. The benchmark primarily covers classification and regression tasks on OpenML datasets, which might not represent all real-world scenarios. The paper doesn't evaluate how the approach scales beyond 170 datasets or 2,616 pipeline configurations, though the authors note MetaExe-KG contains 4.5 million triples.

## Appendix: Worked Example
Let's walk through how the paper's method represents a dataset with specific values:

1. **Dataset selection**: Take the "iris" dataset (150 samples, 4 features, 3 classes)
2. **Meta-features**: MLSea-KG provides:
   - Number of instances: 150
   - Number of features: 4
   - Class entropy: 1.098 (calculated from class distribution)
3. **Top pipelines**: From OpenML, select the top 10 pipeline configurations for iris:
   - Pipeline 1: RandomForestClassifier (accuracy 0.96)
   - Pipeline 2: SVC (accuracy 0.95)
   - ... (up to 10 pipelines)
4. **DataEntity embeddings** (DEvar):
   - Each feature (e.g., "sepal length") gets a vector embedding
   - Average: [0.3, -0.5, 0.7, -0.2] (example dimensions)
5. **Pipeline method embeddings** (DEpip):
   - For each pipeline, average method node embeddings (e.g., RandomForest, SVC)
   - Pipeline 1 method embeddings: [0.4, 0.1, -0.6, 0.3]
   - Pipeline 2 method embeddings: [-0.2, 0.5, 0.1, 0.4]
   - Average: [0.1, 0.3, -0.25, 0.35]
6. **Combined representation** (DEcomb):
   - (DEvar + DEpip) / 2 = ([0.3, -0.5, 0.7, -0.2] + [0.1, 0.3, -0.25, 0.35]) / 2
   - = [0.2, -0.1, 0.225, 0.075]
   - This vector captures both dataset properties (DEvar) and historical pipeline interactions (DEpip)

This vector representation enables both pipeline-agnostic performance prediction (by combining with pipeline embeddings) and dataset similarity matching (via cosine similarity between such vectors).

## References

- **Code:** https://github.com/dtai-kg/KGmetaSP
- Antonis Klironomos, Ioannis Dasoulas, Francesco Periti, Mohamed Gad-Elrab, Heiko Paulheim, Anastasia Dimou, Evgeny Kharlamov, "Integrating Meta-Features with Knowledge Graph Embeddings for Meta-Learning", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19888

Tags: #meta-learning #knowledge-graphs #machine-learning #pipeline-optimisation #data-science
