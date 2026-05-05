---
title: "PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37008"
---

## Executive Summary
PanFoMa introduces a lightweight hybrid neural network combining Transformers and state-space models to efficiently model single-cell transcriptomes for pan-cancer analysis. It addresses the dual challenges of computational efficiency and capturing both local and global gene interactions in cancer genomics. For practitioners, this means achieving better accuracy with reduced compute costs compared to existing foundation models, which is critical for deploying scalable cancer analysis tools in clinical settings.

## Why This Matters for Practitioners
Many current single-cell foundation models like scGPT or GeneFormer process only a subset of genes (typically 2048) due to quadratic computational complexity, excluding low-expression functional genes like transcription factors. If you're building a cancer analysis system that needs to handle full transcriptomes (tens of thousands of genes), PanFoMa enables you to process the complete data without sacrificing accuracy, improving results by up to 7.4% on cell type annotation tasks. This means you can now deploy more accurate cancer subtyping models in production without requiring 3-4x more GPU resources, making it feasible to run these analyses at scale across multiple cancer types in clinical workflows.

## Problem Statement
Existing single-cell foundation models face a fundamental trade-off: Transformers capture complex gene interactions but scale quadratically with gene count (O(n²)), forcing researchers to select only highly variable genes (HVGs), while Mamba-based models offer linear scalability (O(n)) but require fixed gene ordering that ignores biological context. It's like trying to analyse a city's traffic patterns with either a fixed GPS route that ignores real-time congestion (Mamba) or a map that shows every possible connection but becomes unusable for large cities (Transformer). PanFoMa solves this by implementing a hybrid approach where local gene relationships are modelled in parallel chunks (like neighborhoods), and genes are dynamically sorted for global integration (like traffic routing based on real-time conditions).

## Proposed Approach
PanFoMa is structured as a hierarchical hybrid model with two core components: a Local-context Encoder that processes gene interactions within chunks using a lightweight Transformer, and a Global Sequential Feature Decoder that dynamically sorts genes based on their relevance to the cell's global context before processing with a bidirectional Mamba. This design achieves O(C · M² + N log N) computational complexity, balancing expressive power with scalability. The model processes thousands of genes by partitioning them into fixed-size chunks (M=768 genes per chunk), using shared Transformer layers to capture local relationships, then dynamically reordering all genes based on their importance to the cell state before integrating them globally with Mamba.

```python
def panfoma_model(input_genes):
    # Local-context Encoder
    chunks = partition_genes(input_genes, chunk_size=768, num_chunks=4)
    chunk_embeddings = []
    for chunk in chunks:
        # Embed gene IDs and expression values
        chunk_embedding = embed_genes(chunk)
        # Process through shared Transformer layers
        chunk_embedding = transformer_layer(chunk_embedding, num_layers=6)
        chunk_embeddings.append(chunk_embedding)
    
    # Global Sequential Feature Decoder
    global_vector = average_pool([chunk['cls'] for chunk in chunk_embeddings])
    gene_importances = compute_gene_importances(chunk_embeddings, global_vector)
    sorted_genes = sort_genes_by_importance(gene_importances)
    
    # Bidirectional Mamba processing
    forward_mamba = bidirectional_mamba(sorted_genes, direction='forward')
    backward_mamba = bidirectional_mamba(sorted_genes, direction='backward')
    
    # Gated fusion of bidirectional flows
    fused_features = gate_fusion(forward_mamba, backward_mamba)
    return fused_features
```

## Key Technical Contributions
PanFoMa's core innovations address the limitations of both Transformer-based and Mamba-based approaches to single-cell transcriptome modelling. The key mechanisms that make this possible include:

1. **Dynamic gene reordering based on cell-specific context** - Instead of using a fixed gene ordering (like mean expression), PanFoMa computes gene importance scores using a dot product between each gene's local representation and the global cell state vector. This allows the model to adaptively prioritize genes that are functionally relevant to the specific cell's biology, rather than using a one-size-fits-all ordering. The authors demonstrate this is biologically meaningful, as gene importance varies across different cancer subtypes.

2. **Shared-parameter lightweight Transformer for local processing** - By using a six-layer Transformer with shared parameters across layers to process each chunk of genes (768 genes per chunk), PanFoMa reduces computational overhead while still capturing complex local interactions. This is a significant improvement over previous approaches that processed the entire transcriptome with a full Transformer, which would scale quadratically with gene count.

3. **Bidirectional Mamba with gated fusion for global integration** - The model uses a bidirectional Mamba module to process the dynamically sorted gene sequence, then applies a gated fusion mechanism to adaptively weigh information from both directions for each gene. This allows the model to capture bidirectional relationships without the fixed-dimensionality limitations of standard Mamba, which can suffer from "forgetting" distant gene interactions, particularly when modelling distant gene interactions critical for accurate diagnosis.

4. **The pan-cancer benchmark construction pipeline** - The paper details a rigorous data curation process for building PanFoMaBench, which includes standardizing gene identifiers across datasets using HGNC, removing low-quality cells, and filtering out low-expression genes. This produces a high-quality benchmark with 3.5 million cells across 34 cancer subtypes, making it one of the most comprehensive pan-cancer datasets available.

## Experimental Results
PanFoMa outperforms state-of-the-art models across multiple benchmarks:

- **Pan-cancer classification**: Achieved 94.74% accuracy on PanFoMaBench, 3.5% higher than the second-best model (GeneFormer at 91.24%). Macro-F1 improved by 4.0% (92.50% vs 88.51%).
- **Cell type annotation**: Improved accuracy by 7.4% on the MS dataset (85.63% vs 78.25% for GeneMamba), 3.0% on Myeloid b (97.26% vs 94.21% for scGPT), and 1.0% on hPancreas.
- **Batch integration**: Achieved 96.41% Avg batch on the Immune dataset (0.9641 vs 0.9536 for GeneMamba) and 0.9661 Avg batch on Perirhinal Cortex.
- **Multi-omics integration**: Achieved 0.789 Avg bio on 10x Multiome PBMC (3.1% improvement over scGPT's 0.758) and 0.721 Avg bio on BMMC (2.4% improvement over scGLUE's 0.697).

The paper doesn't specify statistical significance testing, but the consistent improvements across multiple datasets and tasks suggest these results are robust.

## Related Work
PanFoMa builds on the recent trend of applying foundation models to single-cell transcriptomics, which has evolved along two architectural lines: Transformer-based models (scGPT, GeneFormer, scFoundation) and Mamba-based models (GeneMamba). While Transformer models capture complex gene interactions, they face quadratic computational complexity when processing full transcriptomes. Mamba-based models offer linear scalability but require fixed gene ordering that ignores biological context. PanFoMa improves upon both lines by introducing a hybrid architecture that preserves Transformer's expressive power while leveraging Mamba's efficiency, and by developing a dynamic gene reordering mechanism that adapts to cell-specific context rather than using a fixed heuristic.

## Limitations
The authors acknowledge that PanFoMa's dynamic gene reordering mechanism requires additional computational overhead for computing gene importance scores, though this is offset by the benefits of more accurate modelling. The benchmark dataset, while comprehensive, is limited to human cancer data, and the model hasn't been evaluated on non-cancer single-cell data. The paper doesn't discuss the computational costs of the dynamic sorting step compared to fixed ordering approaches, though the authors claim this step is efficient. Additionally, the model's performance on rare cancer subtypes with limited data (fewer than 10,000 cells) wasn't specifically evaluated.

## Appendix: Worked Example
Let's walk through PanFoMa's processing of a single cell with 3072 genes (4 chunks of 768 genes each). The Local-context Encoder first processes each chunk through a shared 6-layer Transformer with embedded gene IDs and expression values. After processing, it extracts the [CLS] token for each chunk to form summary vectors.

For cell type annotation, the global cell state vector is calculated as the average of the four [CLS] vectors (each of dimension D). The gene importance scores are computed by dot product between each gene's representation and the global vector, then genes are sorted by these scores.

For example, suppose the global vector is [0.8, 0.2, -0.1, 0.5] (dimension 4 for simplicity). A gene with representation [0.7, 0.3, -0.2, 0.4] would have a score of 0.8*0.7 + 0.2*0.3 + (-0.1)*(-0.2) + 0.5*0.4 = 0.56 + 0.06 + 0.02 + 0.2 = 0.84, making it highly important. A gene with representation [0.1, -0.5, 0.1, -0.3] would have a score of 0.8*0.1 + 0.2*(-0.5) + (-0.1)*0.1 + 0.5*(-0.3) = 0.08 - 0.1 - 0.01 - 0.15 = -0.18, making it less important.

The sorted gene sequence is then fed into the bidirectional Mamba, which processes the sequence in both forward and backward directions. The gated fusion mechanism then combines these streams, with the gate for the first gene being σ(Linear([0.7, 0.3, -0.2, 0.4])) = σ([0.2, 0.1, -0.1, 0.3]), where σ is the Sigmoid function. This gate determines the weight given to forward versus backward information for predicting that gene's representation.

## References

- Xiaoshui Huang, Tianlin Zhu, Yifan Zuo, Xue Xia, Zonghan Wu, "PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37008

Tags: #biomedicine #cancer-diagnosis #single-cell-analysis #transformer #mamba
