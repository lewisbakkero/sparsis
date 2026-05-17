---
title: "A Super Fast K-means for Indexing Vector Embeddings"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.20009"
---

## Executive Summary
SuperKMeans is a specialized k-means variant that accelerates clustering of high-dimensional vector embeddings by up to 7× on CPUs and 4× on GPUs while maintaining indexing quality. It solves the critical bottleneck in building vector search indexes where traditional k-means implementations become compute-bound with modern embedding dimensions (hundreds to thousands).

## Why This Matters for Practitioners
If you're building or maintaining vector search systems using IVF indexes for large embedding collections (1M+ vectors), SuperKMeans could dramatically reduce index construction time without compromising search quality. For example, if your current indexing process takes 10 hours with FAISS on CPU, SuperKMeans could cut this to 1.5 hours, freeing up compute resources for other workloads. If you're using GPU-based vector search, SuperKMeans could reduce indexing time by 75% compared to cuVS. Crucially, these speedups come without additional tuning, SuperKMeans works with standard IVF pipelines and maintains the same recall metrics as existing approaches.

## Problem Statement
Current vector indexing systems are stuck in a performance trap: the more dimensions you have in your embeddings (1536+), the more the standard k-means algorithm becomes a compute bottleneck. Imagine trying to sort a million books in a library where each book has 1536 unique features (not just title or author), and you need to group them by all those features simultaneously. Traditional sorting methods would waste time checking irrelevant features, while SuperKMeans cleverly skips those irrelevant dimensions during clustering.

## Proposed Approach
SuperKMeans optimises the k-means core loop through two key phases: GEMM (computing partial distances with front dimensions) followed by PRUNING (removing dimensions unlikely to change assignments). It uses random rotation to spread variance evenly across dimensions, enabling effective pruning via Adaptive Sampling. For production use, it integrates directly with existing IVF indexing pipelines and includes Early Termination by Recall (ETR) that stops clustering when retrieval quality stops improving.

```python
def super_kmeans(X, k, max_iter=10):
    # Random rotation to enable effective pruning
    R = random_rotation_matrix(X.shape[1])
    X_rotated = X @ R
    
    # Initialize centroids randomly (avoiding k-means++)
    Y_rotated = random_sample(X_rotated, k)
    
    # Set initial dimension cutoff (12.5% of total)
    d_prime = int(X.shape[1] * 0.125)
    
    for iteration in range(max_iter):
        # GEMM phase: compute partial distances with front d_prime dimensions
        distances = -2 * X_rotated[:, :d_prime] @ Y_rotated[:, :d_prime].T + \
                   np.sum(X_rotated[:, :d_prime]**2, axis=1, keepdims=True) + \
                   np.sum(Y_rotated[:, :d_prime]**2, axis=1)
        
        # PRUNING phase: progressively prune dimensions
        for i in range(X_rotated.shape[0]):
            current_min = np.min(distances[i])
            for j in range(Y_rotated.shape[0]):
                if not is_prunable(distances[i, j], current_min, d_prime):
                    # Continue evaluating remaining dimensions
                    for dim in range(d_prime, X_rotated.shape[1]):
                        distances[i, j] += (X_rotated[i, dim] - Y_rotated[j, dim])**2
                    if distances[i, j] < current_min:
                        current_min = distances[i, j]
                        assignment[i] = j
        
        # Update centroids and split empty clusters
        Y_rotated = update_centroids(X_rotated, assignment, k)
        
        # Early Termination by Recall (check if quality improves)
        if etr_enabled and not recall_improved():
            break
    
    # Return unrotated centroids for IVF indexing
    return Y_rotated @ R.T
```

## Key Technical Contributions
The true novelty lies in how SuperKMeans efficiently prunes dimensions without sacrificing quality, something existing approaches failed to achieve.

1. **Random Rotation for Pruning Feasibility**: By applying a random orthogonal rotation (effectively a GEMM) to the data, SuperKMeans spreads variance evenly across all dimensions. This ensures no single dimension carries disproportionate variance, making it possible to reliably prune dimensions using Adaptive Sampling. The rotation is negligible overhead compared to the core loop.

2. **Progressive Pruning with PDX Layout**: Instead of binary pruning (keep or discard), SuperKMeans progressively prunes dimensions in 64-block chunks. It stores the trailing dimensions in a block-column-major order of 64 dimensions (PDX layout), enabling efficient cache utilisation and cache-friendly data access patterns. This design choice prevents the performance degradation seen in previous pruning approaches (e.g., Elkan's algorithm) that wasted time on inefficient memory access.

3. **Initial Threshold for PRUNING Phase**: To avoid the overhead of a full GEMM on the first batch, SuperKMeans uses the centroid assignment from the previous iteration to establish an initial pruning threshold. This allows it to prune over 95% of centroids in the first PRUNING phase based solely on GEMM output, without introducing bias.

4. **Early Termination by Recall (ETR)**: Unlike prior approaches that required users to manually set iteration counts, ETR monitors recall quality during clustering. If the recall doesn't improve over two iterations, it terminates early. The implementation computes ground truth recall at minimal overhead (measured in Subsection 4.7), making it practical for production use.

## Experimental Results
SuperKMeans achieves 7× speedup over FAISS on CPU (1M vectors, 1536 dimensions) and 4× over cuVS on GPU (Section 4.3, Figure 1). Crucially, it maintains equivalent recall quality, both on the OpenAI dataset and in cross-dataset experiments. The paper shows that just 5, 10 iterations are sufficient for optimal retrieval quality (Section 4.4), and using only 20, 30% of data points yields high-quality centroids (Section 4.12). The authors confirm these improvements are statistically significant through multiple controlled experiments across different hardware architectures (Intel Granite Rapids, AMD Zen 5, Apple M4, etc.).

## Related Work
SuperKMeans positions itself directly against FAISS, the current state-of-the-art for vector embedding clustering, by addressing the specific limitations that make FAISS suboptimal for high-dimensional embeddings. Unlike previous k-means variants that attempted to prune using the triangle inequality (which fails under high dimensionality), SuperKMeans leverages Adaptive Sampling (ADSampling) for effective dimension pruning. It also differs from Elkan's variant, which the authors show is slower than full GEMM due to inefficient data access patterns.

## Limitations
The paper focuses on CPU and GPU implementations, but doesn't explore TPU optimisation. The ETR mechanism requires ground truth data for recall measurement, which may be challenging to obtain in some production environments. The authors acknowledge that SuperKMeans is specifically designed for IVF indexing pipelines, so it doesn't directly apply to other vector index types like graph-based indexes.

## Appendix: Worked Example
Let's walk through how SuperKMeans processes a single vector during the PRUNING phase with concrete numbers. Consider a vector embedding dimension of 1024 (d=1024) and a cutoff dimension d'=128 (12.5% of 1024). The GEMM phase computes distances using only the first 128 dimensions, yielding partial distances for each centroid.

For a specific vector, the partial distances to centroids are: [15.2, 12.8, 8.1, 16.3, 20.7, 22.4]. The current minimum is 8.1 (centroid 3). Using Adaptive Sampling's hypothesis testing, the algorithm determines that centroid 1 can be safely pruned because the partial distance (15.2) exceeds a threshold calculated from the current minimum (8.1). The threshold calculation is: θ(128, τ) = 8.1 * (1 + 0.2) = 9.72, making centroid 1's partial distance (15.2) irrelevant for assignment.

The algorithm then explores the next 64 dimensions (128-192) for centroid 3 (the current best). After processing these dimensions, the partial distance becomes 8.3, slightly worse than the initial 8.1. The algorithm continues exploring the remaining dimensions (192-256, 256-320, etc.) until it either confirms centroid 3 remains the best assignment or finds a better centroid.

This process prunes 97% of centroids at the start of the PRUNING phase (Section 3.1), avoiding the need to compute full distances for most centroids. The progressive pruning approach ensures that only a small fraction of centroids requires full distance calculation (less than 3% in practice).

## References

- **Code:** https://github.com/cwida/SuperKMeans.
- Leonardo Kuffo, Sven Hepkema, Peter Boncz, "A Super Fast K-means for Indexing Vector Embeddings", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20009

Tags: #vector-search #k-means #indexing #vector-embeddings #gpu-optimisation
