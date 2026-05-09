---
title: "Accelerating Triangle Counting with Real Processing-in-Memory Systems"
venue: "arXiv cs.DC"
paper_url: "https://arxiv.org/abs/2505.04269"
---

## Executive Summary
This paper presents the first triangle counting (TC) algorithm optimised for the UPMEM Processing-in-Memory (PIM) architecture, addressing memory bandwidth limitations through vertex colouring, reservoir sampling, and Misra-Gries summaries. It demonstrates that their implementation outperforms state-of-the-art CPU-based TC solutions for dynamic graphs in Coordinate List (COO) format, making it particularly relevant for real-time graph analysis workloads in production systems.

## Why This Matters for Practitioners
If you're maintaining production systems that process dynamic graphs in COO format (such as social network analysis, security threat detection, or recommendation systems), this paper shows you should consider migrating memory-bound graph algorithms to PIM architectures. Specifically:
- Implement vertex colouring to eliminate costly inter-core communication in PIM systems
- Apply reservoir sampling for graphs larger than available PIM memory (with <0.5% relative error)
- Use Misra-Gries summaries to handle high-degree nodes (improving throughput by up to 3.5x)
- Evaluate PIM for dynamic graph workloads in COO format rather than static graphs in CSR format
- Recognise that PIM architectures provide significant advantages over CPU implementations for memory-bound graph algorithms, especially when graphs change frequently

## Problem Statement
Current graph processing systems struggle with triangle counting (TC) because it's memory-bound: CPUs and GPUs require repeated memory accesses across large data regions with low data reuse. Imagine trying to count all the triangles in a city's road network while constantly fetching the map from a distant library - you'd spend more time walking to the library than counting triangles on the actual road network.

## Proposed Approach
The authors' solution uses a three-layer strategy to optimise TC for UPMEM PIM:
1. Vertex colouring partitions graph edges across PIM cores without communication
2. Reservoir sampling handles memory constraints at the PIM core level
3. Uniform sampling at the host level reduces data volume before transfer

The system reads COO-format graphs, assigns edges to PIM cores using vertex colouring, processes edges with reservoir sampling when memory is full, and uses Misra-Gries to optimise for high-degree nodes.

```python
def assign_edges_to_cores(edges, colours):
    # Assign random colour to each node
    node_color = {node: random.randint(0, colours-1) for node in all_nodes}
    
    # Generate unique colour triplets for each PIM core
    core_triplets = generate_unique_triplets(colours)
    
    # Assign edges to cores based on colour compatibility
    edge_assignments = defaultdict(list)
    for (u, v) in edges:
        c1, c2 = node_color[u], node_color[v]
        for core, triplet in core_triplets.items():
            if (c1, c2) in compatible_pairs(triplet):
                edge_assignments[core].append((u, v))
    
    return edge_assignments
```

## Key Technical Contributions
The core innovation is optimising TC for PIM constraints through three specific mechanisms:

1. **Vertex colouring for communication-free partitioning**: Unlike previous PIM approaches requiring inter-core communication, their node colouring technique assigns colours randomly to nodes and assigns edges to PIM cores based on compatible colour triplets. Each edge is assigned to cores where the colour triplet matches (e.g., triplet (0,1,2) matches edge colours (0,1), (1,2), or (0,2)). The authors show even load balancing across cores (with C colours, C cores handle N edges, 2×(C choose 2) cores handle 3N edges, and (C choose 3) cores handle 6N edges), ensuring performance scales with core count.

2. **Reservoir sampling for memory-constrained processing**: When a PIM core's DRAM fills up, the system uses reservoir sampling to maintain a representative sample of edges. For the t-th edge, if t ≤ M (memory limit), it's stored; if t > M, a random edge is replaced with probability M/t. This approximation is statistically corrected using T_adjusted = T × [M(M-1)(M-2)] / [t(t-1)(t-2)], ensuring accuracy within a low relative error (less than 0.5% in experiments).

3. **Misra-Gries summary for high-degree node optimisation**: For graphs with high-degree nodes, the authors use the Misra-Gries summary algorithm during host processing to identify high-degree nodes. This summary uses a hash table that maintains up to K frequent nodes, with frequencies adjusted to keep the table size bounded. The top t-degree nodes are remapped to new IDs (highest new ID for most frequent node), ensuring edges from high-degree nodes are processed in an order that minimises unnecessary comparisons during triangle counting.

## Experimental Results
The authors evaluated their implementation on UPMEM using graphs from the SNAP dataset. Key findings:

- For dynamic graphs in COO format, their PIM implementation outperformed state-of-the-art CPU implementations by up to 2.7× (Figure 3 shows graphs with maximum degrees in the tens of thousands having approximately 10,000 edges/ms, while graphs with degrees in the hundreds of thousands had approximately 3,000 edges/ms).
- The Misra-Gries summary improved throughput by up to 3.5× for graphs with high-degree nodes (Figure 3 shows significant throughput degradation without this optimisation).
- Reservoir sampling with M=100,000 edges maintained a relative error of less than 0.5% across all graphs tested.
- The PIM implementation did not outperform CPU implementations for static graphs in CSR format (the authors explicitly state this is not their goal), but outperformed CPU solutions for dynamic COO graphs.

The paper doesn't explicitly report statistical significance testing, but the consistent improvements across multiple graphs indicate robust results.

## Related Work
This paper builds on previous TC research on CPU ([51], [53]) and GPU ([52]) architectures, but makes a key distinction: while prior PIM work was theoretical or required custom graph representations, this is the first implementation on a real PIM system (UPMEM) that works directly with standard COO format without special compression. The authors acknowledge that their work differs from previous PIM approaches by addressing real hardware constraints (limited memory, cross-core communication costs) rather than just proposing architectures.

## Limitations
The authors acknowledge that their implementation targets dynamic graphs in COO format specifically; for static graphs in CSR format, CPU implementations outperform the PIM approach. The paper doesn't test against GPU implementations for static graphs, though it notes CPU implementations outperform their solution for static graphs. They also don't test on graphs with billion+ edges due to hardware constraints. 

My assessment: The paper's focus on dynamic COO graphs is both a strength (addressing a real production need) and limitation (not generalising to all graph workloads). The experimental scope is constrained by UPMEM's hardware limits, and the paper doesn't explore scaling to larger PIM systems beyond their tested configuration.

## Appendix: Worked Example
Let's walk through a concrete example of how the algorithm processes a graph with high-degree nodes:

Consider a graph with 10,000 edges where the top 50 nodes have degrees above 500 (i.e., they connect to more than 500 other nodes). The host processor reads the COO representation:

1. **Misra-Gries Summary**: The host uses K=50 for the summary. As it processes edges, it maintains a hash table of up to 50 nodes with highest degrees. After processing, the top 50 high-degree nodes (including nodes 100, 200, 300, etc.) are identified.

2. **Node Remapping**: These 50 nodes are remapped to new IDs (e.g., node 100 → 10,000, node 200 → 10,001, ..., node 10,0049 → 10,049).

3. **Edge Processing**: Each edge (u, v) is remapped if u or v is in the top 50. For example, edge (100, 101) becomes (10,000, 101), and edge (200, 101) becomes (10,001, 101).

4. **Sorting and Counting**: The PIM core sorts edges by first node ID. Edges like (10,000, 101) appear before edges with lower new IDs (e.g., (101, 102)).

5. **Triangle Counting**: Processing edge (10,000, 101), the PIM core searches for edges starting with node 101 (which are lower than 10,000). Because high-degree node 100 is remapped to 10,000, edges connecting to it (like (10,000, 101)) are processed earlier, reducing the number of comparisons needed to find matching neighbours (edges with common second nodes).

This remapping ensures triangles involving high-degree nodes are counted with minimal unnecessary comparisons, improving throughput significantly for graphs with high-degree nodes.

## References

- Lorenzo Asquini, Manos Frouzakis, Juan Gómez-Luna, Mohammad Sadrosadati, Onur Mutlu, Francesco Silvestri, "Accelerating Triangle Counting with Real Processing-in-Memory Systems", arXiv cs.DC, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2505.04269

Tags: #graph-processing #memory-bound-algorithms #processing-in-memory #triangle-counting #reservoir-sampling #misra-gries #vertex-coloring
