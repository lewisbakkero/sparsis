---
title: "DS-ProGen: A Dual-Structure Deep Language Model for Functional Protein Design"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37037"
---

## Executive Summary
DS-ProGen introduces a dual-structure deep language model that integrates both backbone geometry and surface chemical features for inverse protein folding, achieving a state-of-the-art 61.47% recovery rate on the PRIDE benchmark. This approach enables production systems to generate functionally relevant protein sequences with improved structural fidelity and binding capabilities compared to single-modality methods.

## Why This Matters for Practitioners
If your team builds protein design systems for drug discovery or synthetic biology, DS-ProGen's dual-structure approach provides immediately actionable improvements. Specifically, when developing tools for short-sequence protein design (under 100 residues), this method delivers 20 percentage points higher recovery rates than the next-best model, directly translating to higher success rates in producing viable therapeutic candidates. You should consider implementing this dual-structure approach in your sequence modelling pipelines, particularly for applications requiring high binding specificity to ligands or ions, rather than relying solely on backbone geometry. This means moving beyond legacy methods like ProteinMPNN that focus exclusively on backbone coordinates, and incorporating surface chemical features through a dual-encoder architecture.

## Problem Statement
Current protein design systems face a fundamental trade-off similar to a city planning with only topographical maps versus only street-level photos: you can either see the overall city layout (backbone geometry) or the detailed street features (surface chemistry), but not both simultaneously. The former captures the city's general shape but misses street-level details like building materials and pedestrian pathways (surface chemistry), while the latter shows street features but loses how the streets connect to form the city's structure. This limitation means existing inverse folding methods either miss crucial surface chemistry for binding interactions or fail to maintain structural integrity during sequence design.

## Proposed Approach
DS-ProGen overcomes this limitation with a dual-structure architecture that processes both backbone geometry and surface chemical features in parallel, then fuses them for sequence generation. The backbone encoder extracts geometric features from backbone atoms, while the surface encoder processes atomic types, surface points, and curvature features. These dual streams are projected into a unified space and fed into a Transformer decoder that predicts amino acids autoregressively.

```python
def generate_protein_sequence(structure):
    backbone_features = backbone_encoder(structure)
    surface_features = surface_encoder(structure)
    combined_features = fusion_layer(backbone_features, surface_features)
    sequence = autoregressive_decoder(combined_features)
    return sequence
```

## Key Technical Contributions
The core innovation lies in how DS-ProGen integrates surface chemistry with geometric structure through specific architectural choices:

1. **Dual-branch encoding with geometric vector processing**: Unlike prior methods using only backbone coordinates, DS-ProGen's backbone encoder employs Geometric Vector Perceptron (GVP) layers (Jing et al., 2020) to process vector features like dihedral angles while maintaining rotation invariance. This allows the model to capture both global topology and local geometric relationships through rotation-equivariant vector channels and rotation-invariant scalar channels.

2. **Surface feature enrichment with curvature and chemical context**: The surface encoder doesn't just process atom types, it computes detailed features around each surface point including normal vectors, multiscale mean and Gaussian curvatures across five radii (1Å to 10Å), and the 16 nearest neighboring atoms with their distances and types. This creates a comprehensive chemical and geometric embedding of the protein's exterior surface.

3. **Fusion layer with structural context integration**: The model doesn't simply concatenate features but combines backbone and surface embeddings through a learned fusion layer (R = B + S) before feeding them as context to the sequence decoder. This preserves the distinct information from both modalities while enabling their joint use in sequence prediction.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
DS-ProGen achieves 61.47% recovery rate on the PRIDE test set, significantly outperforming all baselines across all sequence lengths:

| Model              | len<100 | 100≤len<300 | 300≤len<500 | Overall |
|--------------------|---------|-------------|-------------|---------|
| ProteinMPNN        | 41.63   | 48.61       | 52.07       | 48.44   |
| PiFold             | 43.75   | 52.24       | 55.33       | 51.74   |
| ESM-IF             | 39.73   | 52.72       | 58.64       | 51.85   |
| DS-ProGen (backbone-only) | 43.14 | 55.18       | 60.27       | 52.61   |
| **DS-ProGen**      | **63.50** | **64.46**   | **63.15**   | **61.47** |

The most significant improvement appears for short sequences (len<100), where DS-ProGen reaches 63.50%, nearly 20 percentage points higher than the second-best model (InstructPLM at 45.07%). For structural fidelity, DS-ProGen achieves the lowest RMSD (1.401Å) among all models while maintaining a high TM-Score (87.30%).

## Related Work
DS-ProGen builds upon the observation that previous protein language models have focused on either backbone geometry (like ProteinMPNN and ESM-IF) or surface features (like SurfPro), but not both simultaneously. While Nijkamp et al. (2023) and Ferruz et al. (2022) explored autoregressive protein sequence modelling, their approaches lacked dual-structure integration. DS-ProGen extends this work by demonstrating that the synergistic combination of backbone and surface information creates a more comprehensive structural representation, directly addressing the limitation noted in prior work where "backbone coordinates fail to capture crucial chemical features on the exterior surface" while "surface information overlooks the influence of backbone conformations."

## Limitations
The authors acknowledge that the model was evaluated primarily on the PRIDE benchmark without extensive testing on more complex protein design tasks like enzyme engineering. The paper doesn't report statistical significance tests for the reported performance gains, though the authors claim these results are "consistently outperforming all baselines across sequence length ranges." Additionally, while the model integrates surface chemistry, it doesn't explicitly model protein dynamics or conformational changes over time, which could be important for understanding functional mechanisms. The paper doesn't address computational costs or inference latency, which would be critical for production deployment considerations.

## Appendix: Worked Example
Consider a protein with 100 residues that needs to bind a specific ligand. The backbone encoder processes the 3D coordinates of N, C, and Cα atoms (3 coordinates per residue × 100 residues = 300 coordinates), which are encoded into a tensor c_i ∈ R^100×3×3. The backbone geometric encoder then applies 4-layer GVP operations to extract scalar (dihedral angles, pairwise distances) and vector (local spatial orientations) features, resulting in a backbone embedding B ∈ R^100×1024.

Concurrently, the surface encoder processes the protein surface: it extracts atom coordinates and types for 6 common element types (C, N, O, S, Se, H), then generates surface points (up to 8,192 points as per paper's implementation), which are processed into local patches using farthest point sampling and KNN. For each surface point, it calculates normal vectors, multiscale curvatures (10 features across 5 radii), and the 16 nearest neighboring atoms with distances and types (16×7 features).

After processing, the surface features are transformed into S ∈ R^100×1024. The fusion layer combines the backbone and surface embeddings (R = B + S), and the sequence decoder predicts each amino acid autoregressively, conditioning on the structural context. For this 100-residue protein, DS-ProGen would achieve a sequence recovery rate of approximately 63.50%, nearly 20 percentage points higher than the second-best model (InstructPLM at 45.07%).

## References

- Yanting Li, Jiyue Jiang, Zikang Wang, Ziqian Lin, Dongchen He, Yuheng Shan, "DS-ProGen: A Dual-Structure Deep Language Model for Functional Protein Design", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37037

Tags: #biomedicine #protein-design #dual-structure #geometric-encoding #surface-chemistry
