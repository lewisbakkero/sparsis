---
title: "S²Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36997"
---

## Executive Summary
S²Drug is a two-stage contrastive learning framework that integrates protein sequence information with 3D structural data for virtual screening in drug discovery. By addressing redundancy and noise in large-scale protein-ligand datasets through specialized data sampling and introducing a binding site prediction auxiliary task, S²Drug achieves consistent performance improvements over state-of-the-art methods on standard drug discovery benchmarks.

## Why This Matters for Practitioners
If you're building virtual screening pipelines that currently rely solely on 3D structural data, this paper suggests you should implement a two-stage training approach that incorporates protein sequence information. Specifically, you can immediately adopt the bilateral data sampling strategy described in the paper: apply homology-aware downweighting using MMseqs2 at a 40% identity threshold and functional deduplication based on UniProt annotations to reduce protein-side redundancy, while filtering ligands with high affinity variability (σn > 1.0) and removing frequent hitters (f(Ln) > 20). These techniques can improve model generalizability without requiring additional structural data, which is particularly valuable when working with limited or noisy protein-ligand interaction datasets. For production systems, prioritize incorporating sequence data during pretraining (on ChemBL or similar large-scale datasets) before fine-tuning on structural data (PDBBind), as this approach outperforms all baselines by 8-13 points on AUROC metrics.

## Problem Statement
Current virtual screening systems operate like a chef trying to cook a complex dish using only the final plated presentation, ignoring the recipe that created it. Most approaches rely exclusively on 3D structural data (the "plated dish"), while neglecting the protein sequence (the "recipe"). This makes models overly sensitive to minor structural variations (like a chef slightly rearranging the garnish), causing them to miss genuine binding candidates and struggle with proteins that have different conformations (like a dish that changes shape when served).

## Proposed Approach
S²Drug is a two-stage framework that bridges protein sequence and 3D structure for virtual screening. The first stage performs sequence pretraining on ChemBL using an ESM2-based backbone with bilateral data sampling to reduce redundancy and noise. The second stage fine-tunes on PDBBind by fusing sequence and structure information through a residue-level gating module while introducing an auxiliary binding site prediction task.

```python
def s2drug_pipeline(protein_sequence, ligand_structure):
    # Stage 1: Sequence Pretraining on ChemBL
    sequence_embeddings = esm2_encoder(protein_sequence)
    clean_dataset = bilateral_data_sampling(chembl_dataset)
    trained_sequence_model = contrastive_train(sequence_embeddings, clean_dataset)
    
    # Stage 2: Sequence-Structure Fusion Finetuning on PDBBind
    structure_embeddings = uni_mol_encoder(ligand_structure)
    fused_representation = sequence_structure_fusion(
        trained_sequence_model, 
        structure_embeddings,
        binding_site_prediction=auxiliary_task()
    )
    return fused_representation
```

## Key Technical Contributions
The paper introduces several key technical innovations that address challenges in integrating protein sequences with 3D structural data for virtual screening.

1. **Bilateral Data Sampling Strategy**: The paper employs a two-sided data filtering approach that specifically targets redundancy and noise at both the protein and ligand levels. For protein-side redundancy, it clusters sequences using MMseqs2 at a 40% identity threshold and applies homology-aware downweighting with α = 0.5 to penalize large protein families while maintaining diversity. For ligand-side noise, it implements affinity variability filtering (retaining only instances with σn < 1.0) and frequent hitter removal (discarding ligands with f(Ln) > 20 unless they show consistent strong binding).

2. **Residue-Level Gating Module**: This mechanism dynamically fuses sequence and structure information at the residue level with a learnable gate parameter. The gate function βn,i = σ(Wβ^⊤[Wsxs_n,i; Wgxg_n,i] + bβ) adaptively determines the contribution of sequence and structure features for each residue based on their local alignment. The paper shows this allows the model to prioritize sequence information for residues with strong binding patterns and structure information for residues with complex spatial arrangements.

3. **Binding Site Prediction Auxiliary Task**: The authors introduce a novel auxiliary task that predicts whether each residue belongs to the binding site, which is formulated as a sequence labelling problem. This task uses a shared attention mechanism to evaluate interaction relevance between residue sequence representations and ligand probes, with predictions trained using binary cross-entropy loss. The paper demonstrates this improves the model's understanding of spatial arrangements in pocket regions.

4. **Two-Stage Training Paradigm**: The framework's two-stage approach separates sequence pretraining (on large-scale ChemBL data) from sequence-structure fusion fine-tuning (on smaller-scale PDBBind data). This addresses a key limitation in prior work: the inability to effectively leverage the complementarity of sequence data (more accessible) and structural data (more specific) in the same training process.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
S²Drug consistently outperforms all baselines on both DUD-E and LIT-PCBA virtual screening benchmarks. On DUD-E, S²Drug achieves 92.46% AUROC (marked with * for statistical significance at p < 0.01), which is 13.01 points higher than DrugCLIP (79.45%) and 8.73 points higher than DrugHash (83.73%). For the more challenging LIT-PCBA dataset, S²Drug achieves 58.23% AUROC, outperforming DrugCLIP (56.36%) and DrugHash (54.58%) by 1.87 and 3.65 points respectively. The paper reports statistically significant improvements (p < 0.01) for these results across all metrics (AUROC, BEDROC, EF0.5%, EF1%, EF5%).

The paper also demonstrates strong generalizability across homology exclusion scenarios. S²Drug maintains superior performance across all identity cutoffs (90%, 60%, 30%, and HMM), outperforming DrugCLIP and DrugHash even under strict remote homology exclusion. This evidence suggests their method significantly reduces dependency on high train-test distribution similarity.

## Related Work
The paper situates itself within the broader landscape of virtual screening methods, noting that while docking-based methods (like Glide-SP and Vina) simulate physical binding processes, learning-based methods have gained prominence by directly predicting interactions from data. The authors observe that existing learning-based approaches, including DrugCLIP and DrugHash, primarily rely on structural data while neglecting protein sequences.

The paper also situates itself within protein representation learning, noting that while sequence representation learning has advanced in other domains (e.g., using ESM2 models), its integration with structural data for virtual screening remains underexplored. The authors build upon prior work in contrastive learning for protein-ligand interactions but address the critical gap of sequence information integration.

## Limitations
The paper's evaluation primarily focuses on virtual screening performance, with limited discussion of computational efficiency or scalability to very large datasets. The authors note that the binding site prediction task was primarily introduced to enhance virtual screening performance, and they don't provide specific analysis of how much each component contributes to the overall improvement.

The paper doesn't evaluate the framework's performance on extremely large-scale drug discovery pipelines, suggesting that further work is needed to assess its practicality in industrial settings with millions of potential ligands. Additionally, the authors don't explore how their framework might perform with different protein sequence encoders beyond ESM2 or with alternative structural encoders beyond Uni-Mol.

## Appendix: Worked Example
Based on the paper's description, here's a step-by-step walkthrough of how a single protein-ligand pair flows through S²Drug's sequence-structure fusion module with concrete values.

Consider a protein with a binding pocket containing 20 residues (a typical number for a drug binding site). The sequence encoder (ESM2) outputs 640-dimensional embeddings per residue. For simplicity, let's consider residues 5, 10, and 15 of interest.

For residue 5, the sequence embedding is [0.2, 0.4, -0.1, ..., 0.8] (640 dimensions) and the structure embedding (from mean-pooling atom-level representations) is [0.1, -0.2, 0.3, ..., -0.5] (640 dimensions). The gating mechanism calculates β5 = σ(Wβ^⊤[Wsx5; Wgx5] + bβ) = 0.75, meaning 75% of the representation comes from sequence information.

The fused representation for residue 5 is xf5 = 0.75 × Ws × xs5 + 0.25 × Wg × xg5. The learnable matrices Ws and Wg project the features into a shared space, resulting in a fused embedding of [0.15, 0.35, 0.05, ..., 0.6].

For residue 15, where sequence and structure information show poor alignment (β15 = 0.3), the fused representation is xf15 = 0.3 × Ws × xs15 + 0.7 × Wg × xg15, shifting more weight to the structural features.

After processing all 20 pocket residues through this gating mechanism, the model applies two Transformer layers and performs mean pooling to obtain the final fused pocket representation of dimension 640, which is then used for virtual screening.

Note: The exact values for Ws, Wg, and βn,i are learned during training and not specified in the paper, so the example uses representative values. The paper confirms that this approach allows the model to dynamically prioritize the most informative modality for each residue based on local sequence-structure alignment.

## References

- Bowei He, Bowen Gao, Yankai Chen, Yanyan Lan, Chen Ma, "S²Drug: Bridging Protein Sequence and 3D Structure in Contrastive Representation Learning for Virtual Screening", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36997

Tags: #biomedicine #drug-discovery #contrastive-learning #data-sampling #residue-gating
