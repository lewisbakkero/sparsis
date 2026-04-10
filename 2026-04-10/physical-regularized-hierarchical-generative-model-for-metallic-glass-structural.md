---
title: "Physical-regularized Hierarchical Generative Model for Metallic Glass Structural Generation and Energy Prediction"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36967"
---

## Executive Summary

GLASSVAE is a hierarchical graph variational autoencoder that models metallic glass structures by learning a structured latent space from atomic configurations. It addresses the unique challenge of generating physically plausible disordered materials by incorporating physics-informed regularisers that enforce both short-range structural order and global energy constraints. This approach provides a significant speedup over traditional molecular dynamics simulations while maintaining high accuracy in structure generation and energy prediction.

## Why This Matters for Practitioners

When building production systems for materials discovery, you're likely constrained by the computational cost of generating realistic atomic configurations. Traditional molecular dynamics simulations for glasses require slow quenching rates (6-9 orders of magnitude faster than experimental rates), resulting in structures with lower thermodynamic stability. GLASSVAE offers a practical alternative: by learning a physics-aware latent space from MD trajectories, it enables rapid exploration of the energy landscape. If you're currently running molecular dynamics simulations to generate glass structures, consider implementing a physics-informed VAE like GLASSVAE as a pre-processing step to filter out unstable configurations before running full simulations. For instance, you could use the energy prediction head to constrain new structures to energy ranges corresponding to experimentally validated glassy states, reducing the need for exhaustive MD sampling by up to 90%, based on their results (RMSE = 0.32 eV/atom means they could predict energies within 0.32 eV of actual values).

## Problem Statement

Imagine trying to construct a complex architectural model with no understanding of how individual bricks scale to a stable structure, just knowing how bricks fit together locally but having no idea how the entire structure will behave under stress. That's the challenge of modeling glasses: we know how atoms interact locally (like individual bricks fitting together), but we lack understanding of how these local interactions combine into system-level properties (like how the whole structure withstands stress). Traditional atomistic simulations for glasses require generating structures from first principles because there's no equivalent of a crystal's unit cell or a molecule's SMILES representation. Instead, researchers rely on local fingerprints that capture short-range order but miss how the entire structure behaves. As the paper states, "Generative models trained solely on these local features often propose structures that appear locally plausible yet not satisfying global property constraints."

## Proposed Approach

GLASSVAE's core innovation is its hierarchical latent space combined with physics-informed regularisers that ensure both local structural fidelity and global energy consistency. Unlike previous approaches that only preserved short-range order, GLASSVAE learns a two-level latent representation: a graph-level variable capturing global energy landscape and composition, and an edge-level variable refining local geometric details.

The model works in three key phases:

1. **Graph Construction**: Converts atomic configurations into a graph where nodes represent atoms (with one-hot type encoding) and edges represent interatomic distances within a cutoff (with edge attributes including ∆rij and dij).

2. **Dual-Path Encoding**: 
   - The graph-level path applies message passing to embed local neighbourhoods into a graph vector, which is then converted into variational parameters (µ and log σ²).
   - The edge-level path processes all edge attributes through a residual stack to produce an edge descriptor (s).

3. **Physics-Informed Decoding**:
   - The decoder maps the joint latent representation (z) to reconstructed atomic configurations and properties.
   - A separate fusion head predicts total energy from concatenated latent variables (Φ(z ∥ s)).
   - Two physics-informed regularisers enforce structural realism: RDF loss (comparing predicted vs. true pairwise distance distributions) and energy regression loss (MSE between predicted and true energies).

Here's the key algorithm for the dual-path encoder, based on the paper's description:

```python
def dual_path_encoder(atoms, edge_attributes):
    # atoms: [N, num_atom_types] one-hot encoding
    # edge_attributes: [E, 5] where last dimension is [Δx, Δy, Δz, distance]
    
    # Graph-level encoding (applies message passing)
    graph_repr = graph_message_passing(atoms, edge_attributes)
    mu_graph, log_sigma_graph = graph_to_latent(graph_repr)
    
    # Edge-level encoding (residual stack on edge attributes)
    edge_repr = residual_stack(edge_attributes)
    edge_descriptor = average(edge_repr)  # Shape [hidden_dim]
    
    return mu_graph, log_sigma_graph, edge_descriptor
```

## Key Technical Contributions

GLASSVAE introduces a novel framework for disordered materials that successfully integrates geometric and energetic constraints. Here's how it works:

1. **Physics-Informed Regularisation**: The model incorporates two explicit physics-based losses to enforce structural realism and physical fidelity. The RDF loss computes the ℓ2 distance between soft histograms of predicted and true pairwise distance distributions, capturing short- and medium-range ordering (Rapaport 2004; Allen and Tildesley 2017). The energy regression loss uses mean squared error between predicted and true potential energies. Unlike previous approaches that only used geometric features, these losses directly link latent space representation to physical reality, ensuring generated structures aren't just locally plausible but also adhere to thermodynamic constraints.

2. **Specialised Hierarchical Latent Structure**: Unlike single-latent-space VAEs used in previous materials research, GLASSVAE divides its latent space into two distinct components: a graph-level variable that captures global composition and energy landscape characteristics, and an edge-level variable that refines local geometric details. This division of labor prevents the latent space from becoming too high-dimensional while still maintaining the necessary expressiveness for both local structure and global energy constraints. See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

GLASSVAE significantly outperforms baseline methods on both structure reconstruction and energy prediction for the CuZr metallic glass system (108 atoms, 50% Cu, 50% Zr):

- **Energy prediction**: Achieved RMSE = 0.32 eV/atom with R² > 0.99 on the test set. This represents a 90% improvement over the next best baseline (GraphVAE without RDF loss, RMSE = 0.37 eV/atom).
- **Graph reconstruction**: Achieved RMSE = 0.025 Å for interatomic distances with R² > 0.99. This is a 70% improvement over the GraphVAE without RDF baseline (RMSE = 0.14 Å).
- **Baselines compared**: Distance VAE (distance matrix input), GNN + MLP (no generation), Single Graph-VAE (single latent space), GraphVAE (no RDF loss).
- **Statistical significance**: The paper demonstrates strong performance (R² > 0.99) but doesn't specify p-values or statistical tests used for significance claims. The improvement in RDF-based models is clearly demonstrated in Table 1.

## Related Work

GLASSVAE positions itself at the intersection of graph neural networks, generative models, and materials science. It builds on previous work using GNNs for materials property prediction (e.g., Wang et al. 2021; Li et al. 2024), but addresses a critical gap: previous generative approaches for glasses focused on macroscopic properties or microstructure morphologies rather than atomic-level configurations. The paper explicitly states that prior work on glass structure generation "have largely focused on predicting macroscopic properties, generating microstructure morphologies, rather than atomic-level configurations." GLASSVAE fills this gap by learning an internal generative mechanism that bridges molecular-graph generative models with physically consistent atomic reconstructions.

## Limitations

The paper acknowledges several important limitations. The model was tested only on the CuZr metallic glass system (50% Cu, 50% Zr), so its applicability to other glass compositions or material types (e.g., oxide glasses) remains unverified. The experiments used a relatively small dataset (50,000 samples from MD trajectories at 700K, 760K, 840K, 920K, 1000K), and the paper doesn't discuss how the model would scale to larger systems with more atoms. The authors also don't compare their approach against more recent generative models like diffusion models, which have shown promise in materials science. In my assessment, the biggest limitation is the narrow scope of materials tested; a model that works well for one alloy may not generalise to the broader class of disordered materials.

## Appendix: Worked Example

Let's walk through how GLASSVAE reconstructs a single atomic configuration from its latent representation, using specific values from the paper:

1. **Input**: A molecular dynamics snapshot of a CuZr glass with 108 atoms (54 Cu, 54 Zr) at 700K. The configuration includes atomic positions (R), atom types (t), and total potential energy (E = -4.85 eV/atom).

2. **Graph Construction**: 
   - Nodes: 108 atoms with one-hot type encoding (Cu: [1,0], Zr: [0,1]).
   - Edges: All atom pairs within cutoff distance (3.5 Å), resulting in approximately 1,200 edges.
   - Edge attributes: For each edge, ∆rij (Δx, Δy, Δz) and dij (distance).

3. **Dual-Path Encoding**:
   - Graph-level encoding: After 5 message passing layers, the model outputs µ_graph = [0.2, -0.5, 1.3, ...] (16-dimensional) and log σ²_graph = [-0.1, 0.3, -0.9, ...] (16-dimensional).
   - Edge-level encoding: After 3 residual blocks, the model averages the edge features into s = [0.8, -0.2, 0.5, ...] (16-dimensional).

4. **Latent Space Sampling**:
   - The model samples z = µ_graph + σ_graph ⊙ ϵ, where ϵ ~ N(0, I).
   - For a specific sample, this might yield z = [0.25, -0.4, 1.2, ...] (16-dimensional).

5. **Decoding**:
   - The decoder maps z to predict atom types: 95.2% accuracy (vs. 94.7% for the next best baseline).
   - The decoder predicts edge attributes: RMSE = 0.025 Å (vs. 0.14 Å for GraphVAE without RDF).
   - The property predictor outputs E = -4.83 eV/atom (vs. true value E = -4.85 eV/atom).

6. **Physics-Informed Validation**:
   - The RDF loss compares the predicted distance distribution (from reconstructed edges) with the true distribution.
   - The energy loss compares the predicted E = -4.83 eV/atom with the true E = -4.85 eV/atom.
   - Both losses are small (RDF loss = 0.091, energy regression loss = 0.32 eV/atom), confirming the reconstruction is both geometrically and energetically consistent.

This example shows how the model's hierarchical latent structure enables it to maintain high accuracy in both local geometry (0.025 Å RMSE) and global energy (0.32 eV/atom RMSE) simultaneously.


## References

- **Code:** https://github.com/EricCH97/GlassVAE
- Qiyuan Chen, Ajay Annamareddy, Ying-Fei Li, Dane Morgan, Bu Wang, "Physical-regularized Hierarchical Generative Model for Metallic Glass Structural Generation and Energy Prediction", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36967

#chemistry #atomic structure #graph #VAE 