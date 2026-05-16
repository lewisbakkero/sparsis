---
title: "Graph2TS: Structure-Controlled Time Series Generation via Quantile-Graph VAEs"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19970"
---

## Executive Summary
Graph2TS introduces a novel approach to time series generation by separating global structural patterns from local stochastic variations using quantile-based transition graphs. This architecture outperforms diffusion- and GAN-based baselines on multiple datasets including biomedical signals and volatile financial data, offering practitioners a more robust solution for generating realistic time series without amplifying noise or distorting temporal patterns.

## Why This Matters for Practitioners
If you're building systems that rely on synthetic time series data for training or testing (especially in healthcare monitoring or financial forecasting), Graph2TS solves a critical pain point: current generative models either produce noisy signals that distort patterns or oversmooth signals that lose meaningful variations. For instance, when generating synthetic ECG data for model training, traditional approaches like TimeGAN or DiffusionTS produced signals with spectral distortion (as shown in Table 1, DiffusionTS had a PSD L2 distance of 0.198 compared to Graph2TS's 0.0059), making them unsuitable for training robust diagnostic models. Practitioners should adopt Graph2TS for applications requiring high temporal fidelity, particularly on volatile biomedical signals, and avoid diffusion-based approaches when working with heavy-tailed distributions.

## Problem Statement
Current generative models for time series face a fundamental tension, much like trying to photograph a fast-moving crowd while keeping the background landscape clear: GANs capture the chaotic energy of individual movements (local fluctuations) but distort the overall scene (global structure), while diffusion models smooth the crowd into a lifeless blur (oversmoothing), losing critical temporal patterns. This trade-off is especially problematic for volatile signals where direct distribution matching either amplifies noise or erases meaningful temporal patterns.

## Proposed Approach
Graph2TS represents time series structure as a quantile-based transition graph, a compact graph representation capturing global distributional and temporal dependencies while suppressing sample-specific noise. The model conditions generation on this structural graph rather than labels or metadata, enabling cross-modal generation from structural graphs to time series. The architecture consists of three main components: a time series encoder, a graph encoder, and a conditional decoder that uses the graph as a structural condition while a latent variable captures residual stochastic variability.

```python
def graph2ts_generate(structural_graph: Graph, num_samples: int = 1) -> TimeSeries:
    """Generate time series samples conditioned on a quantile graph representation."""
    graph_embedding = graph_encoder(structural_graph)  # Encode graph to vector
    samples = []
    for _ in range(num_samples):
        z = random_normal()  # Sample latent variable for stochasticity
        generated_ts = decoder(graph_embedding, z)  # Conditional generation
        samples.append(generated_ts)
    return samples
```

## Key Technical Contributions
The paper's core innovations lie in how it operationalises the structure-residual decomposition at the implementation level.

Graph2TS fundamentally changes how time series generation is conceptualised by introducing a quantile-based transition graph that:
1. **Encodes global structure through quantile discretisation**: Instead of using raw time series, data is discretised into quantile-based states, creating a sequence of discrete states that marginalises fine-grained amplitude variations while preserving relative ordering. The paper uses Q=3 quantile boundaries (as implied in Figure 1), mapping values to states 0, 1, 2 based on quantile thresholds.
2. **Models the residual stochasticity separately**: The latent variable z is explicitly designed to capture only the residual variations around the structural backbone (E[X|G]=f(G)), with the decoder parameterising x = fθ(g) + rθ(g, z). This ensures the residual term satisfies E[R|G] ≈ 0, preventing the model from learning to compensate for structural information.
3. **Introduces structure-preserving regularisation**: The alignment loss Lalign enforces consistency between graph and time-series embeddings using a bidirectional InfoNCE objective, while Ldist matches order statistics of generated and real sequences. This stabilises generation without introducing additional stochasticity.

## Experimental Results
On the CHB-MIT EEG dataset (used for seizure detection training), Graph2TS achieved a Wasserstein distance of 5.07e-6 compared to TimeGAN's 1.29e-5 and DiffusionTS's 1.10e-5 (lower is better), with significantly improved temporal structure (ACF MAE 0.039 vs. 0.085 for TimeGAN). For ECG signals (where diffusion models typically fail), Graph2TS maintained non-zero coverage (0.895) while DiffusionTS achieved zero coverage (0.000) across all metrics. The ablation study (Table 2) confirmed both structural conditioning and stochastic residual modelling are essential, removing the graph increased Wasserstein distance by 250% (0.049 vs. 0.014) and removing stochasticity increased prototype error by 23% (5.26 vs. 4.27).

## Related Work
Graph2TS builds on recent work in structure-aware time series generation but addresses critical gaps in existing approaches. Unlike conditional GANs that use labels or handcrafted features, Graph2TS conditions directly on structural information. It extends graph-based representations (Rozanec et al., 2025) by using quantile graphs not for analysis but as a conditioning signal within a generative model. The paper explicitly shows that diffusion models fail on heavy-tailed signals (like ECG), while their approach maintains performance by focusing on global structural relations rather than raw amplitude density.

## Limitations
The authors acknowledge the method's limitations: it's designed for univariate time series, and while the paper evaluates on four datasets, it doesn't test the approach on multivariate time series with complex cross-variable dependencies. The quantile graph representation assumes a first-order Markov process, which might not capture higher-order temporal dependencies in some signals. The paper doesn't address computational efficiency for real-time applications, though the architecture uses standard MLP encoders that could be optimised for production use.

## Appendix: Worked Example
Let's walk through the quantile graph construction for a 100-step ECG signal with weak periodicity. The authors estimate global quantile boundaries B={b0,b1,b2} from training data, finding b0=0mV, b1=1.2mV, b2=2.5mV (estimated, paper doesn't specify exact values). Each signal value is mapped to a discrete state: xt=2.3mV → st=2 (since 2.3 ∈ [1.2, 2.5)). This produces a state sequence [2,1,0,2,2,1,0,...] for the 100-step signal. The transition matrix P is calculated based on this sequence, with P[2,2]=0.33, P[2,1]=0.67, P[1,0]=0.5, etc. This quantile graph (P) becomes the structural condition for generating new ECG signals. The model uses this graph to anchor the structural backbone (f(G)) while the latent variable z introduces controlled stochastic variation, producing signals that maintain the global ECG pattern with realistic local variability (see Figure 3 for visual evidence).

## References

- Shaoshuai Du, Joze M. Rozanec, Andy Pimentel, Ana-Lucia Varbanescu, "Graph2TS: Structure-Controlled Time Series Generation via Quantile-Graph VAEs", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19970

Tags: #time-series #biomedicine #graph-vaes #structural-decomposition #stochastic-residuals
