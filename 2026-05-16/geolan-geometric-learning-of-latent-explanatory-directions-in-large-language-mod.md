---
title: "GeoLAN: Geometric Learning of Latent Explanatory Directions in Large Language Models"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19460"
---

## Executive Summary
GeoLAN introduces a geometric framework for training large language models that treats token representations as trajectories in latent space, applying stickiness conditions inspired by the Kakeya Conjecture. By enforcing geometric constraints through differentiable regularisers (KT-CW and KT-Attn), it improves interpretability and fairness without sacrificing performance in mid-sized models.

## Why This Matters for Practitioners
If you're deploying LLMs in high-stakes applications like legal or medical domains, GeoLAN offers a way to enhance model transparency without significant performance loss. For mid-sized models (Gemma-3-4B, Llama-3-8B), it's possible to maintain or slightly improve accuracy while increasing interpretability metrics like PCA probe efficiency by dCohen = 0.97. Practitioners should consider incorporating geometric regularisers during training, especially when interpretability is a priority. However, for small models (Gemma-3-1B), avoid using GeoLAN as it degrades performance.

## Problem Statement
Imagine trying to understand how a complex machine works by only looking at the finished product without seeing the intricate gears and levers inside. Current LLMs operate as black boxes where the relationship between inputs and outputs is obscured, making it difficult to diagnose errors or understand biases. The paper uses a unique analogy: standard Transformer embedding spaces are like a narrow cone (anisotropic), where most of the variance is concentrated in a few directions, effectively collapsing the representation space. This geometric collapse limits the model's capacity and worsens entanglement, forcing distinct semantic concepts to share a limited subspace.

## Proposed Approach
GeoLAN rethinks interpretability as a geometric constraint to be enforced during training, rather than discovered post hoc. It views token representations as continuous geometric trajectories rather than static points, applying mathematical concepts from the Kakeya Conjecture to encourage "stickiness" - preventing feature clustering. The framework introduces two differentiable regularisers (KT-CW and KT-Attn) that penalize representation collapse and attention clustering.

```python
def train_geo_lan(model, dataset, epochs):
    lamda1, lamda2 = 0, 0
    for epoch in range(epochs):
        for batch in dataset:
            # Standard language modelling loss
            loss_ce = cross_entropy_loss(model, batch)
            
            # Geometric regularisers (with annealing)
            loss_kt_cw = kt_cw_loss(model, batch)
            loss_kt_attn = kt_attn_loss(model, batch)
            
            # Total loss with annealing schedule
            total_loss = loss_ce + lamda1 * loss_kt_cw + lamda2 * loss_kt_attn
            
            # Backpropagate and update
            total_loss.backward()
            optimizer.step()
            
            # Anneal regularisation weights over first 500 steps
            lamda1 = min(0.001, lamda1 * 1.05)
            lamda2 = min(0.01, lamda2 * 1.05)
```

## Key Technical Contributions
GeoLAN's contributions go beyond simply adding regularisers to training. The authors developed a complete geometric theory for internal explainability and its practical implementation.

1. **Geometric formulation of token trajectories:** The paper formalizes token representations as differentiable curves (γi: [0, 1] → Rd) where γi(tl) ≈ z(l)i. This perspective allows applying geometric measure theory to the latent space. The authors define token tubes as the δ-tubes around these trajectories, which represent semantic footprints. This geometric framework enables precise mathematical statements about representation collapse, unlike previous approaches that used vague analogies like "entanglement."

2. **Semantic Wolff axioms for representation fields:** The paper introduces two mathematical axioms that formalize what "good" representation geometry should look like:
   - Axiom 1 (Semantic Collapse Constant, CA): Limits the number of token tubes passing through any convex region, preventing concentration in narrow cones.
   - Axiom 2 (Attention Interaction Constant, CB): Ensures attention heads distribute focus across diverse semantic regions rather than collapsing onto a single attention sink.
   These axioms provide a mathematical foundation for what "interpretable" means in the latent space, rather than relying on post-hoc analysis.

3. **KT-CW and KT-Attn losses as differentiable surrogates:** The authors developed differentiable loss functions that approximate the combinatorial Wolff axioms. KT-CW uses random projections (64 probes per batch) to estimate isotropy (Varb,i(⟨ẑb,i, u⟩) ≈ 1/d), while KT-Attn uses spectral entropy to maintain attention diversity. These losses are implemented efficiently for large models, with KT-CW loss penalizing deviations from expected variance (1/d) and KT-Attn loss penalizing low spectral entropy in attention matrices.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper evaluates GeoLAN on Gemma-3 (1B, 4B, 12B) and Llama-3-8B models, fine-tuned on 10 billion tokens from C4 dataset. Key results:

1. **Mid-sized models benefit most:** For Llama-3-8B (Goldilocks zone), GeoLAN reduced cone concentration (dCohen = -1.33, p = 0.027) and increased IsoScore (dCohen = 0.85) while improving PCA probe efficiency (dCohen = 0.97).

2. **Performance retention:** GeoLAN maintained or improved MMLU accuracy for three out of four models:
   - Gemma-3-1B: 0.54% MMLU accuracy (GeoLAN) vs 0.54% (Control)
   - Gemma-3-4B: 0.60% vs 0.59% (0.75% improvement)
   - Llama-3-8B: 0.74% vs 0.73% (0.15% improvement)
   - Gemma-3-12B: 0.45% vs 0.53% (-0.80% decrease)

3. **Bias reduction:** In the largest model (Gemma-3-12B), GeoLAN reduced CrowS-Pairs stereotype rate by 0.0065 (dCohen = -0.38, p = 0.0065).

4. **Scale-dependent effects:** The paper identifies a scale floor where geometric interventions become ineffective for small models (Gemma-3-1B) due to superposition requirements in low-capacity regimes.

## Related Work
GeoLAN builds on recent work in mechanistic interpretability (SAEs, dictionary learning) but fundamentally shifts the perspective from post-hoc analysis to training-time constraints. It draws inspiration from geometric measure theory (Kakeya Conjecture) and the Wolff axioms, adapting mathematical concepts to the domain of language models. Unlike previous work that focused on analysing existing models, GeoLAN provides a method to actively shape the latent space during training.

## Limitations
1. The paper doesn't report on inference-time performance or latency, so practitioners should measure this before deployment.
2. The method is ineffective for small models (Gemma-3-1B), suggesting a scale-dependent trade-off that may limit applicability for resource-constrained environments.
3. The authors don't explore how GeoLAN would interact with other interpretability techniques like sparse autoencoders.
4. The paper notes "rotational symmetry" issues where geometric stability varies across different training seeds, indicating that grain alignment isn't consistent across runs.

## Appendix: Worked Example
Let's walk through a concrete example of how GeoLAN works with a Gemma-3-4B model on a simple input sequence.

Imagine we have a short input sequence: "The cat sat on the mat." This is a 6-token sequence (S = (the, cat, sat, on, the, mat)).

1. **Token trajectories:** At layer 2, the model computes hidden representations for each token. For simplicity, let's assume the model has 4 layers (L = 4) and 128-dimensional embeddings (d = 128).

2. **KT-CW loss application:** To compute the KT-CW loss at layer 2, the model samples 64 random unit vectors (u1 to u64) from a uniform distribution on the 128-dimensional sphere. For each token (e.g., "cat"), it computes the projections of the token embeddings onto each random vector and estimates the variance of these projections. The loss is minimised when this variance approaches 1/128 ≈ 0.0078.

3. **Example calculation:** For token "cat" at layer 2, the embedding vector is (0.2, -0.1, 0.3, ..., 0.1) (128 dimensions). For a random vector u = (0.05, -0.02, 0.03, ..., 0.01), the projection is ⟨ẑb,i, u⟩ = 0.01. The paper reports that for a well-behaved model, the variance of these projections should be approximately 1/128 ≈ 0.0078. The KT-CW loss penalizes when the actual variance deviates from this value.

4. **KT-Attn loss application:** For the attention mechanism at layer 2, head 3, the attention matrix A(2,3) has dimensions (6, 6) for this 6-token sequence. The singular values are computed, normalized to form a probability distribution, and the spectral entropy is calculated. The KT-Attn loss is minimised when the spectral entropy is high (close to log(6) ≈ 1.79).

5. **Example calculation:** For attention matrix A(2,3), the singular values are [0.5, 0.2, 0.15, 0.05, 0.05, 0.05]. The normalized values are [0.5, 0.2, 0.15, 0.05, 0.05, 0.05]. The spectral entropy is - (0.5*log(0.5) + 0.2*log(0.2) + 0.15*log(0.15) + 0.05*log(0.05) * 3) ≈ 1.41. The maximum possible entropy is log(6) ≈ 1.79. The KT-Attn loss penalizes when the entropy is below the target.

6. **Total loss:** The total loss combines the language modelling loss with the KT-CW and KT-Attn losses. The regularisation weights (λ1 and λ2) are annealed during training, starting from 0 and increasing to 0.001 and 0.01, respectively. This annealing schedule allows the model to warm up its representations before the geometric constraints tighten.

## References

- Tianyu Bell Pan, Damon L. Woodard, "GeoLAN: Geometric Learning of Latent Explanatory Directions in Large Language Models", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19460

Tags: #machine-learning #model-interpretability #geometric-regularisation #model-fairness #large-language-models
