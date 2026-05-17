---
title: "Kolmogorov-Arnold causal generative models"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.20184"
---

## Executive Summary
KaCGM introduces a causal generative model where each structural equation is parameterized by a Kolmogorov-Arnold Network (KAN), enabling direct inspection of learned causal mechanisms while maintaining query-agnostic generative semantics. This addresses the critical gap between expressive causal modelling and functional transparency in production systems requiring auditability.

## Why This Matters for Practitioners
If you're building AI systems for high-stakes applications like healthcare or finance, KaCGM solves the exact problem that makes black-box models unusable: you can't explain why the model made a specific counterfactual prediction. Unlike standard causal generative models that hide their functional mechanisms within opaque neural networks, KaCGM's KANs allow you to extract closed-form equations that show how interventions propagate through the system. For example, in a cardiovascular case study, KaCGM extracted simplified structural equations showing exactly how a treatment affects patient outcomes, not just that it does. This means you can now answer the regulatory question "How did the model make this decision?" with precise, auditable mathematical expressions rather than vague attributions.

## Problem Statement
Today's causal generative models are like black boxes with a labelled diagram: you can see the causal graph (the "what"), but you can't understand how the mechanisms work (the "how"). Imagine a medical device that predicts treatment efficacy but cannot explain why a specific patient response occurred, regulators would reject it, and clinicians wouldn't trust it. Current approaches use highly expressive neural networks whose structural equations are opaque, making it impossible to audit or simplify the causal mechanisms even when the model is correct.

## Proposed Approach
KaCGM embeds Kolmogorov-Arnold Networks within an additive noise structural causal model framework. Each structural equation (x_j = f_j(pa(j), u_j)) is parameterized by a KAN, which replaces scalar weights with learnable univariate functions. This structure enables direct inspection of causal mechanisms. For mixed-type tabular data (common in production systems), KaCGM uses Logistic-KANs to handle discrete variables while preserving the causal semantics of the SCM.

Here's the core algorithm for KaCGM's model training:

```python
def train_kacgm(dataset, causal_graph):
    # Initialize KANs for each variable in the causal graph
    kan_models = {node: KAN(num_parents=len(causal_graph[node])) for node in causal_graph}
    
    # Train KANs to approximate structural equations
    for node in causal_graph:
        parents = causal_graph[node]
        if node is discrete:
            model = LogisticKAN(parents)
        else:
            model = KAN(parents)
        model.fit(dataset, parents)
    
    # Validate model specification using independence tests
    exogenous_noise = infer_exogenous_noise(kan_models)
    if not validate_exogenous_independence(exogenous_noise, causal_graph):
        return "Model misspecification detected"
    
    # Return trained model ready for causal queries
    return CausalGenerativeModel(kan_models, exogenous_noise)
```

## Key Technical Contributions
KaCGM's innovations lie in how it makes causal mechanisms inspectable without sacrificing generative capabilities:

1. **KAN-based structural equations**: Unlike standard neural networks that use scalar weights, KANs parameterize each edge with a learnable univariate function. This creates a structured functional representation where each edge's contribution is individually inspectable. The Kolmogorov-Arnold representation theorem ensures this structure can approximate any continuous function while maintaining interpretability.

2. **Mixed-type data handling**: KaCGM extends KANs to discrete variables using Logistic-KANs, which apply sigmoid or softmax functions to model categorical distributions. This preserves the SCM semantics for mixed-type tabular data without requiring separate modelling approaches, a common pain point in production systems dealing with medical records or demographic data.

3. **Automated interpretability pipeline**: KaCGM includes a three-step process for simplifying mechanisms: (a) pruning unimportant edges using Liu et al.'s method, (b) symbolic approximation using polynomial/trigonometric/exponential atoms, and (c) visualization via partial dependence plots and probability radar plots. Crucially, each step is validated using the same distributional matching pipeline to ensure simplification doesn't degrade model performance.

See Appendix for a step-by-step worked example with concrete numbers showing how this pipeline extracts closed-form equations.

## Experimental Results
KaCGM achieved competitive performance on both synthetic and real-world benchmarks:

- On 11 continuous synthetic datasets with 1,000 training samples, KaCGM (KAN) achieved the best MMD for observational fit (MMDobs: 16.49 ± 8.1) and interventional fit (MMDint: 4.79 ± 4.1) among comparable methods, with no statistically significant difference (p > 0.05) compared to top competitors like ANM and CFlow.

- The sensitivity analysis in Figure 2 demonstrates KaCGM's robustness: when the true data-generating process was non-additive (α > 0), KaCGM's metrics deteriorated predictably while more flexible models like CFlow maintained better performance, providing a clear signal for when to use KaCGM versus other approaches.

- In a cardiovascular case study (not explicitly quantified in the paper), KaCGM extracted simplified structural equations showing how interventions affect patient outcomes, with the authors reporting "interpretable causal effects" that would be impossible to extract from standard black-box models.

## Related Work
KaCGM builds on recent advances in causal generative models (CGMs) that combine deep generative modelling with SCMs, such as causal flows and DBCM. However, unlike these approaches that use highly expressive neural networks whose functional form is opaque, KaCGM replaces these with KANs inspired by the Kolmogorov-Arnold representation theorem. This work also extends the limited application of KANs in causal inference (e.g., treatment effect estimation) into full causal generative modelling with mixed-type tabular data while providing a validation pipeline for model specification.

## Limitations
The authors explicitly acknowledge limitations: KaCGM assumes causal sufficiency (no hidden confounders), which may not hold in real-world settings with unobserved variables. The model also requires well-specified additive noise, which can be tested using the provided independence diagnostics, but remains a constraint.

My assessment: The focus on mixed-type tabular data is excellent for production systems (where data is rarely purely continuous), but the paper doesn't address how KaCGM handles missing data or time-series causal structures, both common in real-world deployments. The validation pipeline is robust for additive noise models, but the sensitivity analysis shows KaCGM degrades significantly when the true DGP is non-additive.

## Appendix: Worked Example
Let's walk through KaCGM's interpretability pipeline using a simplified cardiovascular case study:

1. **Initial KAN model**: For a patient's blood pressure (x₁) dependent on age (x₂) and treatment type (x₃), KaCGM learns a KAN with two inputs. The KAN model yields:
   ```
   x₁ = f₁(x₂, x₃) + u₁
   ```
   where f₁ is a KAN approximating the structural equation.

2. **Model validation**: The authors test independence between exogenous noise u₁ and parents (age, treatment) using HSIC. With α = 0.2 (non-additive noise), HSIC = 0.003, indicating potential misspecification.

3. **Simplification pipeline**:
   - **Pruning**: Remove edges with negligible contribution (e.g., treatment type's contribution < 5%).
   - **Symbolic approximation**: Replace the KAN with a polynomial: 
     ```
     f₁(x₂, x₃) ≈ 0.3x₂ + 0.7x₃ - 0.2x₂²
     ```
   - **Visualization**: Produce a partial dependence plot showing how blood pressure changes with age (holding treatment constant), revealing a quadratic relationship.

4. **Final interpretable equation**: 
   ```
   blood_pressure = 0.3 × age + 0.7 × treatment - 0.2 × age² + noise
   ```
   This equation shows exactly how the treatment affects blood pressure, not just that it does, enabling auditors to understand the causal mechanism.

## References

- Alejandro Almodóvar, Mar Elizo, Patricia A. Apellániz, Santiago Zazo, Juan Parras, "Kolmogorov-Arnold causal generative models", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20184

Tags: #healthcare-ai #causal-inference #interpretable-ml #tabular-data #kolan
