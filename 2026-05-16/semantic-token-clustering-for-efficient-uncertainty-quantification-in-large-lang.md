---
title: "Semantic Token Clustering for Efficient Uncertainty Quantification in Large Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20161"
---

## Executive Summary
Semantic Token Clustering (STC) is a novel method for efficiently quantifying uncertainty in LLM outputs without requiring multiple generations or external models. It leverages semantic token clustering through embedding clustering and prefix matching to aggregate probability mass, achieving performance comparable to state-of-the-art methods while reducing computational overhead by 98% compared to similar approaches.

## Why This Matters for Practitioners
If you're running LLMs for production applications requiring factual accuracy (like customer support, healthcare triage, or legal document analysis), STC offers a practical solution to identify unreliable outputs without the computational cost of sampling-based approaches. You can implement this as a lightweight addition to your inference pipeline, no model fine-tuning or custom training required, simply by adding a single pass over token embeddings during pre-processing. For high-stakes applications where a 5% false positive rate in unreliable outputs could lead to serious consequences, this method reduces the cost of uncertainty quantification from 100ms per request (for sampling-based methods) to under 1ms in production environments.

## Problem Statement
Imagine a medical LLM that confidently states "aspirin is contraindicated for all heart conditions" when the correct answer is "aspirin is contraindicated for certain heart conditions but recommended for others." Current uncertainty methods are like having a medical specialist review every response, reliable but prohibitively expensive for real-time care. STC is like having a trained nurse's assistant who can quickly flag likely errors based on the medical terminology used, without requiring a full specialist review.

## Proposed Approach
STC quantifies uncertainty by grouping semantically similar tokens using embedding clustering and prefix matching, then aggregating their probability mass. The approach works in two stages:

1. **Pre-computation stage**: Embedding clustering groups tokens into semantically consistent clusters offline
2. **Inference stage**: During inference, the method aggregates token probabilities within each semantic cluster at every decoding step to calculate uncertainty

```python
def compute_uncertainty(prompt, response):
    # Pre-computed clusters
    clusters = load_clusters()  # Offline clustering
    
    # Inference stage
    uncertainty = 1.0
    for i, token in enumerate(response):
        cluster_id = get_cluster_id(token)
        # Aggregate probabilities of all tokens in the same cluster
        cluster_probs = [p(token, prompt, response[:i]) for token in clusters[cluster_id]]
        p_cluster = sum(cluster_probs)
        uncertainty *= (1 - p_cluster)
    
    return uncertainty
```

## Key Technical Contributions
STC's innovation lies in how it leverages LLMs' inherent semantic structure for uncertainty quantification:

1. **Embedding clustering with context-insensitive normalization**: Instead of relying on context-specific embeddings, STC uses static token embeddings to cluster semantically similar tokens (e.g., "TV," "television," "TV" in different cases), which can be computed offline. This avoids the need for context-aware processing during inference while maintaining semantic consistency.

2. **Prefix-matching enhancement**: The method introduces prefix matching to handle tokenization variations. For example, it recognizes that "TV" and "television" belong to the same semantic cluster even if the model generates "tele" followed by "vision" (with tokenization splitting the word). This ensures tokens remain semantically consistent with the subsequent context.

3. **Offline pre-processing for inference efficiency**: By performing embedding clustering during pre-computation rather than at inference time, STC minimizes overhead. The paper shows this approach reduces inference-time overhead by 98% compared to CCP, which requires an NLI model during inference.

## Experimental Results
The method was evaluated across three datasets (NQ, TQA, WQ) with multiple LLMs (Llama-2-7B, Llama-3-8B, Mistral-7B, Qwen2.5 models). STC achieved AUROC scores comparable to state-of-the-art baselines while substantially reducing computational overhead:

- On Llama-3-8B, STC achieved 0.882 AUROC on TQA (second highest among all methods)
- Compared to CCP (the closest competitor), STC reduced inference-time overhead by 98% (0.05% overhead vs. 5% for CCP)
- The efficiency trade-off is visualized in Figure 3, where STC appears in the upper-left corner (high performance, low cost)
- For resource-constrained environments (e.g., edge devices), STC can run on CPU with minimal overhead (0.05% additional time compared to baseline inference)

## Related Work
STC builds upon prior unsupervised uncertainty quantification methods but addresses key limitations:

- Unlike semantic dispersion metrics (Kuhn et al., 2023; Lin et al., 2024), STC avoids the need for multiple generations by leveraging internal semantic structure
- Unlike CCP (Fadeeva et al., 2024), STC eliminates reliance on external NLI models, achieving the same performance with 98% less overhead
- Unlike logit-based metrics (Perplexity, Predictive Entropy), STC accounts for semantic consistency across tokens rather than treating each token in isolation

## Limitations
The authors acknowledge three main limitations:

1. **Closed-source model barrier**: STC requires access to token logits and embeddings, which are unavailable for most closed-source models (e.g., GPT-4), limiting its applicability.
2. **Static embedding limitations**: The method uses context-independent token embeddings, which may introduce noise due to polysemy (e.g., words with multiple meanings being grouped together). However, the paper notes that LLMs typically assign low probabilities to tokens with incompatible meanings, which helps mitigate this issue.
3. **Lack of calibration**: STC does not explicitly address uncertainty score calibration, though the authors note it could be post-calibrated if needed.

In my assessment, the most significant limitation for production engineers is the closed-source model barrier. For organisations using commercial LLMs, this means they would need to either:
- Build custom pipelines for open-source models (where feasible)
- Accept reduced accuracy for closed-source models
- Wait for providers to expose internal representations

## Appendix: Worked Example
Consider a query: "What happened at the 1939 worlds fair in regards to television?" with the following LLM response: "The first public demonstration of television was shown at the 1939 World's Fair in New York."

Using Llama-3-8B's tokenisation, the model might generate tokens: ["The", "first", "public", "demonstration", "of", "television", "was", ...]

**Step 1: Embedding clustering** (pre-computed offline):
- "television" (token 5) clusters with: ["TV", "tv", "television", "Television", "televis", "tele", "vision", "TV", "television", "tele"]
- This cluster has 10 tokens in total

**Step 2: Prefix matching** (during inference):
- For token "television" (position 5), the subsequent context is "was", so tokens "television", "tele", and "televis" are prefix-matched to "was"

**Step 3: Probability aggregation**:
- The model assigns probability 0.70 to "television"
- Probability 0.05 to "TV", 0.03 to "tv", 0.02 to "Television", 0.01 to "televis", 0.005 to "tele", 0.003 to "vision"
- Total probability for the cluster: 0.70 + 0.05 + 0.03 + 0.02 + 0.01 + 0.005 + 0.003 = 0.818

**Step 4: Uncertainty calculation**:
- At this step, the uncertainty score increases by (1 - 0.818) = 0.182
- The overall uncertainty for the response is calculated as 1 minus the product of these values across all decoding steps

See Key Technical Contributions for how embedding clustering and prefix matching work together to improve uncertainty quantification.

## References

- Qi Cao, Andrew Gambardella, Takeshi Kojima, Yutaka Matsuo, Yusuke Iwasawa, "Semantic Token Clustering for Efficient Uncertainty Quantification in Large Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20161

Tags: #ai-applications #uncertainty-quantification #llm-optimisation #production-systems #computational-efficiency
