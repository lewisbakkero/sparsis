---
title: "MAPLE: Metadata Augmented Private Language Evolution"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19258"
---

## Executive Summary
MAPLE addresses a critical bottleneck in API-based differentially private synthetic data generation by using metadata to better align initial synthetic data with target domains. It significantly improves privacy-utility trade-offs, accelerates convergence, and reduces API costs compared to prior Private Evolution approaches, making it particularly valuable for organizations processing sensitive data through proprietary LLM APIs.

## Why This Matters for Practitioners
If you're building systems that generate synthetic data for sensitive domains like healthcare or finance using proprietary LLMs such as GPT-4 or Gemini, MAPLE lets you generate higher-quality synthetic data with 70-80% fewer API calls than current methods. For instance, when generating synthetic medical abstracts from bioRxiv data, MAPLE achieves 15-20% higher MAUVE scores with the same privacy budget (ε=2) while reducing the number of API iterations from 9 to 2. This translates directly to lower costs and faster development cycles for compliance engineering teams handling sensitive data. Specifically, if your current data synthesis process uses AugPE with GPT-3.5 (requiring 8-9 API iterations for medical abstracts), switching to MAPLE with Qwen 2.5 7b will reduce your API costs by approximately 77% while improving synthetic data quality.

## Problem Statement
Imagine trying to teach a chef to cook French cuisine using only a generic recipe book - it works fine for basic dishes, but fails completely when attempting to recreate a complex dish like coq au vin. Similarly, current API-based synthetic data generation methods treat all text as homogeneous, leading to poor alignment when the target data comes from highly specialized domains. When private data distribution (e.g., medical abstracts) deviates from foundation model priors (e.g., general web text), the initial synthetic data becomes misaligned, requiring many iterations to converge - like trying to find a specific needle in a haystack by randomly moving the haystack instead of first narrowing the search space.

## Proposed Approach
MAPLE enhances the initialization phase of Private Evolution (PE) by incorporating differentially private metadata into the prompt for the initial text generation phase. It first extracts metadata from the private data in tabular format, then uses a lightweight differentially private metadata generator (AIM) to create DP synthetic metadata. Finally, it composes a prompt for the initial API call containing both the synthetic metadata and in-context examples of (metadata, text) pairs, which leverages the LLM's in-context learning capability. This creates a more aligned starting point for the PE process.

```python
def maple_initialization(private_data, metadata_schema, n_examples=50):
    # Step 1: Extract metadata from private data
    metadata = extract_metadata(private_data, metadata_schema)
    
    # Step 2: Generate differentially private synthetic metadata
    dp_metadata = dp_metadata_generator(metadata)
    
    # Step 3: Select in-context examples
    in_context_examples = select_examples(
        donated_data, 
        metadata, 
        n=n_examples, 
        distance_metric=hamming_distance
    )
    
    # Step 4: Create prompt with metadata and examples
    prompt = f"Generate text based on the following metadata: {dp_metadata}\n\nIn-context examples:\n"
    for metadata, text in in_context_examples:
        prompt += f"Metadata: {metadata}, Text: {text}\n"
    
    # Step 5: Generate initial synthetic data using RANDOM_API
    synthetic_data = random_api_call(prompt)
    
    return synthetic_data
```

## Key Technical Contributions
MAPLE introduces novel methods to overcome the initialization bottleneck in PE frameworks by effectively grounding synthetic data generation in target domains:

1. **Differentially private metadata extraction pipeline**: Instead of assuming metadata exists in the raw data, MAPLE extracts metadata directly from text using a structured tabular schema. The pipeline uses a lightweight DP generator (AIM) requiring only CPU training, making it suitable for resource-constrained environments. This avoids the privacy leakage that would occur if raw metadata were directly used in prompts.

2. **In-context learning with metadata-aware examples**: MAPLE leverages LLMs' in-context learning capability by including both metadata and text examples in the prompt. The authors discovered that using only metadata or only examples provides marginal improvement over vanilla PE, but combining both yields significant gains (a 15-20% MAUVE improvement on bioRxiv). The examples are selected based on metadata similarity (Hamming distance), not text similarity, to better align with the private data distribution.

3. **Metadata richness impacts convergence**: The paper demonstrates that richer metadata (9 attributes vs. 2) leads to faster convergence (2 iterations vs. 4 iterations) while achieving higher quality synthetic data. This insight allows practitioners to optimise the metadata schema for their domain without compromising privacy. For instance, with only 2 metadata attributes, convergence takes 4 iterations instead of 2, but still far fewer than AugPE's 9 iterations.

## Experimental Results
MAPLE consistently outperforms AugPE (the current state-of-the-art PE-based method) across all privacy regimes and evaluation metrics on two challenging domain-specific datasets:

1. **bioRxiv dataset** (29k scientific abstracts, average 300 tokens):
   - MAUVE score: MAPLE achieves 0.72 at ε=2 vs. AugPE's 0.62 (15.4% improvement)
   - Average JSD on metadata: MAPLE achieves 0.23 vs. AugPE's 0.38 (39.5% improvement)
   - NTP accuracy: MAPLE achieves 0.30 vs. AugPE's 0.24 (25% improvement)
   - API iterations: MAPLE converges in 2 iterations vs. AugPE's 9 iterations

2. **OpenReview dataset** (8.4k peer reviews):
   - Downstream classification accuracy (area): MAPLE achieves 0.68 vs. AugPE's 0.62 (9.7% improvement)
   - Downstream classification accuracy (rating): MAPLE achieves 0.71 vs. AugPE's 0.64 (10.9% improvement)
   - API iterations: MAPLE converges in 2 iterations vs. AugPE's 8 iterations

Notably, MAPLE with Qwen 2.5 7b achieves better results than AugPE with GPT-3.5 on the OpenReview dataset, demonstrating that better initialization compensates for smaller model size.

## Related Work
MAPLE extends the Private Evolution (PE) framework by focusing on the critical initialization phase, whereas prior work has focused on refining the iterative evolution process. The paper distinguishes itself from DP-finetuning approaches, which are often infeasible with proprietary APIs, by providing an API-based solution that requires no model fine-tuning. It builds on the metadata approach of Tan et al. (2025) and Hu et al. (2025) but introduces in-context learning with metadata-aware examples as a key innovation. Unlike Sun et al. (2025), who directly provide metadata in prompts without leveraging in-context learning, MAPLE's combination of metadata and in-context examples is essential for high-quality synthetic data.

## Limitations
The authors acknowledge that MAPLE requires a small set of donated examples (50 samples) for in-context learning, which might be unavailable in highly sensitive settings. The paper doesn't evaluate MAPLE on non-textual data modalities like images or audio, though they suggest this is a promising direction. Additionally, while MAPLE improves convergence, it doesn't address the fundamental privacy cost of each API call, though it does reduce the number of calls needed. The paper also doesn't explore how to optimise metadata schema design for specific domains, leaving this as a manual process.

## Appendix: Worked Example
Let's walk through how MAPLE generates synthetic medical abstracts from the bioRxiv dataset (29k scientific abstracts, average length 300 tokens):

1. **Metadata extraction**: For each abstract, MAPLE extracts 9 metadata attributes including "study type" (randomized controlled trial/observational study), "disease" (specific condition), "drug" (treatment), "sample size" (number of participants), "study duration" (months), "primary outcome" (measured metric), "secondary outcome", "funding source", and "publication year".

2. **DP metadata generation**: The private metadata (from 29k abstracts) is processed through the AIM DP generator to create synthetic metadata with ε=0.1 for metadata generation (total privacy budget ε=1.0 allocated as 0.1 to metadata and 0.9 to PE iterations).

3. **In-context example selection**: From a donated dataset (50 abstracts), MAPLE selects 10 examples with the smallest Hamming distance on metadata attributes. For instance, if the private data has a high frequency of "disease": "diabetes", "drug": "metformin", and "study type": "randomized controlled trial", the selected examples will have similar metadata patterns.

4. **Prompt composition**: The initial prompt contains:
   - Metadata: "study type: randomized controlled trial, disease: diabetes, drug: metformin, sample size: 200-300, study duration: 12 months, primary outcome: HbA1c reduction, secondary outcome: cardiovascular events, funding source: NIH"
   - In-context examples:
     "Metadata: study type: randomized controlled trial, disease: diabetes, drug: metformin, sample size: 250, study duration: 12, primary outcome: HbA1c reduction, secondary outcome: cardiovascular events; Text: A randomized controlled trial of metformin in patients with type 2 diabetes showed a mean HbA1c reduction of 1.2% over 12 months."
     "Metadata: study type: observational study, disease: hypertension, drug: lisinopril, sample size: 500, study duration: 24, primary outcome: systolic BP reduction, secondary outcome: renal function; Text: Observational study of lisinopril in hypertension patients found a mean systolic BP reduction of 15 mmHg after 24 months."

5. **Initial generation**: The RANDOM_API call using this prompt produces synthetic abstracts with much higher alignment to the medical domain than vanilla AugPE's random initialization.

6. **Convergence**: Only 2 iterations of PE refinement are needed to achieve high-quality synthetic data, compared to 9 for AugPE. This demonstrates that a good initialization dramatically reduces the number of rounds needed to reach quality synthetic data.

## References

- **Code:** https://github.com/microsoft/DPSDA
- Eli Chien, Yuzheng Hu, Ryan McKenna, Shanshan Wu, Zheng Xu, Peter Kairouz, "MAPLE: Metadata Augmented Private Language Evolution", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19258

Tags: #privacy #differential-privacy #synthetic-data #large-language-models #metadata #api-based
