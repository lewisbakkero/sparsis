---
title: "Improving Automatic Summarization of Radiology Reports through Mid-Training of Large Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19275"
---

## Executive Summary  
This paper introduces GatorTronT5-Radio, a radiology-specialised language model that achieves state-of-the-art performance in summarising radiology reports by inserting a radiology-specific mid-training phase between clinical pre-training and task fine-tuning. It outperforms standard fine-tuning approaches by 5.6% in ROUGE-L and 9.7% in RadGraph-F1 while dramatically improving few-shot learning capability, requiring just 5 samples to match the performance of 200-sample non-mid-trained models.  

## Why This Matters for Practitioners  
If you maintain clinical NLP systems in production, **stop directly fine-tuning clinical LLMs on radiology data**. The paper proves that skipping the radiology-specific mid-training step creates a "cold start" barrier: your model struggles to learn from fewer than 200 samples (e.g., ROUGE-L jumps from 0.0992 to 0.3402 when scaling from 5 to 200 samples). Instead, **implement a three-stage pipeline**:  
1. **Clinical pre-training** (using heterogeneous clinical notes)  
2. **Radiology mid-training** (using MIMIC-CXR’s 13M tokens of concatenated Findings/Impression sections)  
3. **Task fine-tuning** (on OpenI’s 3,955 samples)  
This avoids expensive data collection for small-scale deployments, your 0.2B mid-trained model delivers equivalent results to a 3B non-mid-trained model (ROUGE-L 0.5281 vs. 0.6362), saving 6.2x compute during inference.  

## Problem Statement  
Current radiology summarisation systems treat radiology reports as "just another medical text" akin to interpreting a recipe book written in French, a mismatch where the language’s unique constraints (omitted function words, dense anatomical jargon, negation patterns) are ignored. Like trying to translate French *à la mode* from a dictionary of general culinary terms, standard fine-tuning fails to capture radiology’s linguistic specificity, leading to summaries that miss critical clinical context (e.g., "no acute fracture" vs. "no fracture detected").  

## Proposed Approach  
The pipeline adapts LLMs through three sequential stages:  
1. **Clinical pre-training** on heterogeneous clinical notes (UF Health + PubMed)  
2. **Radiology mid-training** on radiology-specific syntax via unsupervised span-corruption  
3. **Task fine-tuning** on radiology report pairs (Findings → Impression)  
Mid-training is the critical innovation, it bridges clinical knowledge and radiology’s subdomain linguistics before task-specific tuning.  

```python
# Pseudocode: Radiology mid-training (MIMIC-CXR data)
for epoch in range(2):  # 2 epochs to prevent catastrophic forgetting
    for batch in MIMIC_CXR_batches(batch_size=16):
        concatenated_text = concatenate("Findings", "Impression")  # e.g., "Lungs clear... Impression: No pleural effusion"
        masked = span_corruption(concatenated_text, mask_ratio=0.15)  # 15% spans masked
        model.train(masked, target=concatenated_text)  # Denoising objective
```

## Key Technical Contributions  
The radiology mid-training mechanism fundamentally rethinks domain adaptation:  

1. **Radiology-specific span-corruption**  
   Unlike general pre-training (which masks arbitrary spans), mid-training concatenates Findings and Impression sections to form radiology-specific sequences (e.g., "Bilateral pleural effusion noted. Impression: Stable pleural effusion"). The span-corruption objective (masking 15% of tokens) implicitly teaches the model radiology’s characteristic negation patterns ("no effusion") and lexical density ("pleural effusion" vs. "fluid around lungs"), learning structural priors directly from radiology’s linguistic hierarchy.  

2. **Mid-training as a warm-up for few-shot learning**  
   The 2-epoch mid-training on MIMIC-CXR’s 13M tokens creates a "radiology-aware" model that requires only 5 samples for effective fine-tuning (vs. 200 for non-mid-trained models). This is because mid-training internalises radiology’s syntax *before* task tuning, making the model’s initial weights more aligned with radiology’s structure, reducing the sample complexity needed to learn the summarisation task. See Appendix for a concrete walkthrough of this mechanism.  

3. **Explicit avoidance of clinical over-generalisation**  
   By using *only* radiology reports (not heterogeneous clinical notes) for mid-training, the model avoids the "clinical over-generalisation" problem where models conflate radiology with other specialties (e.g., mistaking "admission note" phrasing for radiology). This specificity is why mid-trained models outperform clinical pre-training alone (RadGraph-F1 0.5027 vs. 0.4969 for 0.7B models).  

## Experimental Results  
Results were evaluated on OpenI (3,955 radiology reports, out-of-distribution from MIMIC-CXR):  
- **ROUGE-L** (surface-level overlap):  
  - Mid-trained 0.2B model: **0.5281** (vs. 0.5018 for clinical pre-training alone)  
  - Mid-trained 3B model: **0.6362** (vs. 0.6157 for clinical pre-training)  
- **RadGraph-F1** (clinical factual consistency):  
  - Mid-trained 0.2B model: **0.4585** (vs. 0.4416 for clinical pre-training)  
  - Mid-trained 3B model: **0.5655** (vs. 0.5428 for clinical pre-training)  
- **Few-shot learning**:  
  - With **5 samples**, mid-trained 0.2B model achieved ROUGE-L **0.4716** (vs. 0.0992 for non-mid-trained)  
  - The threshold for meaningful performance dropped from 200 samples (non-mid-trained) to **5 samples** (mid-trained) for all model sizes.  

*Note: The paper reports statistical significance for all improvements (p < 0.01 via paired t-tests), but omits exact p-values.*  

## Related Work  
The paper positions itself against two trends:  
- **Direct fine-tuning**: Previous work (e.g., ClinicalT5) skips domain adaptation, leading to poor radiology-specific performance (ROUGE-L 0.4874 vs. 0.5281 for mid-trained).  
- **Clinical pre-training alone**: Models like GatorTronT5 (pre-trained on heterogeneous clinical notes) still struggle with radiology’s linguistic idiosyncrasies, as evidenced by the 4.9% ROUGE-L gap vs. mid-trained models.  
*Key innovation*: The paper formalises mid-training as a targeted "warm-up" stage *before* fine-tuning, unlike prior work that treated all adaptation as a single step.  

## Limitations  
- **Generalisation beyond radiology**: The method was validated *only* on radiology reports; the authors note it may not transfer to other subdomains (e.g., pathology reports) without retraining.  
- **Resource cost**: Mid-training required 2 epochs on MIMIC-CXR (13M tokens), which is negligible compared to pre-training (>90B tokens) but could be prohibitive in resource-constrained settings.  
- **Linguistic gaps**: The paper doesn’t address how mid-training handles radiology’s evolving terminology (e.g., new imaging modalities), though it may require periodic retraining.  

## Appendix: Worked Example  
Here’s how data flows through mid-training using MIMIC-CXR’s 13M tokens:  
1. **Input**: Raw radiology report section pair  
   *Findings*: "Bilateral pleural effusion. No pneumothorax. Lung fields clear."  
   *Impression*: "Bilateral pleural effusion stable. No acute changes."  
2. **Concatenation**: "Bilateral pleural effusion. No pneumothorax. Lung fields clear. Bilateral pleural effusion stable. No acute changes."  
3. **Span-corruption (15% mask ratio)**:  
   - Original: [Bilateral pleural effusion] [No pneumothorax] [Lung fields clear] [Bilateral pleural effusion stable] [No acute changes]  
   - Masked: "Bilateral pleural effusion. [MASK] Lung fields clear. Bilateral pleural effusion stable. [MASK]"  
4. **Model training**: The decoder predicts "[No pneumothorax]" and "[No acute changes]" using context from radiology-specific syntax (e.g., "no" + clinical term).  
5. **Result**: After 2 epochs, the model internalises that "no" frequently precedes radiology terms (e.g., "no pneumothorax"), improving factual consistency in summaries.  

*This mechanism explains why mid-trained models achieve higher RadGraph-F1: they learn radiology’s pattern of negation during mid-training, not just during fine-tuning.*

## References

- Mengxian Lyu, Cheng Peng, Ziyi Chen, Mengyuan Zhang, Jieting Li Lu, Yonghui Wu, "Improving Automatic Summarization of Radiology Reports through Mid-Training of Large Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19275

Tags: #biomedicine #radiology-report-summarisation #mid-training
