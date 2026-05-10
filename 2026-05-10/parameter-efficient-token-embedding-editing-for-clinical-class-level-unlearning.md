---
title: "Parameter-Efficient Token Embedding Editing for Clinical Class-Level Unlearning"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19302"
---

## Executive Summary

STEU (Sparse Token Embedding Unlearning) enables targeted removal of specific clinical predictions from language models with minimal parameter modifications. It achieves near-complete forgetting of target disease classes while preserving performance on remaining tasks by modifying only 0.19% of model parameters, eliminating the need for costly full retraining. For healthcare AI engineers, this means complying with privacy regulations without disrupting production systems or incurring significant computational overhead.

## Why This Matters for Practitioners

If you're maintaining clinical language models in production (like those predicting disease diagnoses from EHRs), you'll face requests to remove specific disease predictions due to regulatory changes or data quality issues. Current approaches require modifying up to 19.6% of model parameters through encoder updates, which is computationally expensive and difficult to audit in shared deployment environments. STEU provides a practical solution: it suppresses target-class predictions with minimal parameter changes (0.19% of model size), allowing you to comply with deletion requests while maintaining model utility. For a production system with a 108-million-parameter model, this means only 201,000 parameters need modification per deletion request, far more efficient than retraining the entire model.

## Problem Statement

Today's clinical language models face a paradox: they're trained on sensitive health data but must later comply with privacy regulations requiring removal of specific information. Current unlearning approaches are like trying to erase a specific sentence from a novel by rewriting the entire book, modifying large portions of the model to achieve what could be accomplished by simply changing a few words. As the paper states, "existing unlearning approaches usually intervene at the encoder level, modifying a large portion of model parameters" when a more targeted approach could suffice.

## Proposed Approach

STEU identifies tokens strongly associated with a target disease class using pointwise mutual information (PMI), then updates only those token embeddings alongside a small classifier head while keeping all encoder layers frozen. This creates a highly localized intervention with minimal parameter changes.

```python
def steu_unlearning(model, forget_class, dataset, k=256):
    # Step 1: Token selection using PMI
    tokens = select_tokens_with_pmi(model, forget_class, dataset, k)
    
    # Step 2: Freeze all encoder layers
    freeze_encoder(model)
    
    # Step 3: Prepare constrained update surface
    trainable_params = get_trainable_embeddings(tokens) + get_classifier_head()
    
    # Step 4: Train with specified loss functions
    for epoch in range(5):
        forget_batch, retain_batch = get_batches(dataset, forget_class)
        loss = compute_loss(forget_batch, retain_batch, forget_class, model)
        optimise(trainable_params, loss)
    
    return model
```

## Key Technical Contributions

The paper's main innovation lies in how it identifies and manipulates the specific components of a model that contain the target class signal.

1. **PMI-driven token selection**: STEU uses a frequency-weighted pointwise mutual information score to identify tokens strongly associated with the target class:  
   `score(t) = log2(P(t|forget)/P(t|all)) × log(1 + nf(t))`  
   This method deterministically selects tokens carrying class-specific information (e.g., "myocardial infarction" would score high for the heart disease class) while downweighting rare tokens. Crucially, this selection process allows STEU to identify the minimal set of tokens needed for behavioral unlearning, rather than modifying the entire embedding layer.

2. **Combined embedding and classifier head updates**: Unlike embedding-only approaches that leave residual prediction behaviour (forget F1 = 0.276), STEU updates both the selected token embeddings and the classifier head. This dual approach allows the model to suppress target-class predictions without degrading performance on other classes. As the paper shows, "embedding-only updates leave substantial residual forget-class behaviour, whereas adding the head reduces forget F1 from 0.276 to 0.0004 with only a very small increase in parameter count."

3. **Parameter-constrained unlearning**: The method achieves its efficiency by freezing all encoder layers and restricting updates to a tiny fraction of the model. For BioClinicalBERT (108.3 million parameters), STEU updates just 201,000 parameters (0.19%), compared to 21.3 million parameters (19.6%) for encoder-level baselines. This constraint makes the intervention highly auditable and suitable for production environments with parameter-change constraints.

## Experimental Results

STEU was evaluated across multiple clinical datasets (MIMIC-IV, MIMIC-III, and eICU) using three transformer backbones (Bio ClinicalBERT, BERT-base, and DistilBERT). The results consistently demonstrate that STEU suppresses target-class predictions while preserving performance on remaining classes with minimal parameter changes:

- **Forget effectiveness**: STEU achieves near-complete forgetting (forget F1 = 0.0004 on MIMIC-IV with BioClinicalBERT), compared to 0.0000 for encoder-level methods (which also achieved complete forgetting).
- **Retained utility**: STEU maintains higher retained utility than all baselines (retain avg F1 = 0.4766 on MIMIC-IV with BioClinicalBERT), while encoder-level methods produced lower utility (e.g., 0.4912 for direct suppression).
- **Parameter efficiency**: STEU modifies only 0.19% of model parameters (201K) versus 19.6% (21.3M) for encoder-level baselines, representing a two-order-of-magnitude improvement in parameter efficiency.

The paper specifically reports that in the primary MIMIC-IV setting, STEU achieves "near-complete forgetting (forget F1 = 0.0004) while maintaining competitive retained utility (retain avg F1 = 0.4766) after modifying only 0.19% of model parameters."

## Related Work

STEU positions itself as a middle ground between two existing unlearning approaches: head-only interventions (limited in ability to reshape internal representations) and encoder-level updates (requiring large parameter modifications). The paper explicitly compares against gradient ascent (modifying 19.6% of parameters), direct suppression (19.6% of parameters), and influence-weighted unlearning (19.6% of parameters), all of which modify large portions of the model.

The authors acknowledge that certified data removal methods provide theoretical guarantees but often operate at the decision-boundary level rather than intervening in the representation layers. STEU instead targets the embedding layer, where lexical signals first enter the model, enabling localized behavioral unlearning.

## Limitations

The paper explicitly acknowledges several limitations of their approach:
- They focus on behavioral class-level unlearning, not certified sample-level removal: "This does not constitute certified sample-level removal or a formal guarantee of zero influence from specific training examples."
- Influence-based methods are computationally expensive for transformer checkpoints and sensitive to Hessian approximations.
- Their experiments currently report single-seed runs with a limited set of baselines, excluding approaches like partitioned retraining (SISA).
- PMI serves as a simple, deterministic token-selection heuristic, but future work should compare it with gradient-based attribution methods.
- The method is designed for classification tasks and does not directly extend to generative settings where embedding perturbations may degrade fluency.

## Appendix: Worked Example

Let's walk through a concrete example of STEU in action for the target class "myocardial infarction" using BioClinicalBERT on MIMIC-IV.

1. **Token selection process**: The paper uses PMI to identify tokens strongly associated with myocardial infarction. For example:
   - "myocardial infarction" scores high (PMI = 3.2)
   - "acute myocardial infarction" scores high (PMI = 2.9)
   - "cardiac catheterization" scores moderately high (PMI = 2.1)
   - "heart attack" scores moderately high (PMI = 1.8)
   - Common clinical terms like "patient" or "discharge" score low (PMI < 0.5)

2. **Selected tokens**: Using top-k=256, STEU selects the 256 tokens with highest PMI scores. For myocardial infarction, this includes 128 specific clinical terminology terms (like "myocardial", "infarction", and related phrases) and 128 common terms that co-occur with the condition.

3. **Constrained updates**: The embedding matrix for BioClinicalBERT has 108.3 million parameters. STEU targets only the 256 token embedding rows (approximately 256 × 768 = 196,608 parameters) plus the classifier head (4,608 parameters), totaling 201,216 parameters (0.19% of the model).

4. **Training objective**: The loss function combines:
   - `Lforget = BCE(zθ′(Bf), y(0))` (forcing target class probability to zero for forget-class inputs)
   - `Lutility = 1/(C-1) Σ BCE(σ(zθ′,c(Br)), σ(zθ0,c(Br)))` (preserving performance on non-target classes)

5. **Resulting behaviour**: After training, the model no longer predicts "myocardial infarction" for clinical notes containing phrases like "acute myocardial infarction" or "cardiac catheterization", while maintaining performance on predicting other conditions like "pneumonia" or "diabetes."

## References

- Iyad Ait Hou, Shrenik Borad, Harsh Sharma, Pooja Srinivasan, Rebecca Hwa, Aya Zirikly, "Parameter-Efficient Token Embedding Editing for Clinical Class-Level Unlearning", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19302

Tags: #biomedicine #clinical-ai #unlearning #parameter-efficient #healthcare-privacy
