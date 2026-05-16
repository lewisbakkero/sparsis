---
title: "The Phish, The Spam, and The Valid: Generating Feature-Rich Emails for Benchmarking LLMs"
venue: "The Spam"
paper_url: "https://arxiv.org/abs/2511.21448"
---

## Executive Summary
PhishFuzzer introduces a metadata-enriched email generation framework that produces 23,100 diverse, structurally consistent email variants across three strict classes (Phishing, Spam, Valid), complete with attacker intent annotations, full URL and attachment metadata. This dataset enables realistic benchmarking of LLMs for email security systems, addressing the critical gap where traditional detection systems fail against LLM-generated phishing content.

## Why This Matters for Practitioners
If you're building email security systems today, this paper reveals two critical engineering decisions. First, your model must incorporate structural metadata (URLs, sender domains, attachment filenames) to maintain accurate phishing detection, Gemini-3.1-Pro's Phishing F1 increased from 0.939 to 0.958 with metadata, representing a 20-30% reduction in classification errors. Second, you must design your system to handle the subjective boundary between spam and valid emails, as both models systematically misclassified spam as valid, with Qwen-2.5-72B's Spam Recall dropping from 25.11% to 19.06% when adding metadata. Your system should use hybrid architectures combining LLM semantic reasoning with heuristic filters rather than relying solely on metadata-enhanced LLMs.

## Problem Statement
Today's email security systems resemble a tollbooth that only checks for specific vehicle types (e.g., "cars with red licence plates") while ignoring that attackers have learned to change their "licence plate colour" using LLMs. Traditional systems fail against LLM-rephrased content because they rely on outdated rule-based heuristics and static features, like keyword matching, without accounting for the linguistic sophistication now achievable with LLMs.

## Proposed Approach
PhishFuzzer operates through four sequential steps: (1) creating a seed dataset from manually curated and public emails, (2) using LLMs to infer attacker intent and populate missing metadata fields, (3) enriching all emails with validated intent and structural metadata, and (4) generating structured variants along entity and length dimensions. Each of the 3,300 seed emails produces six synthetic variants (three length types × two entity types), yielding a final dataset of 23,100 emails with strict three-class labels and complete metadata.

```python
def generate_email_variants(seed_emails, length_types, entity_types):
    variants = []
    for seed in seed_emails:
        for length in length_types:
            for entity in entity_types:
                variant = {
                    "original": seed,
                    "length": length,
                    "entity": entity,
                    "intent": seed.intent,
                    "url": generate_url(seed, entity),
                    "attachment": generate_attachment(seed, entity)
                }
                variants.append(variant)
    return variants
```

## Key Technical Contributions
PhishFuzzer's core innovations address critical gaps in email security research:

1. **Intent-aware metadata generation with structural constraints**: The framework populates missing URL and attachment fields while preserving linguistic coherence and attack intent. For phishing variants, it requires deceptive domains (e.g., "google-secure.com" instead of "google.com") that would bypass SPF checks, while allowing realistic corporate domains for spam and valid emails. This ensures generated metadata remains contextually plausible and security-relevant.

2. **Total Flip Score (TFS@K) metric for systematic failure analysis**: Unlike standard metrics, TFS@K identifies templates where a model consistently misclassifies all six variants (TSR(ti, K) = 0). This metric isolates fundamental model blind spots, like the 43 phishing templates Qwen-2.5-72B misclassified as Valid in its Basic setting, rather than random errors. See Appendix for a step-by-step example of how TFS@K identifies these systematic failures.

3. **Provenance-aware dataset expansion**: The framework generates variants while tracking their origin (private inbox vs. public dataset), enabling analysis of how well models generalise across data sources. This revealed Qwen-2.5-72B degrades significantly on private seeds (56% accuracy vs. 74% on public seeds in Basic setting), while Gemini-3.1-Pro maintains consistent performance.

4. **Three-class evaluation framework**: Unlike previous work with limited class distinctions (e.g., E-PhishLLM's two classes), this dataset enforces strict separation between Phishing, Spam, and Valid emails with attacker intent annotations, enabling more nuanced analysis of model performance.

## Experimental Results
The PhishFuzzer dataset (23,100 emails: 3,300 seeds, 19,800 variants) was used to benchmark Qwen-2.5-72B and Gemini-3.1-Pro across four dimensions:

- **Phishing detection**: Adding metadata improved both models' Phishing F1 scores significantly, Gemini-3.1-Pro from 0.939 to 0.958 (a 2.0% absolute increase), Qwen-2.5-72B from 0.893 to 0.917 (a 2.4% absolute increase). This represents approximately 20-30% fewer classification errors for phishing detection.

- **Spam detection**: Metadata introduced a severe trade-off, both models showed degraded Spam F1 scores (Qwen-2.5-72B dropped from 0.428 to 0.310, Gemini-3.1-Pro from 0.428 to 0.383). Qwen-2.5-72B's Spam Recall fell from 25.11% to 19.06% with metadata, indicating a systematic shift where spam was misclassified as valid.

- **Model reliability**: Gemini-3.1-Pro showed stronger semantic understanding with a higher Confidence Index (62.88% for Full prompts vs. Qwen-2.5-72B's 55.33%), meaning it remained more consistent across variants. However, Qwen-2.5-72B was more confidently wrong, systematically failing on 575 templates compared to Gemini's 491 under Full prompts.

- **Zero-shot generalisation**: Gemini-3.1-Pro maintained consistent performance between private and public seeds (72% vs. 71% accuracy), while Qwen-2.5-72B declined significantly (56% vs. 74%).

## Related Work
PhishFuzzer builds on E-PhishLLM, which generated synthetic phishing and valid emails but lacked structural granularity (full URL strings, attachment names) and intent annotations. Unlike legacy datasets (SpamAssassin, Enron, Nazario), which suffer from outdated content and data quality issues, PhishFuzzer addresses the critical gap where legacy datasets are insufficient proxies for LLM-generated phishing content, as shown by E-PhishLLM's cross-dataset evaluation revealing up to 40 percentage point accuracy drops for classical ML pipelines.

## Limitations
The paper only benchmarks two zero-shot LLMs; future work should compare with fine-tuned models like BERT and traditional machine learning classifiers. The classification of spam remains highly subjective (depending on recipient preferences), which the authors acknowledge as a fundamental challenge. The authors also note that human mislabeling in public datasets, uncovered during systematic failure analysis, requires further investigation. The work is limited to email security and doesn't extend to other communication types like SMS or chat protocols.

## Appendix: Worked Example
Let's walk through generating a single phishing email variant using PhishFuzzer:

Start with a seed phishing email (private inbox) labelled "Follow the link" with a missing URL field. The email requests users to "verify their account" with a link to "https://secure-paypal-login.com" (fabricated but deceptive).

1. **Intent identification**: The framework confirms intent is "Follow the link" based on prior benchmarking (Gemini-2.5-Flash achieved 97.98% strict accuracy in intent labelling).
2. **Entity type selection**: The variant uses a globally recognised entity (e.g., "PayPal" instead of a fabricated brand).
3. **Length type selection**: The variant is Medium (10-16 sentences).
4. **URL generation**: The framework generates a deceptive URL structure while maintaining context ("https://login.paypal-secure.com" instead of "https://www.paypal.com").
5. **Attachment generation**: For phishing variants, the attachment is fabricated but plausible (e.g., "payment_confirmation.pdf" with deceptive content).

The resulting synthetic email maintains the original intent ("Follow the link") while altering sentence structure to create a novel variant. This process produces 6 variants per seed email across two entity types (globally recognised vs. fabricated) and three length types (Short, Medium, Long).

## References

- **Code:** https://github.com/DataPhish/PhishFuzzer
- Rebeka Toth, Tamas Bisztray, Nils Gruschka, "The Phish, The Spam, and The Valid: Generating Feature-Rich Emails for Benchmarking LLMs", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2511.21448

Tags: #email-security #llm-benchmarking #metadata-enriched-data #phishing-detection #spam-classification
