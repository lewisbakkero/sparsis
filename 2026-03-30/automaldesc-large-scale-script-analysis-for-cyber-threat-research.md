---
title: "AutoMalDesc: Large-Scale Script Analysis for Cyber Threat Research"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36959"
---

## Executive Summary

Security operations are currently limited by a "Context Gap": tools can flag a script as malicious (detection), but explaining *how* it harms a system requires hours of manual reverse-engineering. 

**AutoMalDesc** closes this gap. By using an iterative self-training pipeline, researchers moved from a tiny 900-sample seed to a 157,000-sample dataset, boosting Batch script analysis accuracy from a coin-flip (**52.7%**) to a production-ready **82.4%**.

## Why This Matters for Practitioners

If your SOC is drowning in PowerShell or Batch alerts, you are likely paying experts to perform repetitive "description" tasks. 
- **Operational Scale:** Process 100k+ scripts without hiring 100 more analysts.
- **Explainable AI:** Move beyond "Malicious/Benign" flags to human-readable summaries (e.g., "This script disables Windows Defender via registry keys").
- **Cold-Start Solution:** You don't need a massive labeled dataset to start; the model "bootstraps" its own intelligence from a small expert seed.

## Problem Statement

Current static analysis is binary. An engineer sees an alert for `malicious_script.ps1` but still has to open the file to see if it’s a credential stealer or a simple downloader. This manual step is the #1 cause of high Mean Time to Remediate (MTTR).

## Proposed Approach

AutoMalDesc uses a self-paced learning pipeline that starts with a small seed dataset of 900 expert-curated examples, then iteratively generates and validates synthetic data to improve output quality. The core mechanism involves:

1. **Initial training**: Fine-tuning Llama-3.3-70B-Instruct (using LoRA with rank 8) on the seed dataset
2. **Pseudo-label generation**: Using the trained model to generate explanations and metadata (maliciousness labels and language identification) for 157,126 unlabeled scripts
3. **Rigorous quality filtering**:
   - Remove empty responses (9,095 samples)
   - Remove truncated summaries (17 samples) and incomplete JSONs (7,825 samples)
   - Apply consistency checks across three temperature settings (0.4, 0.6, 0.8)
   - Use Phi-3.5-Mini (99.5% test accuracy) to filter samples where summaries contradict malicious labels (219 samples)
   - Apply 90% confidence threshold on logit probabilities

This pipeline is applied iteratively: V1 uses seed data, V2 combines seed data with V1-generated data.

```python
def iterative_dataset_extension(seed_data, unlabeled_data, iterations=2):
    # Initial model trained on seed data
    model = fine_tune(Llama3_3_70B_Instruct, seed_data, lora_rank=8, alpha=16)
    
    for iteration in range(iterations):
        # Generate pseudo-labels for unlabeled data
        predictions = model.generate(unlabeled_data, temperature=[0.4, 0.6, 0.8])
        
        # Filter quality
        filtered = filter_pseudo_labels(predictions)
        
        # Add filtered data to training set
        seed_data += filtered
        
        # Retrain model with expanded dataset
        model = fine_tune(model, seed_data, lora_rank=8*(iteration+1), alpha=16*(iteration+1))
    
    return model, seed_data

def filter_pseudo_labels(predictions):
    """Filters predictions based on multiple quality checks"""
    # Remove empty responses
    predictions = [p for p in predictions if p is not None and len(p) > 0]
    
    # Apply consistency checks across temperature settings
    consistent_indices = []
    for i, _ in enumerate(predictions[0]):
        labels = [pred[i]['malicious'] for pred in predictions]
        if all(l == labels[0] for l in labels):
            consistent_indices.append(i)
    
    # Apply summary-label verification using Phi-3.5-Mini
    filtered_indices = []
    phi_model = load_phi_3_5_mini()
    for i in consistent_indices:
        summary = predictions[0][i]['summary']
        malicious = predictions[0][i]['malicious']
        if phi_model.verify(summary, malicious):
            filtered_indices.append(i)
    
    # Apply confidence threshold
    final_indices = []
    for i in filtered_indices:
        if predictions[0][i]['confidence'] > 0.9:
            final_indices.append(i)
    
    return [predictions[0][i] for i in final_indices]
```

## Key Technical Contributions

AutoMalDesc introduces a novel iterative methodology for enhancing cybersecurity script analysis through LLM-generated annotations, with specific technical innovations that differentiate it from prior approaches:

1. **Self-improving pipeline for malware analysis**: The framework uses an iterative self-paced learning process where an initial model trained on 900 seed examples generates synthetic data for a larger unlabeled corpus, then validates through multiple quality checks (consistency across temperature settings, summary-label verification, confidence thresholds). Unlike prior work like Self-Instruct or STaR, which focused on general language tasks, this pipeline is specifically designed for the cybersecurity domain where expert-labeled data is scarce and expensive.

2. **Scalable dataset generation without manual annotation**: By leveraging synthetic data generation and validation cycles, the authors eliminate the need for extensive manual data annotation. The approach starts with a small seed (900 samples) but scales to over 100K examples (101,277 refined samples), reducing dependency on costly sandbox analysis and expert annotations. This is different from prior approaches like Lu et al.'s MalS dataset, which required light human refinement, as AutoMalDesc requires minimal human input beyond the initial seed.

3. **Comprehensive evaluation framework**: The authors developed a multi-faceted evaluation approach combining quantitative metrics (maliciousness label and language accuracy) with qualitative assessment from both human experts and LLM-based judges. This goes beyond traditional metrics to assess technical precision and linguistic coherence of generated summaries, which is critical for malware analysis where explanations must be both accurate and understandable.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results

The authors evaluated their approach on three tasks using a test set of 3,636 samples (700 per language), with 6% of samples containing obfuscation techniques:

- **Language detection accuracy**:
  - Base model: 93.7% average accuracy (89.2% Bash, 92.3% Batch, 98.4% JavaScript, 89.9% PowerShell, 100% Python)
  - V1 model: 96.8% average accuracy (98.0% Bash, 94.2% Batch, 99.5% JavaScript, 92.7% PowerShell, 99.8% Python)
  - V2 model: 97.1% average accuracy (99.3% Bash, 94.9% Batch, 99.8% JavaScript, 91.8% PowerShell, 100% Python)

- **Malware detection accuracy**:
  - Base model: 83.1% average accuracy (92.4% Bash, 52.7% Batch, 90.8% JavaScript, 94.2% PowerShell, 89.8% Python)
  - V1 model: 89.3% average accuracy (93.2% Bash, 77.9% Batch, 90.6% JavaScript, 95.3% PowerShell, 91.8% Python)
  - V2 model: 91.5% average accuracy (96.3% Bash, 82.4% Batch, 92.2% JavaScript, 95.3% PowerShell, 92.6% Python)

- **Batch malware detection improvement**: 29.9% absolute improvement from Base to V2 (52.7% to 82.4%), confirmed statistically significant by McNemar's test (p < 10^-5), with V2 correcting 110 false positives and 49 false negatives from V1's predictions.

The authors demonstrated statistically significant improvements between iterations (p < 10^-5) across 3,600 diverse samples in five scripting languages. For script comprehension, human evaluators preferred V1 50.46% vs. V2 49.54% (54 vs. 53 selections out of 107 total, with 9 ties), with 27 total hallucinations across both models (V1: 13, V2: 14).

## Related Work

AutoMalDesc builds upon two key areas of prior work: self-improving language models and static security analysis of scripts. For self-improving approaches, it adapts techniques like Self-Instruct (Wang et al. 2023b) and STaR (Zelikman et al. 2022) to the cybersecurity domain, where data scarcity is a critical challenge. Unlike previous work, AutoMalDesc focuses specifically on generating natural language explanations of malware behavior rather than just classification.

For static security analysis, the authors extend prior work on semi-supervised malware classification (e.g., Alam et al. 2025, Feng et al. 2025b) and natural language explanations (e.g., Fujii and Yamagishi 2024, Lu et al. 2025). What distinguishes AutoMalDesc is its iterative self-training approach for generating detailed security explanations across five scripting languages, demonstrating consistent quality improvements with minimal human input, unlike prior work that required human refinement or focused on binary classification.

## Limitations

The paper acknowledges three key limitations:

1. **Context length constraints**: The model cannot analyze longer scripts effectively, particularly affecting PowerShell and JavaScript samples with extensive code blocks.

2. **Language detection ceiling**: Language detection accuracy showed minimal improvement across iterations (97.14% in V2), suggesting a performance ceiling. Across the test, 27 hallucinations were noted. In a high-stakes environment, this means the tool is an *assistant* for analysts, not a total replacement.

3. **Trade-off between readability and precision**: V2 models tend to be more "chatty" and readable, which experts found slightly less precise than the "robotic" V1 outputs.

Additionally, the paper doesn't address how the framework would handle emerging scripting languages not included in the initial five (Bash, Batch, JavaScript, PowerShell, Python), which could limit its adaptability to new attack vectors.

## Appendix: Worked Example

Let's walk through the iteration process for a single batch script using concrete numbers from the paper:

1. **Initial seed dataset**: 900 samples (807 benign/malicious across languages, as shown in Table 1)
2. **Pseudo-label generation**: The V1 model (fine-tuned on seed data) generates explanations for 157,126 scripts
3. **Quality filtering**:
   - Remove empty responses (9,095 samples) → 148,031 remaining
   - Remove truncated summaries (17) and incomplete JSONs (7,825) → 140,189 remaining
   - Consistency checks across three temperatures (0.4, 0.6, 0.8) eliminate samples with inconsistent malicious labels → 120,300 remaining (14.28% reduction)
   - Phi-3.5-Mini filters 219 samples with summary-label discrepancies → 120,081 remaining
   - 90% confidence threshold removes samples with low logit probabilities → 101,277 valid samples (as shown in Table 1)

For a specific batch script sample:
- Base model (pretrained Llama-3.3-70B-Instruct) predicts "benign" with 78% confidence
- V1 model (first iteration) generates a summary: "This script modifies registry settings related to security policies and creates scheduled tasks for persistence" with 92% confidence
- V2 model (second iteration) refines it to: "This script configures Windows Defender exclusions and creates persistent scheduled tasks, indicating potential malware behavior" with 95% confidence

The iterative process improves the model's ability to correctly identify batch scripts as malicious (from 52.7% to 82.4% accuracy), with V2 correcting 110 false positives and 49 false negatives from V1's predictions.


## References

- **Code:** https://github.com/CrowdStrike/automaldesc
- Alexandru-Mihai Apostu, Andrei Preda, Alexandra Daniela Damir, Diana Bolocan, "AutoMalDesc: Large-Scale Script Analysis for Cyber Threat Research", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36959
