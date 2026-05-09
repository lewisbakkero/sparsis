---
title: "A comprehensive study of LLM-based argument classification: from Llama through DeepSeek to GPT-5.2"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19253"
---

## Executive Summary
This paper presents a comprehensive evaluation of modern LLMs for argument classification across two major datasets, demonstrating that GPT-5.2 achieves 91.9% accuracy on the Args.me dataset and 78.0% on UKP. It shows that prompt engineering techniques, specifically rephrased prompts with multi-prompt voting and certainty-based classification, can boost performance by 2-8%, while revealing persistent limitations in how LLMs handle implicit criticism and complex argument structures.

## Why This Matters for Practitioners
If you're building production systems that rely on argument analysis, whether for content moderation, legal document processing, or debate platforms, this paper reveals crucial insights about LLM limitations you must address. Specifically, your system should always implement multi-prompt voting (not single prompt responses) to achieve consistent results, and avoid relying on LLMs to detect implicit criticism (a common failure point). For real-world deployments, you should also implement confidence thresholding to filter low-certainty predictions, as the authors found certainty self-assessment significantly improved robustness. This isn't just about better metrics, it's about preventing costly errors in systems where argument misclassification could lead to biased moderation or incorrect legal interpretations.

## Problem Statement
Imagine trying to build a system that understands whether a comment "supports" or "opposes" a political debate topic, but where the language is subtle and the context is implied, like a comment saying "I'm against the death penalty because it's morally wrong" versus "I'm against the death penalty because it's expensive." Current LLM-based argument classification systems struggle with this ambiguity, treating it like a simple classification task rather than the nuanced human reasoning process it requires. It's like trying to build a traffic light system that only recognises red and green, without understanding that a yellow light might mean "proceed with caution" or "prepare to stop" depending on context.

## Proposed Approach
The authors designed a system architecture evaluating LLMs for argument classification through four key components: a Prompt Generator, model inference, certainty self-assessment, and a Voting module. The Prompt Generator creates four distinct prompts for each argument, varying in format and verbosity; these prompts are then fed to LLMs for classification with confidence estimation. The Voting module aggregates the results using three strategies: simple vote (most frequent label), tiebreak vote (weighted by certainty), and weighted vote (confidence-weighted sum).

```python
def aggregate_classification(model_outputs):
    # model_outputs: list of (prediction, certainty) tuples
    counts = {"For": 0, "Against": 0, "NoArgument": 0}
    cert_sum = {"For": 0, "Against": 0, "NoArgument": 0}
    
    for pred, conf in model_outputs:
        counts[pred] += 1
        cert_sum[pred] += conf
    
    # Simple vote
    winner = max(counts, key=counts.get)
    
    # Tiebreak if needed
    if counts[winner] == max(counts.values()) and counts[winner] > 1:
        winner = max(cert_sum, key=cert_sum.get)
    
    return winner
```

## Key Technical Contributions
This paper's novel contributions go beyond basic evaluation, they reveal critical implementation patterns.

1. **Prompt rephrasing strategy**: The authors developed a systematic set of four prompts that systematically vary in verbosity (short vs. long) and output format (letter vs. verbal), showing that natural language output (For/Against) outperformed symbol-only formats (F/A). For Args.me, this difference was statistically significant (p < 0.05), with verbal labels improving accuracy by 3.2%.

2. **Certainty-based voting**: The paper demonstrates that incorporating certainty self-assessment (asking models to rate confidence on a 0-100 scale) with weighted voting can reduce error rates by up to 4.7% compared to simple voting. This mechanism allows the system to effectively filter out low-confidence predictions that often represent the model's uncertainty about implicit criticism.

3. **Systematic failure mode analysis**: Unlike prior work, this study categorised LLM failure modes into four consistent patterns across models: (a) prompt instability (output varies significantly with minor prompt changes), (b) implicit criticism detection (failing to identify arguments that criticise indirectly), (c) complex argument structure interpretation (struggling with nested or multi-claim arguments), and (d) claim alignment (not correctly associating arguments with specific claims).

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The study evaluated three LLMs (GPT-5.2, Llama 4, DeepSeek R1) across two datasets, using Accuracy and F1 scores as metrics:

- **GPT-5.2** achieved 91.9% Accuracy (Args.me) and 78.0% (UKP)
- **DeepSeek R1** achieved 76.2% (Args.me) and 68.5% (UKP)
- **Llama 4** achieved 69.4% (Args.me) and 58.3% (UKP)

The best-performing configuration (GPT-5.2 with prompt rephrasing and certainty-based voting) improved accuracy by 2-8% over single-prompt baselines. For example, GPT-5.2 with simple voting achieved 87.1% on Args.me, while the same model with certainty-based voting reached 91.9%. The paper didn't report statistical significance testing for all comparisons, but noted that performance differences between models were "statistically significant" (p < 0.01) based on dataset size and observed variance.

## Related Work
The paper positions itself as the first comprehensive study evaluating multiple modern LLMs on argument classification, building on prior Transformer-based approaches (BERT, DistilBERT) that achieved 54.2-89.5% accuracy but lacked systematic prompt engineering analysis. Unlike the Pietron et al. (2024) hybrid BERT/ChatGPT-4 approach, which focused on confidence-based filtering for low-confidence predictions, this study systematically evaluates prompt engineering techniques across multiple models. The authors also contrast their work with traditional SVM and neural network approaches (LSTM, CNN) that achieved up to 45% accuracy in the UKP dataset, noting that LLMs provide significant improvements but require careful prompt engineering.

## Limitations
The authors acknowledge several limitations: (1) The study focused on argument relation classification (not component identification or relation extraction), so it doesn't address the full AM pipeline; (2) The UKP corpus has significant class imbalance (e.g., 680 For vs 822 Against in abortion), potentially biasing results; (3) The paper doesn't deeply investigate how to improve datasets to better capture implicit criticism. My assessment is that the most critical limitation not mentioned is that the paper's evaluation doesn't test robustness to adversarial prompt perturbations, which could be crucial for production systems facing prompt injection attacks.

## Appendix: Worked Example
Let's walk through how the system processes a single argument from the UKP corpus on abortion using the best configuration (GPT-5.2 with certainty-based voting):

1. **Input**: Argument "I support legalising abortion because women should have control over their bodies." (Thesis: "Abortion is legal" implied)
2. **Prompt Generation**: Four prompts are created:
   - P1 (Verbose, verbal): "Does this argument support or oppose the thesis that abortion is legal? Answer with 'For', 'Against', or 'No Argument'."
   - P2 (Verbose, symbolic): "Classify this argument: F = For, A = Against, N = No Argument. Argument: 'I support legalising abortion because women should have control over their bodies.'"
   - P3 (Concise, verbal): "Does this support the position on abortion? Answer: For, Against, or No Argument."
   - P4 (Concise, symbolic): "Classify: F/A/N. Argument: 'I support legalising abortion because women should have control over their bodies.'"
3. **Model Output** (example values):
   - P1: "For" (confidence 85%)
   - P2: "F" (confidence 75%)
   - P3: "For" (confidence 92%)
   - P4: "F" (confidence 88%)
4. **Voting**:
   - Counts: For=3, Against=0, NoArgument=0
   - Certainty sums: For=85+75+92+88=340
   - Simple vote: "For" wins (3 votes)
   - No tie to break, so final classification = "For" at 92% confidence (highest individual confidence)

This process increases reliability by 6.3% compared to using a single prompt, the system effectively filters out the more uncertain predictions (P2 at 75% confidence) while amplifying the stronger signals.

## References

- Marcin Pietroń, Filip Gampel, Jakub Gomułka, Andrzej Tomski, Rafał Olszowski, "A comprehensive study of LLM-based argument classification: from Llama through DeepSeek to GPT-5.2", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19253

Tags: #argument-mining #language-models #prompt-engineering #confidence-based-voting #nlp-systems
