---
title: "StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2602.08934"
---

## Executive Summary
StealthRL is a reinforcement learning framework that trains paraphrasing policies to evade multiple AI-text detectors simultaneously. It achieves near-zero detection rates (0.024 mean TPR@1%FPR) across three of four detectors with a 97.6% attack success rate, revealing critical robustness gaps in current detection systems. For production engineers, this demonstrates that detectors evaluated only on clean data may fail catastrophically against adaptive adversaries.

## Why This Matters for Practitioners
If you're deploying AI-text detectors for academic integrity or content moderation, this paper shows that relying solely on clean-distribution accuracy is dangerously insufficient. A detector achieving 95% accuracy on clean text may fail completely under adversarial conditions. You should immediately incorporate adversarial robustness testing into your evaluation pipeline using tools like StealthRL before deployment. Prioritise evaluating at the 1% false positive rate operating point used in production systems, not at default thresholds. For critical applications like university plagiarism detection, this means demanding robustness metrics like TPR@1%FPR alongside standard AUROC scores.

## Problem Statement
Current AI-text detectors face a critical evaluation gap: they're tested on clean distributions but must withstand adaptive adversaries who iteratively refine paraphrases. This is analogous to testing a car's safety with a crash test dummy in a controlled lab while ignoring real-world aggressive drivers who deliberately target weak points. Detectors that achieve 95% accuracy on clean text may fail catastrophically when adversaries exploit their decision boundaries, as demonstrated by StealthRL's near-zero detection rates at the critical 1% false positive rate operating point.

## Proposed Approach
StealthRL trains a paraphrase policy against a multi-detector ensemble using Group Relative Policy Optimisation (GRPO) with LoRA adapters on Qwen3-4B-Instruct. The system optimises a composite reward balancing detector evasion with semantic preservation, evaluated at the 1% FPR operating point used in production systems. The architecture consists of four components: a paraphrase policy (Qwen3-4B with LoRA), a multi-detector ensemble (RoBERTa + Fast-DetectGPT), a semantic similarity reward (E5 embeddings), and a training pipeline that evaluates transfer to held-out detectors.

```python
def stealthrl_training():
    # Initialize policy with LoRA adapters
    policy = Qwen3_4B_Instruct(lora_rank=32, lora_alpha=32)
    
    # Training loop over 3 epochs
    for epoch in range(3):
        for batch in training_dataset:
            # Generate multiple candidate paraphrases
            candidates = [policy.generate(text, temp=1.0, top_p=0.9) for _ in range(8)]
            
            # Compute composite reward for each candidate
            for candidate in candidates:
                evasion_reward = 1 - (0.6 * roberta(candidate) + 0.4 * fast_detectgpt(candidate))
                semantic_reward = cosine_similarity(e5_embedding(text), e5_embedding(candidate))
                total_reward = 1.0 * evasion_reward + 0.1 * semantic_reward
                
            # Update policy using GRPO
            policy.update_with_grpo(candidates, total_reward)
    
    return policy
```

## Key Technical Contributions
StealthRL's core innovation lies in its adversarial evaluation protocol for AI-text detectors. Each contribution addresses a specific gap in existing evaluation approaches:

1. **Multi-detector ensemble training with GRPO**: Unlike AuthorMist which trains against a single detector, StealthRL trains against a weighted ensemble (RoBERTa 60%, Fast-DetectGPT 40%) using GRPO. This eliminates the need for a separate value network by using group-level relative rewards, reducing memory requirements while enabling efficient training of evasion policies against diverse detector architectures.

2. **Strict 1% FPR operating point evaluation**: The paper explicitly evaluates at 1% false positive rate (FPR) rather than default thresholds, mirroring production deployment conditions where detectors must avoid falsely flagging human writers. This reveals a significant performance gap between clean-distribution accuracy and real-world robustness.

3. **Comprehensive transfer evaluation**: By testing against two held-out detectors (Binoculars and MAGE) not seen during training, StealthRL demonstrates that evasion strategies transfer across architectures, revealing shared vulnerabilities rather than detector-specific brittleness. This challenges the assumption that ensemble defences improve robustness.

4. **Composite reward design with semantic preservation**: The reward function balances detector evasion (α=1.0) with semantic preservation (β=0.1) using E5 embedding cosine similarity. This prevents degenerate solutions (e.g., empty text) while maintaining 90.1% semantic similarity on average.

## Experimental Results
StealthRL achieves a 0.024 mean TPR@1%FPR across four detectors (RoBERTa, Fast-DetectGPT, Binoculars, MAGE), reducing the mean AUROC from 0.79 (no attack) to 0.43 with a 97.6% attack success rate. Crucially, it achieves near-zero detection on three detectors and shows strong transfer to two held-out detectors (0.002 TPR@1%FPR on Binoculars and 0.089 on MAGE).

The evaluation uses the full filtered MAGE test pool (15,310 human / 14,656 AI samples). Performance metrics are reported across six attack methods (M0-M5) including baselines:
- M0 (No Attack): 0.79 AUROC, 0.09 TPR@1%FPR
- M1 (Simple Paraphrase): 0.74 AUROC, 0.32 TPR@1%FPR
- M2 (StealthRL): 0.43 AUROC, 0.024 TPR@1%FPR
- M3 (Adversarial Paraphrasing): 0.62 AUROC, 0.18 TPR@1%FPR
- M4 (AuthorMist): 0.57 AUROC, 0.23 TPR@1%FPR
- M5 (Homoglyph): 0.45 AUROC, 0.05 TPR@1%FPR

The LLM-based quality evaluation (gpt-5-nano) reveals a tradeoff: StealthRL achieves the strongest evasion (ASR=0.976) but lower quality (linguistic quality=2.51) compared to simpler baselines (M1: ASR=0.716, quality=3.78).

## Related Work
StealthRL builds on AuthorMist (which applies RL for evasion against a single detector) and extends it to multi-detector ensemble training with transfer evaluation. It improves upon DIPPER's simple paraphrasing and SilverSpeak's character-level substitutions by training adaptive policies that balance evasion with semantic preservation. Unlike watermark-based detection approaches (which embed signals during generation), StealthRL focuses on post-hoc evasion of detectors, exposing vulnerabilities in current detection methods rather than proposing new detection techniques.

## Limitations
The evaluation covers only four detectors without watermarks, which embed signals during generation and may be more robust to paraphrasing attacks. The experiments use the MAGE benchmark in English only, with limited coverage of cross-domain or multilingual scenarios. StealthRL achieves lower semantic fidelity (E5 similarity=0.901) compared to simpler paraphrasing baselines (M1: 0.974), indicating a quality-evasion tradeoff that needs addressing. The paper does not explore defensive strategies like adversarial training, which could improve detector resilience against such attacks.

## Appendix: Worked Example
Let's walk through a single text paraphrase using StealthRL with concrete numbers. Start with AI-generated text: "The cat sat on the mat." This text has a detector confidence score of 0.92 for RoBERTa and 0.87 for Fast-DetectGPT.

The StealthRL policy generates eight candidate paraphrases (group size G=8):
1. "A feline sat on the rug." (RoBERTa: 0.12, Fast-DetectGPT: 0.08)
2. "The dog lay on the carpet." (RoBERTa: 0.95, Fast-DetectGPT: 0.89)
3. "A cat is sitting on the mat." (RoBERTa: 0.05, Fast-DetectGPT: 0.03)
4. "The kitten rested on the carpet." (RoBERTa: 0.09, Fast-DetectGPT: 0.06)
5. "A feline was perched on the rug." (RoBERTa: 0.13, Fast-DetectGPT: 0.09)
6. "The cat slept on the mat." (RoBERTa: 0.88, Fast-DetectGPT: 0.82)
7. "A cat was seated on the rug." (RoBERTa: 0.10, Fast-DetectGPT: 0.07)
8. "The feline rested on the mat." (RoBERTa: 0.08, Fast-DetectGPT: 0.05)

Calculate detector evasion rewards:
- Candidate 1: 1 - (0.6*0.12 + 0.4*0.08) = 0.936
- Candidate 3: 1 - (0.6*0.05 + 0.4*0.03) = 0.974
- Candidate 8: 1 - (0.6*0.08 + 0.4*0.05) = 0.956

Semantic similarity (E5 embeddings):
- Candidate 3: 0.93
- Candidate 8: 0.91

Total reward (α=1.0, β=0.1):
- Candidate 3: 0.974 + 0.1*0.93 = 1.067
- Candidate 8: 0.956 + 0.1*0.91 = 1.047

The highest-rated candidate (3) receives the largest reward, and the policy updates to favour this paraphrase. After training, this process produces paraphrases with 0.024 mean TPR@1%FPR across detectors, reducing AUROC from 0.79 to 0.43.

## References

- **Code:** https://github.com/suraj-ranganath/StealthRL.
- Suraj Ranganath, Atharv Ramesh, "StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2602.08934

Tags: #security-and-privacy #ai-detection #reinforcement-learning #adversarial-attacks #text-paraphrasing
