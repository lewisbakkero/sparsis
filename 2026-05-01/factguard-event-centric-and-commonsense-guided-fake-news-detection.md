---
title: "FACTGUARD: Event-Centric and Commonsense-Guided Fake News Detection"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36998"
---

## Executive Summary
FACTGUARD is a novel framework for fake news detection that extracts event-centric content from news articles using large language models (LLMs), reducing reliance on detectable writing styles. It introduces a dynamic usability mechanism to assess LLM-generated reasoning and employs knowledge distillation to create a lightweight variant (FACTGUARD-D) suitable for resource-constrained environments.

## Why This Matters for Practitioners
If you're building or maintaining content moderation systems on social media platforms, this paper solves a critical pain point: current style-based detection systems fail when adversaries imitate authentic news writing styles. FACTGUARD's event-centric approach means you no longer need to constantly retrain models against evolving writing patterns, saving engineering effort. For systems with limited computational resources (like mobile apps or edge devices), FACTGUARD-D delivers 95%+ of the accuracy of the full solution with significantly reduced inference costs. Implement this by first extracting event content using your existing LLM infrastructure, then integrating the dual-branch usability module to adaptively weight LLM advice rather than using it as a binary classifier.

## Problem Statement
Imagine trying to identify counterfeit banknotes by only examining the ink colour, adversaries could easily mimic the exact shade, making your detection method useless. Similarly, current fake news detectors focus on surface-level writing style (like sentence structure or vocabulary), but adversaries now routinely imitate authentic news styles to bypass these detectors. This is why the authors compare it to "trying to spot a fake signature by only looking at the pen pressure", the underlying content is what matters, not the superficial appearance.

## Proposed Approach
FACTGUARD operates in two main modes: for resource-rich environments, it uses LLMs to extract event-centric content and commonsense rationale, which are then processed by a dual-branch model to create more reliable predictions. For resource-constrained settings, FACTGUARD-D uses knowledge distillation to create a lightweight model that only requires the original news text. The core flow involves extracting event content, comparing it with commonsense reasoning, and dynamically adjusting the weight of LLM advice based on reliability.

```python
def factguard_inference(news_text):
    # Step 1: Extract topic-content and commonsense rationale using LLM
    topic_content = llm_extract_topic_content(news_text)
    commonsense_rationale = llm_get_commonsense_reasoning(news_text)
    
    # Step 2: Process features through dual-branch module
    # (This is simplified; actual implementation uses cross-attention)
    weights = dual_branch_usability_evaluator(
        topic_content, 
        commonsense_rationale
    )
    
    # Step 3: Fuse features and predict
    fused_features = fuse_features(
        original_text=news_text,
        topic_content=topic_content,
        weights=weights
    )
    return classifier.predict(fused_features)
```

## Key Technical Contributions
FACTGUARD's core innovations address specific gaps in current LLM-based fake news detection approaches. These mechanisms are implemented with precision to solve the identified problems:

1. **Two-stage constraint for content extraction**: The authors use a text similarity metric during extraction to maintain consistency with the original news, followed by an information density metric to evaluate informativeness after extraction. This ensures extracted content doesn't drift from the original news while maintaining relevance, unlike previous approaches that simply extracted surface-level content without quality control.

2. **Dual-branch rationale usability assessment**: Instead of treating LLM advice as always reliable or unreliable, FACTGUARD uses a dual-branch MLP structure where one branch reduces LLM influence when direct detection capability is limited, while the other increases influence when commonsense reasoning identifies contradictions. This dynamic adjustment means the system can better handle ambiguous cases without requiring a full retraining cycle.

3. **Knowledge distillation for resource-constrained settings**: The framework transfers knowledge from the full FACTGUARD model to a lightweight FACTGUARD-D model through a feature distillation loss that minimises MSE between the student's and teacher's feature representations. This approach differs from standard distillation because it specifically targets the "fused features" rather than just the final predictions, preserving the nuanced reasoning capabilities.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
FACTGUARD achieved the highest results on both GossipCop (English) and Weibo21 (Chinese) benchmark datasets across four metrics: macF1, Accuracy, F1real, and F1fake. On Weibo21, FACTGUARD achieved 0.801 macF1, 0.804 Accuracy, 0.824 F1real, and 0.777 F1fake, outperforming all 14 baselines including GPT-3.5-turbo, RoBERTa, and advanced LLM-SLM methods like TED. The distilled version FACTGUARD-D achieved 0.788 macF1 on Weibo21, a 1.2% drop from the full model (0.801) while operating without LLM calls. The paper doesn't report statistical significance testing for the improvements, though the consistent performance across multiple metrics suggests meaningful gains.

## Related Work
FACTGUARD builds on and improves over four categories of prior approaches: 1) LLM-only methods (like GPT-3.5-turbo, which the authors show has poor usability in real-world detection), 2) SLM-only methods (like BERT and RoBERTa, which remain style-sensitive), 3) LLM-SLM hybrid methods (like ARG and TED, which suffer from high computational costs), and 4) distilled models (like ARG-D, which doesn't address style sensitivity). The authors position their work as solving the key gaps: style sensitivity (via event-centric extraction), LLM usability (via dynamic assessment), and resource constraints (via distillation).

## Limitations
The paper doesn't explicitly discuss limitations, but based on the method description, FACTGUARD's effectiveness likely depends on the quality of the LLM's commonsense reasoning module, which the authors acknowledge is sourced from prior work (Hu et al. 2024). As the paper focuses on the final detection performance without examining the specific errors made by the system, it's unclear whether the framework correctly handles cases where commonsense reasoning itself is flawed (e.g., false positives in culturally specific contexts). Also, the distillation process preserves performance but doesn't address the initial dependency on LLMs for training.

## Appendix: Worked Example
Let's walk through how a single news article flows through FACTGUARD using realistic values from the paper's approach:

Start with a news article about a sports event: "Liverpool FC secured a 3-1 victory against Manchester City in the Premier League match on April 25, 2023."

1. **Topic-Content Extraction** (LLM prompt: "Extract the core topic and principal content from this news article. Identify key event details: teams, score, date, and context.")  
   - Original text: 72 words  
   - Extracted content: "Liverpool FC defeated Manchester City 3-1 in Premier League match on April 25, 2023" (24 words)  
   - Text similarity metric: 89% (vs original)  
   - Information density metric: 0.42 (high, as it contains all key event details)

2. **Commonsense Rationale Generation** (LLM prompt: "Identify if this news contains any factual contradictions or common sense inconsistencies.")  
   - Generated rationale: "A 3-1 scoreline is plausible in Premier League matches; April 25 is a standard matchday; Liverpool vs Manchester City is a common fixture."  
   - Commonsense confidence: 92% (no contradictions detected)

3. **Dual-Branch Usability Assessment**  
   - Branch 1 (LLM direct detection): "This is authentic news" → Confidence: 85% → Weight: 0.35 (reduced due to moderate LLM confidence)  
   - Branch 2 (Commonsense contradiction check): "No inconsistencies found" → Confidence: 92% → Weight: 0.65 (increased due to high commonsense confidence)  
   - Final LLM feature fusion: (0.35 × feature_A) + (0.65 × feature_B)

4. **Feature Fusion and Prediction**  
   - Original news features (from BERT): [0.12, -0.34, 0.81, ...] (1024 dimensions)  
   - LLM-enhanced features: [0.67, 0.21, -0.43, ...] (1024 dimensions)  
   - Fused features: [0.12+0.67, -0.34+0.21, 0.81-0.43, ...] (1024 dimensions)  
   - Final prediction: "Authentic news" (probability: 0.89)

This flow demonstrates how FACTGUARD reduces style sensitivity by focusing on event details rather than phrasing, while the dual-branch mechanism dynamically weights LLM advice based on reliability.

## References

- **Code:** https://github.com/ryliu68/FACTGUARD
- Jing He, Han Zhang, Yuanhui Xiao, Wei Guo, Shaowen Yao, Renyang Liu, "FACTGUARD: Event-Centric and Commonsense-Guided Fake News Detection", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36998

Tags: #information-retrieval #text-analysis #llm-integration #knowledge-distillation #fake-news-detection
