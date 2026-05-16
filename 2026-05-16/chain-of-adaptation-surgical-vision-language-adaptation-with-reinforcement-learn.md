---
title: "Chain-of-Adaptation: Surgical Vision-Language Adaptation with Reinforcement Learning"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20116"
---

## Executive Summary
Chain-of-Adaptation (CoA) is a reinforcement learning framework that adapts general vision-language models to surgical domains without degrading their core visual-language capabilities. It introduces a four-stage reasoning format that preserves pretrained model priors while enabling domain specialization. This approach achieves 10-27% higher F1-scores on surgical benchmarks compared to standard fine-tuning, with improved robustness and generalisation.

## Why This Matters for Practitioners
If you're building production systems that need to integrate specialized domain knowledge (e.g., medical imaging, industrial inspection, or legal document analysis), this paper directly addresses the common problem of catastrophic forgetting during domain adaptation. Instead of retraining from scratch or using standard fine-tuning that distorts the base model's capabilities, CoA enables you to maintain the model's general multimodal competence while specializing for domain-specific tasks. For instance, in medical imaging systems, this means you can adapt a general-purpose vision-language model to understand surgical terminology without losing its ability to describe non-clinical scenes. Practically, you should replace standard fine-tuning with CoA for domain-specific adaptation, using the four-stage reasoning format to maintain general capabilities, and implement RLVR with GRPO to optimise for task-specific rewards while preserving the base model's behaviour.

## Problem Statement
Imagine trying to adapt a general-language model to understand surgical procedures through standard fine-tuning. The current approach is like teaching a chef who can cook all cuisines to specialise in French cuisine by only giving them French recipes, they'll forget how to cook Italian dishes, and might start making up French terms they've never heard. This is precisely what happens with standard fine-tuning: the model forgets its general capabilities while learning surgical terminology, leading to shorter, less diverse outputs and even hallucinations.

## Proposed Approach
CoA is a two-stage framework that first bootstraps domain knowledge through a cold start phase and then optimises for task performance using reinforcement learning. It uses a structured four-stage reasoning format that separates general visual grounding from domain-specific reasoning, preventing interference between low-level visual semantics and high-level surgical logic. This structured format serves as a regulariser that maintains general visual-linguistic grounding while enabling domain specialisation.

```python
def chain_of_adaptation(image, prompt):
    # Stage 1: General description (maintains base model's capabilities)
    desc = model.generate(image, prompt="Describe the image using only low-level visual attributes")
    
    # Stage 2: Evidence (integrates task info, visual observations, domain knowledge)
    evidence = model.generate(image, prompt=f"Based on the description, list evidence for surgical analysis: {desc}")
    
    # Stage 3: Thought (structured reasoning)
    thought = model.generate(image, prompt=f"Using the evidence, reason step-by-step: {evidence}")
    
    # Stage 4: Answer (final conclusion)
    answer = model.generate(image, prompt=f"Based on the thought process, provide the final answer: {thought}")
    
    return {
        "general description": desc,
        "evidence": evidence,
        "thought": thought,
        "answer": answer
    }
```

## Key Technical Contributions
CoA's core innovation is its four-stage structured reasoning format and how it's integrated with a reinforcement learning training pipeline. Unlike standard fine-tuning or chain-of-thought approaches, CoA maintains the model's general capabilities while adapting to domain-specific tasks.

1. **Structured separation of general and domain-specific reasoning**: The <general description> stage explicitly prevents domain terminology, focusing only on low-level visual attributes. This acts as a regulariser that maintains the base model's visual grounding capabilities, preventing catastrophic forgetting as demonstrated by the 64% reduction in output length observed during standard fine-tuning.

2. **Cold start phase with pseudo-labelled data**: By using Gemini-Flash-2.5 to generate CoA-formatted responses from 10,000 unlabeled surgical images, CoA bootstraps domain knowledge without relying on scarce labelled data. This process establishes a "domain-aware representation prior" that facilitates stable reinforcement learning optimisation.

3. **Composite reward function with format enforcement**: CoA's reward function r(y) = rtask(y) if y conforms to CoA format, else 0 ensures both structural validity and semantic accuracy. This novel mechanism prevents the model from generating malformed outputs while still optimising for domain-specific task performance.

4. **GRPO-based RLVR training pipeline**: The use of Group Relative Policy Optimisation (GRPO) instead of standard supervised fine-tuning enables the model to improve through internal comparison rather than explicit supervision. This preference-based alignment allows the model to learn from itself through iterative answering and feedback, making it particularly effective in data-scarce domains.

## Experimental Results
CoA was evaluated on two standard surgical benchmarks: EndoVis2018 and CholecT50. The results show significant improvements over standard fine-tuning (SFT):

- On EndoVis2018: F1-score improved from 0.657 (SFT) to 0.837 (+27%)
- On CholecT50: F1-score improved from 0.587 (SFT) to 0.644 (+10%)

The paper also shows that CoA maintains general capabilities better than SFT:
- SFT reduced average output length by 64% (from 476 tokens to 171 tokens)
- CoA maintained output length (similar to base model) while improving surgical task performance

Ablation studies confirmed that the structured four-stage format consistently outperformed vanilla CoT-based variants in both in-distribution and out-of-distribution surgical tasks, as well as general-domain VQA benchmarks.

## Related Work
CoA builds on recent advances in RL-based VLM training (e.g., RL-VLM-F [36], Chen et al. [7]), but specifically addresses the unique challenges of surgical adaptation where data are scarce and annotations are limited. While prior work like LLaVA-Surg [17] and GP-VLS [25] used standard fine-tuning on synthetic instruction data, they suffered from limited annotation diversity and overfitting to frequent categories. CoA improves upon these by preserving the model's base capabilities through its structured reasoning format and RL training framework, rather than overwriting them with domain-specific fine-tuning.

## Limitations
The paper doesn't address how CoA would perform on highly specialised surgical procedures requiring domain expertise beyond what's available in common surgical datasets. While the cold start phase used 10,000 unlabeled surgical images, the paper doesn't specify whether these images represented sufficient diversity across surgical specialties. Additionally, the paper doesn't report inference latency or memory requirements for CoA, which would be critical for production deployment.

## Appendix: Worked Example
Let's walk through how CoA processes a single surgical image of a laparoscopic procedure:

1. **Input**: A surgical image from a laparoscopic cholecystectomy video showing instruments interacting with tissue (as shown in Figure 2 of the paper)
   
2. **General description stage**: The model generates a low-level description focusing on visual attributes:
   "This is a close-up, endoscopic view of a surgical procedure involving the dissection and mobilisation of the duodenum. The image shows a portion of the duodenum, which appears as a pale, pinkish, and somewhat translucent structure with visible blood vessels."

3. **Evidence stage**: The model integrates this description with visual observations and domain knowledge:
   "Key elements: Two surgical instruments (graspers at lower left, possibly a clamp at top); Primary target: Pale pink, translucent structure (duodenum); Surrounding tissue: Darker red (possible pancreas or adjacent bowel); Instruments are actively manipulating tissue layers to separate or enter thin layers."

4. **Thought stage**: The model reasons step-by-step:
   "The instruments (graspers and dissecting tools) are manipulating tissue layers. The pale pink structure is likely the duodenum being mobilised. The darker red tissue is likely adjacent structures like the pancreas. The procedure appears to be dissection and mobilisation of the duodenum."

5. **Answer stage**: The model provides the final answer:
   "Dissection of the duodenum"

This structured process maintains the model's ability to describe the image in detail (output length 476 tokens) while correctly identifying the surgical step and terminology.

## References

- **Code:** https://github.com/QwenLM/Qwen3-VL
- Jiajie Li, Chenhui Xu, Meihuan Liu, Jinjun Xiong, "Chain-of-Adaptation: Surgical Vision-Language Adaptation with Reinforcement Learning", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20116

Tags: #biomedicine #vision-language-models #reinforcement-learning #domain-adaptation #structured-reasoning
