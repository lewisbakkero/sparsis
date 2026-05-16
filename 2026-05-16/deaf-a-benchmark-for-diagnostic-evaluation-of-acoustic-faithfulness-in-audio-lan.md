---
title: "DEAF: A Benchmark for Diagnostic Evaluation of Acoustic Faithfulness in Audio Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.18048"
---

## Executive Summary
DEAF is a benchmark that rigorously tests whether Audio Multimodal Large Language Models (Audio MLLMs) genuinely process acoustic signals or merely infer from textual content. It reveals a pervasive text dominance pattern across seven evaluated models, where acoustic signals are often overridden by textual cues. This matters for engineers building production audio systems, as it exposes a fundamental gap between benchmark performance and true acoustic understanding.

## Why This Matters for Practitioners
If you're deploying Audio MLLMs in production systems, whether for voice assistants, speech analytics, or accessibility tools, you're likely relying on models that fundamentally misunderstand audio. The paper shows that when there's a conflict between what's said (text) and how it's said (acoustics), models consistently choose the text, not the audio. For example, if a speaker says "I'm happy" with a depressed tone, the model will say "happy" despite the acoustic evidence. This means your system might misinterpret sarcasm, emotional context, or background sounds critical for applications like customer service analytics or safety-critical voice interfaces. You should implement acoustic faithfulness checks in your evaluation pipeline, particularly for edge cases involving emotional prosody or background noise, and consider models with higher ARS scores. If you're building new audio systems, design your evaluation metrics to measure acoustic robustness (ARS), not just standard speech recognition accuracy.

## Problem Statement
Today's audio systems are like a chef who can perfectly transcribe a recipe but has no idea what the ingredients taste like, when the recipe says "add sugar" but the dish is sour, they'll just follow the text. Similarly, current Audio MLLMs can excel at standard speech benchmarks where acoustic and semantic content align (like "sad speaker says sad things"), but they fail to detect when these elements conflict (like a happy speaker saying sad things). This creates a dangerous illusion: high benchmark scores suggest robust audio understanding, but models are actually just performing text-based inference. The problem is that this text dominance makes models unreliable in real-world scenarios where audio and text diverge.

## Proposed Approach
DEAF introduces a diagnostic framework that disentangles acoustic faithfulness from text-based bias through controlled conflict stimuli. It constructs three types of acoustic-semantic conflicts (emotion, background sounds, speaker identity) across three evaluation levels of increasing textual interference. The framework uses LLM-as-Judge to score open-ended responses, measuring both acoustic sensitivity (whether models notice acoustic changes) and prediction correctness (whether predictions follow acoustic evidence).

```python
def dea_f_eval(model, audio, text, conflict_type, level):
    # Generate matched (acoustic aligns with text) and mismatched (acoustic contradicts text) samples
    matched_audio = generate_matched_audio(audio, text, conflict_type)
    mismatched_audio = generate_mismatched_audio(audio, text, conflict_type)
    
    # Evaluate at different levels of textual interference
    if level == 1:  # Only audio conflict
        response_matched = model.generate(matched_audio, question)
        response_mismatched = model.generate(mismatched_audio, question)
    elif level == 2:  # Neutral audio, misleading prompt
        response_matched = model.generate(matched_audio, "Misleading prompt: " + prompt + question)
        response_mismatched = model.generate(mismatched_audio, "Misleading prompt: " + prompt + question)
    else:  # Level 3: Dual interference
        response_matched = model.generate(matched_audio, "Misleading prompt: " + prompt + question)
        response_mismatched = model.generate(mismatched_audio, "Misleading prompt: " + prompt + question)
    
    # Score using LLM-as-Judge
    score = llm_judge(response_matched, response_mismatched, conflict_type)
    return score
```

## Key Technical Contributions
DEAF's core innovation lies in its ability to precisely measure and disentangle text dominance from acoustic processing through three key contributions:

1. **Multi-dimensional conflict coverage:** Unlike prior work focused solely on emotional prosody, DEAF spans three independent acoustic dimensions (emotion, background sounds, speaker identity), each with 1,248, 1,260, and 248 stimuli respectively. For example, in Speaker Identity-Semantic Conflict (SIC), they synthesize speech with ElevenLabs TTS to create 248 clips where the vocal characteristics (gender, age) contradict the semantic content (e.g., young voice saying "I'm an elderly woman"). This allows comprehensive testing of acoustic understanding beyond emotional cues.

2. **Progressive textual interference framework:** The three-level evaluation design isolates different failure modes: Level 1 exposes semantic content bias (audio conflict only), Level 2 isolates prompt sycophancy (neutral audio with misleading prompt), and Level 3 combines both. This allows precise attribution of errors to either semantic bias (L1 vs L3) or prompt interference (L2 vs L3). For instance, when models show higher ARS at L2 than L1 (e.g., Qwen2-Audio ESC: 14.7% → 34.0%), it indicates they resist misleading prompts when audio itself is unambiguous.

3. **Diagnostic metrics for acoustic faithfulness:** They introduce ARS (Acoustic Robustness Score), which combines Accuracy (proportion of mismatched samples answered correctly) and ASS (Acoustic Sensitivity Score, fraction of samples where responses differ between matched/mismatched audio). ARS = 2·Acc·ASS/(Acc+ASS), ensuring both sensitivity to acoustic variation and prediction correctness. For example, a model with high ASS (50%) but low Acc (10%) gets low ARS (17.4%), revealing its acoustic perception isn't translating to accurate decisions.

See Appendix for a step-by-step worked example of ARS calculation with actual numbers from the paper.

## Experimental Results
DEAF evaluated seven Audio MLLMs across three conflict types and three evaluation levels. The key findings:

- **ARS degradation pattern:** ARS drops dramatically as textual interference increases. For ESC (emotion conflict), ARS falls below 7% for all models at Level 3, while SIC (speaker identity) retains moderate robustness (34-44% ARS at L3). Qwen3-Omni achieves the highest ARS across all models (45.0% average), while GPT-4o-Audio shows near-zero ARS (1.1% average).

- **Model performance:**
  - Qwen3-Omni: 45.0% ARS average
  - Gemini-2.5 Flash: 46.8% ARS average
  - Audio Flamingo 3: 34.3% ARS average
  - GPT-4o-Audio: 2.7% ARS average
  - Gemini-3 Flash: 39.8% ARS average
  - Qwen2-Audio: 34.3% ARS average
  - SALMONN: 31.1% ARS average

- **Environment discrimination:** Qwen3-Omni shows the strongest environmental discrimination (EDI = 10.0 at L1, 12.2 at L3), while GPT-4o-Audio shows near-zero discrimination (EDI ≤ 1.1).

- **Semantic explicitness effect:** Only Audio Flamingo 3 and SALMONN show statistically significant sensitivity to explicit vs. implicit semantic cues (p < 0.01), with most model-task pairs unaffected (p > 0.05). This confirms that text dominance is primarily driven by textual interference level rather than semantic explicitness.

## Related Work
DEAF builds on previous work like LIS- T EN (Chen et al., 2025) and EMIS (Corrêa et al., 2025), which focused on emotional prosody conflicts. However, DEAF advances by covering three acoustic dimensions (not just emotion), implementing a progressive textual interference framework, and introducing new diagnostic metrics. It also extends the vision-language research by Frank et al. (2021) and Wang et al. (2026), which demonstrated text dominance in vision-language models, to the audio modality. DEAF is the first unified benchmark to systematically diagnose acoustic faithfulness across multiple dimensions and levels of textual interference.

## Limitations
The authors acknowledge several limitations: DEAF covers only three acoustic dimensions (emotion, background sound, speaker identity), leaving out temporal reasoning, multi-speaker interaction, and complex acoustic scenes. Most stimuli are generated through TTS and controlled audio synthesis, potentially not capturing real-world speech variability. The evaluation relies on LLM-as-Judge for scoring open-ended responses, which may introduce bias without human verification. The study evaluates only seven models in a zero-shot setting, and they don't include human performance baselines to calibrate model failures. This means DEAF might not fully capture the failure modes of models in real-world scenarios with more complex audio.

## Appendix: Worked Example
Let's walk through how ARS is calculated for Qwen3-Omni on the Speaker Identity-Semantic Conflict (SIC) task at Level 1. The paper reports 41.1% Accuracy (Acc) and 61.2% Acoustic Sensitivity Score (ASS) for this condition.

1. **Accuracy (Acc):** 41.1% means that for 41.1% of mismatched samples, the model's response aligned with the acoustic ground truth. With 104 mismatched samples in SIC, this means 42 correct responses (41.1% × 104 = 42.7).

2. **Acoustic Sensitivity Score (ASS):** 61.2% means that for 61.2% of samples, the model's response differed between matched and mismatched audio conditions. With 104 samples, this means 64 samples where responses changed (61.2% × 104 = 63.6).

3. **ARS calculation:** Using the formula ARS = 2·Acc·ASS/(Acc+ASS):
   - ARS = 2 × 0.411 × 0.612 / (0.411 + 0.612)
   - ARS = 2 × 0.2517 / 1.023
   - ARS = 0.5034 / 1.023
   - ARS = 0.492 or 49.2% (the paper reports 49.9% ARS for Qwen3-Omni at L1 SIC)

This calculation shows that while Qwen3-Omni has strong acoustic sensitivity (61.2% of samples showed different responses for matched/mismatched audio), its prediction accuracy (41.1%) is lower, resulting in a moderate ARS of 49.2%. This indicates the model can detect acoustic changes but doesn't consistently use this information to make accurate predictions.

## References

- Jiaqi Xiong, Yunjia Qi, Qi Cao, Yu Zheng, Yutong Zhang, Ziteng Wang, Ruofan Liao, Weisheng Xu, Sichen Liu, "DEAF: A Benchmark for Diagnostic Evaluation of Acoustic Faithfulness in Audio Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.18048

Tags: #audio-processing #multimodal-ai #model-evaluation #acoustic-robustness #text-dominance
