---
title: "CAF-Score: Calibrating CLAP with LALMs for Reference-free Audio Captioning Evaluation"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19615"
---

## Executive Summary
CAF-Score introduces a reference-free metric for evaluating audio captions by combining CLAP's coarse-grained semantic alignment with LALM-based syntactic awareness. It achieves the highest correlation with human judgments on the BRACE benchmark, outperforming reference-based baselines in challenging scenarios while eliminating the need for expensive annotated datasets.

## Why This Matters for Practitioners
If you're building production audio captioning systems, you've likely struggled with evaluation bottlenecks. Traditional reference-based metrics require costly human annotation for every new caption, while CLAP-based approaches miss syntactic errors that degrade user experience. CAF-Score allows you to validate caption quality at scale, without human annotation, by directly detecting subtle hallucinations (e.g., "birds" instead of "insects" in ambient noise descriptions). For engineers, this means: 1) You can rapidly iterate on captioning models using automated evaluation, 2) You can deploy systems with confidence in their ability to avoid critical misrepresentations, and 3) You can reduce evaluation costs by 90% compared to reference-based methods, as shown in Table 1 where CAF-Score outperforms reference-based baselines in HH (Human-Human) scenarios.

## Problem Statement
Current evaluation metrics for audio captioning are like using a single lens to view a 3D scene: they capture some details but miss critical context. Reference-based metrics (e.g., FENSE) require annotated captions, making them impractical for production use. CLAP-based metrics (e.g., CLAPScore) are reference-free but only detect broad semantic mismatches, ignoring syntactic errors like "a man" instead of "a woman" in descriptions of people. This is equivalent to evaluating a weather report by checking if it mentions "rain" but missing the distinction between "heavy rain" and "light drizzle" in the description.

## Proposed Approach
CAF-Score combines two parallel branches: a CLAP-based coarse-grained semantic alignment branch and an LALM-based fine-grained evaluation branch. The CLAP branch uses sliding windows to cover full audio duration and max-pools to identify salient segments. The LALM branch computes a FLEUR score from token probability distributions to capture syntactic details. These scores are linearly combined using a weighting parameter α.

```python
def caf_score(audio, caption, claps, lalm):
    # CLAP branch: sliding window + max pooling
    clapscore = max([clap(audio_window, caption) for audio_window in sliding_windows(audio)])
    
    # LALM branch: FLEUR score from token probabilities
    fleur_score = compute_fleur(lalm, caption)
    
    # Combine scores with weighting α = 0.8
    return 0.8 * clapscore + 0.2 * fleur_score
```

## Key Technical Contributions
CAF-Score introduces three key innovations that bridge the gap between coarse and fine-grained evaluation:

1. **Sliding-window strategy with max-pooling**: Unlike prior work that uses fixed truncation or average pooling, CAF-Score processes audio in 7s windows (10s for M2D-CLAP) and applies max-pooling to identify segments with high alignment scores. This preserves critical acoustic events (e.g., bird chirps in ambient noise) that might be diluted by averaging over background sounds.

2. **Probabilistic FLEUR scoring**: Instead of using raw LALM outputs (which produce frequent ties), CAF-Score computes scores from digit token probability distributions. For example, if two captions both output "0.85" but have different probability distributions at decimal positions, CAF-Score yields distinct scores (0.876 vs. 0.851), resolving ties that would otherwise obscure quality differences.

3. **Hybrid calibration with α = 0.8**: The authors discovered that 80% weight on CLAP's coarse alignment and 20% on LALM's fine-grained reasoning yields optimal performance. This balance was validated through sensitivity analysis (Figure 3), showing that overemphasising LALM (α < 0.5) reduces robustness on diverse audio content.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
On the BRACE benchmark, CAF-Score achieved 75.28% accuracy on AudioCaps-Main (the highest among reference-free metrics) and 97.11% on BRACE-Hallucination (outperforming reference-based baselines FENSE at 96.40% and CLAIR-A at 91.51%). The optimal configuration (Qwen3-Omni-Instruct + M2D-CLAP) achieved 73.11% overall accuracy on BRACE-Main and 97.11% on BRACE-Hallucination. Notably, CAF-Score outperformed reference-based metrics in HH (Human-Human) scenarios where both captions were written by humans, demonstrating superior ability to detect subtle quality differences (Table 1).

## Related Work
CAF-Score extends FLEUR (originally for vision-language evaluation) to audio using LALMs, addressing a gap in the audio domain. While CLAP-based metrics (e.g., CLAPScore) provide reference-free evaluation, they fail to capture fine-grained semantic nuances. CAF-Score bridges this by using CLAP's coarse alignment as a foundation and LALM as a calibration mechanism. The authors also adopt BRACE, a modern benchmark designed for LALMs, which evaluates both misalignment (BRACE-Main) and hallucinations (BRACE-Hallucination).

## Limitations
The paper doesn't evaluate CAF-Score on longer audio clips beyond 30 seconds (typical audio duration in BRACE). The authors note LALM reasoning variants (e.g., AudioFlamingo3-Think) produced less stable scores, limiting the method to non-reasoning variants. Additionally, the sliding-window strategy requires careful tuning of window lengths (7s for MS-CLAP, 10s for others) to avoid performance degradation.

## Appendix: Worked Example
Consider an audio clip of 15 seconds (birds, wind, insects) with two candidate captions:
1. "The outdoor ambient noises includes birds, wind, and insects."
2. "The outdoor ambient noises includes birds, wind, and bees."

**Step 1: CLAP branch**  
- Input audio split into 7s windows: [0-7s, 7-14s, 14-21s] (truncating to 15s)
- Caption embeddings compared to each window:
  - Window 1 (0-7s: birds, wind): Similarity = 0.85
  - Window 2 (7-14s: wind, insects): Similarity = 0.82
  - Window 3 (14-15s: insects): Similarity = 0.78
- Max-pooling selects 0.85 (SS-CLAPScore = 0.85)

**Step 2: LALM branch**  
- LALM (Qwen3-Omni-Instruct) processes caption 1:  
  - Digit probabilities at decimal places:  
    - First decimal: P('8')=0.7, P('9')=0.3  
    - Second decimal: P('5')=0.6, P('4')=0.4  
  - FLEUR = (8×0.7 + 9×0.3)×0.1 + (5×0.6 + 4×0.4)×0.01 = 0.876  
- Caption 2 ("bees" instead of "insects"):  
  - Similar digit probabilities but different weights:  
    - First decimal: P('8')=0.6, P('9')=0.4  
    - Second decimal: P('5')=0.7, P('4')=0.3  
  - FLEUR = (8×0.6 + 9×0.4)×0.1 + (5×0.7 + 4×0.3)×0.01 = 0.857  

**Step 3: Combined score**  
- CAF-Score for caption 1: 0.8×0.85 + 0.2×0.876 = 0.855  
- CAF-Score for caption 2: 0.8×0.85 + 0.2×0.857 = 0.851  
- The 0.004 difference reveals that caption 1 (correct) is better than caption 2 (hallucination), resolving a tie that would occur with raw scores.

## References

- **Code:** https://github.com/inseong00/CAF-Score.
- Insung Lee, Taeyoung Jeong, Haejun Yoo, Du-Seong Chang, Myoung-Wan Koo, "CAF-Score: Calibrating CLAP with LALMs for Reference-free Audio Captioning Evaluation", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19615

Tags: #audio-processing #evaluation-metrics #large-language-models #reference-free-evaluation #audio-captioning
