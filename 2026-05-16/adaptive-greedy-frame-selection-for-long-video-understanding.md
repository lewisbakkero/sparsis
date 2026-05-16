---
title: "Adaptive Greedy Frame Selection for Long Video Understanding"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20180"
---

## Executive Summary
This paper introduces a query-aware frame selection method for long video understanding that dynamically balances question relevance with semantic coverage. By formulating frame selection as a monotone submodular optimisation problem, the method achieves up to 10.83% higher accuracy than uniform sampling at the same frame budget, with no additional computational overhead. For production engineers, this means significant performance gains in video question-answering systems without increasing inference costs.

## Why This Matters for Practitioners
If you're building a production video analysis system using large vision-language models (VLMs), this paper shows you can achieve up to 10.83% higher accuracy at the same frame budget by using their adaptive frame selection method. For example, at a budget of 10 frames for Qwen3-VL on MLVU, their best method (Relevance oriented) achieved 72.91% accuracy compared to 65.95% for the base VLM with uniform sampling. This means you can either maintain performance while reducing the number of frames processed (saving up to 70% in compute costs for the same accuracy) or improve accuracy while keeping the same frame budget. The paper's deployable router (using a lightweight question-type classifier) further shows you can realize 59.6% of the potential improvement from perfect category-aware selection without needing category labels at test time.

## Problem Statement
Today's video analysis systems face a fundamental tension between processing too many frames (increasing visual token costs) and sampling too sparsely (missing critical moments). Imagine trying to answer "What was the main event in a 10-hour documentary?" by either watching all 10 hours (impractical) or randomly picking 10 frames (which might miss key scenes). The problem isn't just about selecting more frames, it's about selecting the right frames for the question at hand, as different question types require different trade-offs between relevance to the question and broad semantic coverage.

## Proposed Approach
The authors propose a query-aware frame selection method that dynamically balances question relevance with semantic coverage. The system creates a candidate pool of frames (1 FPS, capped at 1,000 frames), embeds each candidate in two complementary spaces (SigLIP for question relevance and DINOv2 for semantic similarity), and optimizes a weighted sum of these signals using a greedy algorithm. It routes questions to one of four interpretable presets based on question type, using a lightweight question-type classifier.

```python
def adaptive_frame_selection(video, question, K=10):
    # Candidate pool construction (1 FPS, capped at 1000 frames)
    candidates = create_candidate_pool(video, max_frames=1000)
    
    # Embedding in two spaces
    siglip_scores = compute_siglip_relevance(candidates, question)
    dino_scores = compute_dino_coverage(candidates)
    
    # Define four presets (α, β)
    presets = {
        "coverage_only": (0, 1),
        "coverage_oriented": (0.5, 1),
        "relevance_only": (1, 0),
        "relevance_oriented": (1, 0.5)
    }
    
    # For deployable routing, choose preset based on question type
    question_category = predict_question_type(question)
    preset = choose_preset(question_category, presets)
    
    # Greedy optimisation of weighted sum
    selected = greedy_optimization(
        candidates, 
        siglip_scores, 
        dino_scores,
        K,
        preset["alpha"],
        preset["beta"]
    )
    
    return selected
```

## Key Technical Contributions
The authors make three key technical contributions:

1. They formulate frame selection as a query-aware subset selection problem that combines two complementary signals: question relevance (via SigLIP) and semantic coverage (via DINOv2). The critical insight is that the weighted sum of these signals yields a normalized monotone submodular function, which theoretically guarantees a (1-1/e) approximation for the optimal selection under a fixed frame budget. This is not merely a practical heuristic but has a formal guarantee.

2. They define four interpretable presets by varying the trade-off between relevance and coverage (Coverage only, Coverage oriented, Relevance only, Relevance oriented), which allows for both simple baseline comparisons and a practical routing strategy. The presets are not arbitrary, each corresponds to a specific weight configuration in the objective function that the authors can theoretically justify.

3. They design a deployable routing mechanism using a lightweight text-only question-type classifier that predicts question categories (e.g., plotQA, needle, ego) before selecting frames. Crucially, they validate this routing strategy on a held-out test split and show it realizes 59.6% of the potential improvement from perfect category-aware routing without requiring category labels at test time.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
On MLVU with Qwen2-VL, the best method (Relevance oriented) achieved 72.91% accuracy at K=10 frames compared to 65.95% for uniform sampling (a +6.96% improvement) and 64.65% for AKS (a +8.26% improvement). With Qwen3-VL at K=10, the improvement was even more pronounced: 72.91% vs. 65.95% (a +6.96% improvement) and 68.03% for AKS (a +4.88% improvement). For LongVideoBench with Qwen3-VL, their method achieved 66.39% accuracy at K=20 frames compared to 63.92% for AKS (a +2.47% improvement). The Oracle Strategy (which uses ground-truth categories) exceeded the best fixed strategy by 1.40% on average across settings, showing the potential headroom from perfect category-aware selection.

## Related Work
The authors position their work as building on and extending prior frame selection methods like Adaptive Keyframe Sampling (AKS), which combines prompt relevance with keyframe coverage, but they go beyond by making the relevance-coverage trade-off question-dependent. They connect to classic relevance-diversity retrieval approaches like Maximal Marginal Relevance and submodular coverage objectives, while demonstrating that a single fixed selection rule is insufficient across heterogeneous question types. Their work is complementary to recent adaptive acquisition frameworks that decide when and how to gather additional video evidence.

## Limitations
The authors acknowledge that their deployable router requires training on a dataset with category labels, which might not be available for all applications. Their experiments were limited to Qwen2-VL and Qwen3-VL, so results might differ with other VLMs. The paper doesn't explore the impact of different VLMs on the effectiveness of the routing strategy, nor does it address how the method would perform with extremely long videos (beyond the 1,000 frame candidate pool cap).

## Appendix: Worked Example
Let's walk through a specific example from the paper using MLVU with Qwen3-VL at K=10 frames for a "needle" question (e.g., "What was the colour of the car at 02:15?"). The video is a 5-minute documentary (300 seconds) with 30 frames per second (9,000 frames total). The candidate pool is created by taking 1 frame per second, resulting in 300 frames. Since this is less than the 1,000 frame cap, all 300 frames are included as candidates.

For this needle question, the authors would prefer a Relevance oriented preset (α=1, β=0.5). The SigLIP relevance scores for the candidate frames are calculated, with frame 128 (at 2 minutes 8 seconds) having the highest score (0.85) as it contains the car in question. The DINOv2 coverage scores are calculated to identify frames that represent the broader semantic structure of the video.

The greedy algorithm starts with an empty set and iteratively selects frames that maximise the weighted sum (α × relevance + β × coverage). At the first iteration, frame 128 is selected (relevance score 0.85, coverage score 0.32). For subsequent iterations, the marginal gain for each candidate is calculated based on its remaining relevance and how well it covers the remaining semantic space not yet represented by selected frames.

After selecting 10 frames, the system uses these frames to answer the question. The results show that this method achieved 72.91% accuracy for needle questions at K=10, significantly outperforming uniform sampling (65.95%) and AKS (68.03%). This specific example demonstrates how the method's query-aware selection directly addresses the need for localized evidence in needle-type questions.

## References

- Yuning Huang, Fengqing Zhu, "Adaptive Greedy Frame Selection for Long Video Understanding", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20180

Tags: #video-analysis #vision-language-models #submodular-optimisation #query-aware-selection
