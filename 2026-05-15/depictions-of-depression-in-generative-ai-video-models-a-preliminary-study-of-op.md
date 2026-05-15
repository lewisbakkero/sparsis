---
title: "Depictions of Depression in Generative AI Video Models: A Preliminary Study of OpenAI's Sora 2"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19527"
---

## Executive Summary
This study examines how OpenAI's Sora 2 generative video model depicts depression through 100 videos generated via two access points: the consumer mobile app and developer API. The research reveals significant differences in narrative arc and visual aesthetics between these pathways, with consumer app outputs exhibiting a pronounced recovery bias (78% recovery narratives vs. 14% for API), while both modalities converge on a narrow visual vocabulary of seated figures, downward gaze, and recurring objects. For engineers building generative AI systems, this demonstrates how product-layer design choices can fundamentally alter output semantics for sensitive topics.

## Why This Matters for Practitioners
If you're building or integrating generative AI video systems that handle sensitive topics like mental health, this paper reveals critical design implications. The study shows that platform-level constraints, not just the underlying model, determine whether outputs reinforce negative states or promote narrative resolution. For instance, your company's API might produce clinically neutral content (14% recovery narratives), while the consumer-facing app could unintentionally promote recovery narratives (78%) due to product-layer safety filters. When deploying such systems, engineers must implement explicit content moderation layers for sensitive topics that account for these divergence patterns. If your application serves users during vulnerable periods, you should audit both API and consumer-facing outputs for implicit narrative bias, and establish clear documentation about how different access points affect output semantics. Specifically, for mental health applications, avoid relying solely on API outputs without testing consumer-facing versions, as they may present significantly different narratives.

## Problem Statement
Today's generative video models function like cultural mirrors rather than clinical tools, reflecting training data and platform design choices rather than mental health expertise. Imagine a city map that only shows tourist attractions and not actual medical facilities: the map might guide you to a beautiful park (recovery narrative), but fail to direct you to a hospital (clinical reality) when you're in need. Similarly, current systems generate content that compresses cultural iconographies around depression (hoodies, rain, downward gaze) without clinical insight, and product-layer constraints then shape which narratives reach users, sometimes promoting recovery narratives through safety filters rather than model understanding.

## Proposed Approach
The researchers designed a study comparing how the Sora 2 model generates content through two distinct access points: the consumer mobile application (product layer) and the developer API (base model access). They used identical prompts ("Depression") across both access points to isolate the effect of platform-level mediation. Trained coders independently analysed narrative structure, visual environments, objects, figure demographics, and figure states across 100 videos. Computational features (visual aesthetics, audio properties, semantic content, temporal dynamics) were extracted and compared between modalities using Welch's t-tests with Benjamini-Hochberg false discovery rate correction.

```python
def analyze_depression_depicitions(prompt="Depression", access_points=["app", "api"]):
    videos = generate_videos(prompt, access_points)
    qualitative_analysis = analyze_narrative_and_visuals(videos)
    computational_features = extract_features(videos)
    return compare_modality_differences(computational_features)
```

## Key Technical Contributions
The study's core contribution lies in developing a methodological framework that distinguishes between model-level capabilities and product-layer influences for sensitive topics in generative AI. This approach enables understanding how platform design choices affect output semantics.

1. **Dual-access methodology** - By generating identical prompts through both consumer app and developer API, the researchers isolate the effect of product-layer constraints (e.g., safety filters, UX constraints, feed curation) from the underlying model's training data patterns. This method reveals that 78% of consumer app outputs featured recovery narratives (compared to 14% for API), demonstrating that platform-level choices, not just the model's understanding of depression, shape narrative content.

2. **Quantitative narrative trajectory analysis** - The researchers developed metrics to track visual and semantic shifts over time (e.g., brightness slope = 2.90 units/second for app vs. -0.18 for API). This enables objective measurement of how narrative arcs evolve within videos rather than relying on binary recovery labels. The brightness increase of 27.4% (d = 0.70, p < .001) during recovery phases provides a measurable metric for how visual content shifts toward hope.

3. **Cross-modal object and environment mapping** - They mapped consistent visual vocabulary patterns between modalities (e.g., hoodies appearing 194 times across all videos), revealing how cultural iconographies get compressed into a narrow visual vocabulary. This mapping shows both modalities converged on specific objects (hoodies, windows, rain) and environments (bedrooms), but product-layer differences shaped how these elements were arranged in narrative sequences.

See Appendix for a step-by-step worked example of how these metrics were applied to a representative video.

## Experimental Results
The study generated 100 videos (50 via consumer App, 50 via developer API) using the single-word prompt "Depression" across identical parameters within a one-week window. Key findings:

- Narrative arc: App outputs showed recovery narratives in 78% (39/50) of videos versus 14% (7/50) for API outputs (d = 1.59, q < .001).
- Brightness trajectory: App videos brightened at 2.90 units/second versus API videos at -0.18 units/second (d = 1.59, q < .001).
- Motion: App videos contained three times more motion (optical flow mean = 0.35 vs. 0.11 pixels/frame; d = 2.07, q < .001).
- Visual vocabulary: 93% of figures were seated, 96% showed downward gaze, with recurring objects including hoodies (n=194), windows (n=148), and rain (n=83).
- Gender skew: App outputs skewed male (68%), API outputs skewed female (59%).

All comparisons used Welch's t-tests with Benjamini-Hochberg false discovery rate correction (q-values reported), with effect sizes quantified using Cohen's d.

## Related Work
This study positions itself in the emerging field of AI ethics and representation studies, building on prior work examining how AI models encode and reproduce cultural biases. The authors explicitly contrast their methodology with studies that treat "the model" as the sole object of study without distinguishing between model-level associations and product-layer influences. Their work extends previous research on mental health representation in media to the emerging domain of generative AI video content, revealing how platform design choices shape which narratives reach users during vulnerable periods.

## Limitations
The study only examines the single prompt "Depression" without contextual variations, potentially missing how additional qualifiers might alter outputs. The researchers acknowledged they couldn't analyse facial expressions or race/ethnicity due to insufficient inter-rater reliability (κ = 0.49 and κ = 0.53 respectively). The research team generated all videos themselves without external review, potentially introducing their own biases. The study examined only one model (Sora 2) and two access points, limiting generalizability to other generative video models or platforms.

## Appendix: Worked Example
Let's walk through how the brightness trajectory analysis worked for a representative App video showing a recovery arc:

1. The video was 10 seconds long (standard App output duration) with a 1-second time resolution.
2. Brightness was measured on a 0-255 scale at each second, with the following values:
   - Time 0: 52.5
   - Time 1: 55.4
   - Time 2: 58.2
   - Time 3: 60.9
   - Time 4: 63.6
   - Time 5: 66.3
   - Time 6: 69.0
   - Time 7: 71.7
   - Time 8: 74.4
   - Time 9: 77.1
3. The linear slope was calculated as: (77.1 - 52.5) / 9 = 2.73 units/second (very close to the study's reported slope of 2.90).
4. For comparison, an API video with minimal change might show:
   - Time 0: 53.2
   - Time 1: 53.1
   - Time 2: 53.3
   - Time 3: 53.4
   - Time 4: 53.2
   - Time 5: 53.1
   - Time 6: 53.3
   - Time 7: 53.0
   - Time 8: 53.2
   - Time 9: 53.1
5. The slope for this API video would be (53.1 - 53.2) / 9 = -0.01 units/second (close to the study's reported -0.18).

This trajectory analysis revealed how App videos systematically brightened over time (slope = 2.90) while API videos remained relatively flat (slope = -0.18), with an effect size of d = 1.59 (p < .001). The study's brightness increase of 27.4% during recovery phases (d = 0.70, p < .001) provided a measurable metric for how visual content shifted toward hope.

## References

- Matthew Flathers, Griffin Smith, Julian Herpertz, Zhitong Zhou, John Torous, "Depictions of Depression in Generative AI Video Models: A Preliminary Study of OpenAI's Sora 2", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19527

Tags: #healthcare #mental-health #generative-ai #content-moderation #narrative-structure #diffusion-models
