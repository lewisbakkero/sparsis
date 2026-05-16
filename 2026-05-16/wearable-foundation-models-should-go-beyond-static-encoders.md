---
title: "Wearable Foundation Models Should Go Beyond Static Encoders"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19564"
---

## Executive Summary
This paper argues that current wearable foundation models (WFMs) are fundamentally limited by their reliance on static encoders that treat health monitoring as short-term pattern recognition rather than longitudinal reasoning. The authors identify three foundational shifts needed to enable WFMs to support chronic health management through structured data ecosystems, longitudinal-aware modelling, and agentic inference systems.

## Why This Matters for Practitioners
If you're building health monitoring systems for chronic conditions such as diabetes management or cardiac rehabilitation, this paper suggests you're currently operating with a flawed paradigm. Current WFMs trained on short temporal windows (e.g., 5-minute segments for activity recognition) cannot provide meaningful insights for conditions that evolve over months. Instead of optimising for immediate classification accuracy, you should redesign your data pipelines to capture continuous longitudinal trajectories, tracking how a user's ECG patterns shift over weeks rather than predicting activity classes from isolated samples. Start by implementing context-aware data collection (e.g., integrating self-reported symptom logs with physiological signals), then evaluate how your existing models perform when trained on multi-day rather than single-day segments. The most critical engineering decision isn't about model architecture, it's about building data infrastructure that captures health as a continuous process, not a series of snapshots.

## Problem Statement
Current WFMs operate like a security camera that only records 5-second clips of a building's hallway, perfect for identifying who walked through at a specific moment, but utterly useless for spotting patterns of suspicious activity over weeks. Similarly, today's health monitoring systems capture physiological signals in short bursts (e.g., 1-minute windows for activity recognition) but treat them as independent snapshots rather than connected elements of a longer health narrative. This approach fails to detect gradual changes like a patient's increasing heart rate variability preceding a cardiac episode, which requires seeing how data evolves across days, weeks, or months.

## Proposed Approach
The authors propose a three-pronged framework for next-generation WFMs that shifts from retrospective pattern recognition to anticipatory health reasoning. First, they advocate for structurally rich data ecosystems that integrate multimodal physiological signals with contextual metadata across long timeframes. Second, they propose longitudinal-aware multimodal modelling that prioritizes temporal abstraction and personalization rather than population-level prediction. Third, they introduce agentic inference systems where models don't just predict labels but support decision-making under uncertainty. The core architecture connects three stages: (a) data collection with integrated contextual metadata; (b) multimodal pretraining with longitudinal awareness; (c) inference systems that support planning and intervention.

```python
def wearable_foundation_model_pipeline(data_stream):
    # Stage 1: Data Collection (integrated multimodal trajectory)
    context = gather_contextual_metadata(data_stream)  # Includes behaviour, environment, self-reports
    longitudinal_trajectory = integrate_multimodal_signals(
        data_stream, context
    )  # Combines ECG, IMU, PPG across days/weeks
    
    # Stage 2: Longitudinal-aware Pretraining
    model = pretrain_multimodal_model(
        trajectory=longitudinal_trajectory,
        objective=long_context_contrastive_learning
    )
    
    # Stage 3: Agentic Inference
    recommendation = model.generate_recommendation(
        current_observation=longitudinal_trajectory[-1],
        historical_context=longitudinal_trajectory[:-1]
    )
    return recommendation
```

## Key Technical Contributions
The paper's contributions focus on reframing the entire WFM pipeline rather than specific algorithms. The key innovations are:

1. **Longitudinal data integration framework**: Current datasets like Apple Heart and Movement Study (2.5B hours) remain underutilised because they're structured as isolated segments rather than continuous trajectories. The authors demonstrate that integrating contextual metadata (e.g., self-reported stress levels) with physiological signals enables models to distinguish between similar heart rate patterns caused by exercise versus anxiety, which requires seeing how data evolves across days rather than at single points.

2. **Temporal abstraction mechanism**: Unlike current WFMs that treat each 1-minute segment as independent, the authors propose a hierarchical temporal representation where lower-level features (e.g., heart rate variability) are aggregated across longer windows before being fed into the model. This allows the system to detect gradual changes like a rising resting heart rate over a month rather than focusing on instantaneous spikes.

3. **Agentic inference design**: The paper moves beyond static prediction to systems that actively support decision-making. For instance, rather than simply classifying a sleep stage, the model would generate a recommendation like "Based on your increasing heart rate variability over the past 7 days (from 55ms to 68ms), consider reducing caffeine intake before 3 PM to potentially improve sleep quality," with confidence thresholds to trigger medical consultation when uncertainty exceeds a threshold.

## Experimental Results
The paper doesn't provide specific quantitative results comparing static versus longitudinal models, as its focus is on identifying systemic issues rather than presenting a new model. The authors cite that existing approaches like WBM (trained on 2.5 billion hours) still limit downstream applications to fixed prediction tasks (e.g., sleep duration estimation, diabetes detection), suggesting their static paradigm constrains practical utility. The paper doesn't specify statistical significance metrics or baseline comparisons for new approaches, as it's primarily a conceptual framework.

## Related Work
The authors position their work in contrast to current WFM literature (e.g., LSM, WBM, SensorLM) that primarily focuses on improving static pattern recognition. They build on recent multimodal pretraining approaches (Narayanswamy et al., 2025; Luo et al., 2024) but argue these approaches inherit the same static encoder limitations. The paper explicitly references works like SleepFM (Thapa et al., 2026) that introduce limited agentic elements but remain bound to closed-world formulations (e.g., fixed activity label sets).

## Limitations
The paper doesn't propose or evaluate specific implementations of the three foundational shifts, leaving technical details to future work. It acknowledges that current data ecosystems lack sufficient open, longitudinal datasets for validation (e.g., most population-scale datasets remain proprietary). The authors don't address computational constraints of processing continuous longitudinal data streams at scale, nor do they specify how to handle missing data in long-term trajectories.

## Appendix: Worked Example
Imagine a diabetes management system tracking a patient over 30 days. The raw data includes ECG, PPG, and activity data (IMU) sampled at 1Hz, plus daily self-reported stress levels.

1. **Data Collection**: The system integrates physiological signals with contextual metadata. On Day 1, the patient reports "high stress" during a work meeting while heart rate variability (HRV) is 55ms. On Day 15, they report "moderate stress" with HRV at 62ms. On Day 30, self-reported stress is low but HRV has increased to 71ms.
   
2. **Longitudinal Processing**: Instead of treating each day's HRV as an isolated point, the system constructs a trajectory showing HRV increasing from 55ms to 71ms over 30 days. This trajectory is compared against the patient's own baseline (e.g., their average HRV over the previous month was 58ms).

3. **Agentic Inference**: Based on the trajectory showing sustained HRV increase (exceeding their 5% daily threshold), the system generates: "Your heart rate variability has increased 27% over the past month (from 55ms to 71ms), suggesting rising cardiovascular strain. We recommend scheduling a consultation with your cardiologist if this trend continues for another 7 days." The system calculates confidence (82% based on trajectory consistency) and sets a monitoring threshold for future days.

## References

- Yu Yvonne Wu, Yuwei Zhang, Hyungjun Yoon, Ting Dang, Dimitris Spathis, Tong Xia, Qiang Yang, Jing Han, Dong Ma, Sung-Ju Lee, Cecilia Mascolo, "Wearable Foundation Models Should Go Beyond Static Encoders", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19564

Tags: #life-sciences #health-monitoring #longitudinal-data #agentic-systems #context-aware-computing
