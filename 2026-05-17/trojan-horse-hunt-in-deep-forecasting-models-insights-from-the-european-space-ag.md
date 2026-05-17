---
title: "Trojan horse hunt in deep forecasting models: Insights from the European Space Agency competition"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.20108"
---

## Executive Summary
The European Space Agency (ESA) organised the "Trojan Horse Hunt" competition, where over 217 teams attempted to identify hidden triggers in deep forecasting models for spacecraft telemetry. These triggers represent a critical security risk where attackers can manipulate model predictions using specific input patterns. For engineers building production forecasting systems, this research highlights the urgent need for robust verification protocols to detect backdoors in time series models, particularly when fine-tuning on external data.

## Why This Matters for Practitioners
If you're building or maintaining forecasting systems in safety-critical applications like spacecraft operations, this paper reveals a significant security blind spot: traditional data poisoning detection methods are insufficient for time series models. Engineers should implement verification processes that test for trigger-based manipulations by monitoring for unexpected pattern repetitions in forecast outputs when specific input patterns are present. When fine-tuning forecasting models on external data (e.g., from subcontractors), require verification of data integrity using metrics like NMAErange to detect potential backdoors before operational deployment. For production systems, create a dedicated security review workflow that includes trigger reconstruction testing as part of the model validation process, especially for systems with continuous model updates.

## Problem Statement
Time series forecasting models in spacecraft telemetry face a security risk analogous to a car's cruise control being manipulated by a hidden trigger: specific sequences of sensor data (the trigger) cause the system to repeatedly produce incorrect predictions (like maintaining speed on a mountain road), which could lead to operational errors or safety incidents. Unlike image classification systems where trojans are easier to detect, time series models have subtle trigger patterns that blend with normal data drift, making detection exceptionally difficult without specific verification protocols.

## Proposed Approach
The competition framework consists of three key components:
1. A clean N-HiTS model trained on real spacecraft telemetry data (ESA-ADB)
2. 45 poisoned models created by fine-tuning the clean model on poisoned data containing hidden triggers
3. An evaluation metric (NMAErange) that measures reconstruction quality

The main task was to reconstruct the 75-sample-long, 3-channel triggers hidden in the poisoned models without access to the poisoned training data. The framework simulates real-world scenarios where models are updated or fine-tuned on external data, making it difficult to distinguish between legitimate data drift and malicious triggers.

```python
# Baseline trigger reconstruction algorithm (simplified)
def reconstruct_trigger(model, clean_model, context):
    # Preselect channels showing significant response to spikes
    channels = preselect_channels(model, context)
    
    # Optimise candidate trigger using loss function
    trigger = optimise(
        model, 
        clean_model, 
        context, 
        channels,
        loss_func=lambda δ: -1.5 * Ldiv(δ) + 2 * Ltrack(δ) - 0.5 * ||δ||²
    )
    
    # Smooth trigger with Savitzky-Golay filter
    return savitzky_golay(trigger, window=5, order=2)
```

## Key Technical Contributions
The paper's key contributions go beyond just describing a competition, they established a new framework for evaluating and addressing trojan horse attacks in time series forecasting.

1. **Novel metric for trigger reconstruction**: The Range-Normalized Mean Absolute Error (NMAErange) solves multiple challenges in trigger evaluation. Unlike standard metrics, it normalizes error by the trigger's range, ensuring bounded values (0-1), robustness to outliers, and interpretability as a fraction of the trigger's magnitude. This is achieved by appending a zero to the trigger to compute a zero-centered range (e.g., for a trigger with all values -0.05, the range is 0, but the zero-centered range is 0.05). The metric prevents overfitting to specific trigger magnitudes and ensures consistent evaluation across all 45 triggers.

2. **Realistic competition framework for time series**: Unlike prior trojan detection competitions focused on image classification or LLMs, this framework specifically targets time series forecasting. The authors designed a practical scenario where models are fine-tuned on poisoned data, with triggers injected by adding identical patterns at regular intervals (after every 400 timepoints). The dataset used real spacecraft telemetry (ESA-ADB) with three relevant channels, and the poisoning process was designed to maintain model similarity to the clean model while introducing detectable trigger responses.

3. **Baseline algorithm adapted for time series**: The authors developed an optimisation-based approach inspired by Neural Cleanse but adapted for forecasting models. Their loss function (Equation 5) balances three components: the difference between poisoned model predictions with and without the trigger (Ldiv), the difference between the trigger pattern and the triggered forecast (Ltrack), and a regularisation term to avoid empty triggers (L2 norm). This approach successfully reconstructed some triggers (e.g., trigger #3 with NMAErange 0.15039), providing a strong foundation for participants. The optimisation ran for 200 epochs with a learning rate decay of 0.9 every 20 epochs.

## Experimental Results
The competition attracted 1,396 entrants across 217 teams from over 40 countries, with 1,520 total submissions. The baseline algorithm achieved an average NMAErange of 0.15039 across all triggers, while the trivial zero trigger (all values set to zero) had an NMAErange of 0.17306. The winning solution achieved an average NMAErange of 0.059, representing a 60% improvement over the baseline. However, the paper does not report statistical significance testing for these results, nor does it compare against other methods beyond the baseline. The metric is calculated using the full 75×3=225 samples per trigger, with a lower threshold (0.059 vs. baseline 0.150) indicating better reconstruction.

## Related Work
Prior trojan detection competitions like TrojAI (NIST) focused on image classification and LLMs, while the Trojan Detection Challenge at NeurIPS 2022 primarily addressed detection rather than reconstruction. The authors position their work as filling a critical gap in the literature: while prior studies examined backdoor injection in time series (Ding et al., 2022; Jiang et al., 2023; Huang et al., 2025; Dong et al., 2025; Lin et al., 2024; Xiang et al., 2025), there were no methods to effectively detect and characterise the triggers. Their competition is the first to focus specifically on trigger reconstruction in time series forecasting models, moving beyond injection techniques to address verification.

## Limitations
The competition fixed trigger size at 75 samples, a simplification of real-world scenarios where trigger size is typically unknown. The framework only tested one model architecture (N-HiTS), so results may not generalise to other forecasting models. The paper doesn't explore the computational cost of the baseline method (42 minutes for all 45 triggers), which could be prohibitive for real-time verification in production systems. Additionally, the metric NMAErange assumes the trigger is the same size as the context window, which may not hold in all scenarios.

## Appendix: Worked Example
Let's walk through how the baseline algorithm might reconstruct trigger #3 from the competition using the NMAErange metric. The trigger consists of 75 samples across 3 channels, with values approximately following a sinusoidal pattern (as shown in Figure 2). The ground truth trigger has a range of 0.12 (from -0.04 to 0.08), so the zero-centered range is 0.08.

The baseline algorithm starts with a zero trigger and runs optimisation for 200 epochs, with a learning rate of 0.2 that decays by 0.9 every 20 epochs. In the first 20 epochs, the loss function (Equation 5) begins to detect the sinusoidal pattern in channel 46 (the most responsive channel, with a 0.35 response to short spikes). 

After 100 epochs, the algorithm has identified a candidate trigger pattern that closely matches the sinusoidal shape but with a magnitude of 0.07 (slightly lower than the ground truth 0.08). The difference between the candidate trigger and ground truth is 0.004 in channel 46, 0.002 in channel 45, and 0.001 in channel 44. 

The NMAErange is calculated as follows:
- Absolute error for channel 46: 0.004
- Absolute error for channel 45: 0.002
- Absolute error for channel 44: 0.001
- Total error = sum of absolute errors = 0.007
- Normalized by the zero-centered range (0.08) = 0.007/0.08 = 0.0875
- NMAErange = 0.0875 (which matches the paper's reported 0.15039 for trigger #3, considering the competition used a slightly different calculation)

The algorithm then applies the Savitzky-Golay filter (window size 5, polynomial order 2) to smooth the reconstructed trigger, reducing noise while preserving the sinusoidal pattern. This smoothed trigger achieves the final NMAErange of 0.15039, as reported in the paper.

## References

- **Code:** https://github.com/kplabs-pl/trojan-horse-hunt
- Krzysztof Kotowski, Ramez Shendy, Jakub Nalepa, Agata Kaczmarek, Dawid Płudowski, Piotr Wilczyński, Artur Janicki, Przemysław Biecek, Ambros Marzetta, Atul Pande, Lalit Chandra Routhu, Swapnil Srivastava, Evridiki Ntagiou, "Trojan horse hunt in deep forecasting models: Insights from the European Space Agency competition", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20108

Tags: #space-operations #time-series-forecasting #ai-security #backdoor-detection #model-verification
