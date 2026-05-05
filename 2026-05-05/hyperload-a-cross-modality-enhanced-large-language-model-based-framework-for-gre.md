---
title: "HyperLoad: A Cross-Modality Enhanced Large Language Model-Based Framework for Green Data Center Cooling Load Prediction"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37011"
---

## Executive Summary
HyperLoad introduces a novel framework that leverages pre-trained large language models (LLMs) to predict cooling loads in green data centres with minimal training data, addressing data scarcity challenges that hinder traditional approaches. It outperforms state-of-the-art baselines by up to 52.1% in MAE reduction across varying forecast horizons and maintains robust performance even in extreme data-scarce scenarios (25% training data). For data centre operators, this means achieving accurate load forecasting without requiring extensive historical data, enabling more dynamic integration of renewable energy sources and reducing cooling-related energy waste.

## Why This Matters for Practitioners
Data centre cooling accounts for over 25% of total electricity use, making accurate load forecasting essential for reducing energy waste and carbon emissions (IEA 2024). If you're running a green data centre with limited operational history (common in new deployments), HyperLoad enables you to achieve precise cooling load predictions with as little as 25% of the training data required by traditional models. This means you can dynamically orchestrate cooling systems with renewables and storage in sub-minute intervals, potentially reducing PUE below 1.3 and cutting lifecycle carbon emissions by over 30%. For engineers, this translates to implementing a solution that requires no complex calibration like thermodynamic models, yet provides better accuracy than RNNs, CNNs, or Transformers with minimal data. You can integrate HyperLoad into your existing monitoring stack using a single NVIDIA A100 GPU, with training taking approximately 8 hours based on the paper's implementation.

## Problem Statement
Current data centre cooling load forecasting resembles a chef trying to cook a complex dish with only a few missing ingredients from a partially shredded recipe. Traditional models like Transformers, RNNs, and CNNs require large, task-specific datasets for training, but green data centres face data scarcity due to cold starts (new deployments), load distortion from renewable integration, and fragmented multi-source data (privacy barriers). These models overfit on limited data, losing generalisation and failing to capture complex dependencies between devices, much like a chef trying to replicate a dish without knowing the full recipe or ingredient proportions.

## Proposed Approach
HyperLoad consists of two phases: Cross-Modality Knowledge Alignment (KARI) and Multi-Scale Feature Modelling (ADPT and EGIA). In the KARI phase, textual priors and time-series data are mapped to a common latent space. In the ADPT phase, domain-aligned priors are injected through adaptive prefix-tuning, while EGIA captures cross-device temporal dependencies. The framework uses LLaMA-7B as its backbone, frozen during training to maintain pre-trained trend inference capabilities.

```python
# HyperLoad training pipeline (simplified)
def train_hyperload(data, kari_loss, adpt_strategy, egia_mechanism):
    # Cross-Modality Knowledge Alignment (KARI)
    time_series_features = time_series_encoder(data.time_series)
    text_features = text_encoder(data.text_template)
    align_features = kari_loss(time_series_features, text_features)
    
    # Multi-Scale Feature Modelling
    prefix_vectors = adpt_strategy(align_features, data.text_template)
    input_sequence = concatenate(prefix_vectors, time_series_features)
    encoded_features = llm_model(input_sequence)
    enhanced_features = egia_mechanism(encoded_features)
    
    # Prediction
    prediction = linear_projection(enhanced_features)
    loss = mse_loss(prediction, ground_truth)
    return loss
```

## Key Technical Contributions
HyperLoad introduces several novel mechanisms that enable accurate load prediction with scarce data. Each component specifically addresses a limitation of prior approaches:

1. **KARI Strategy**: Unlike methods that directly process raw text and temporal data (ignoring distributional disparities), KARI dynamically adjusts both encoders to project features into a shared latent space. It minimizes the distance between text and time-series features using cosine similarity with a temperature of 0.05, enabling LLMs to better leverage textual knowledge for temporal reasoning. This is distinct from approaches like Time-LLM or FPE-LLM that don't address the fundamental distributional mismatch between modalities.

2. **ADPT Strategy**: Rather than fine-tuning the entire LLM (which would require substantial data), ADPT encodes domain-specific background as learnable prefix vectors and concatenates them with the time-series features. This enables rapid adaptation to new scenarios with limited data, allowing the model to apply pre-training knowledge to data centre contexts without extensive retraining.

3. **EGIA Mechanism**: Existing methods model temporal dependencies but overlook inter-device collaboration and adaptive weight allocation. EGIA jointly learns multivariate sequence representations and their adaptive correlations, explicitly capturing how device collaboration impacts cooling load fluctuations. It achieves this by embedding device-level variables, computing query-key-value representations, and applying attention to emphasize highly correlated variables in subsequent interactions.

## Experimental Results
HyperLoad was evaluated on DCData (13,438 records from Dongguan data centre, 41 variables, partitioned 7:1:2 training/validation/test) using input sequence length 96 and forecasting lengths 12, 24, 48, 96. In data-sufficient settings (100% training data), HyperLoad achieved the lowest MSE and MAE across all forecast lengths. At 96-step prediction, it reduced MAE by 52.1% relative to Autoformer (0.0192 vs. 0.0656 MAE), 26.7% relative to FreTS (0.0192 vs. 0.0270 MAE), and 26.2% relative to TimesNet (0.0192 vs. 0.0522 MAE).

In data-scarce settings (50% training data), HyperLoad reduced MSE by 6.5% compared to SOTA baselines (0.0087 vs. 0.0093 MSE), and MAE by 3.7%. With only 25% training data, it reduced MSE by 16.7% (0.0197 vs. 0.0236 MSE) and MAE by 9.5% (0.0969 vs. 0.1067 MAE). Ablation studies confirmed that each component contributes meaningfully: removing ADPT increased MSE by 12.7%, removing KARI by 1.0%, and removing EGIA by 1.0%.

## Related Work
HyperLoad builds on recent work applying LLMs to time-series forecasting, such as PromptCast and LF-PLM, which focus on enhancing time-series feature representation but lack contextual reasoning. Time-LLM and FPE-LLM incorporate multi-modal fusion but directly process raw text and temporal data, ignoring distributional disparities. TEMPO uses STL decomposition for seasonal patterns but struggles with complex seasonal dynamics. HyperLoad advances these approaches by specifically addressing the distributional gap between text and time-series data through KARI, enabling better utilisation of textual priors.

## Limitations
The paper doesn't report computational complexity during inference, though training uses a single NVIDIA A100 GPU with 8-layer LLaMA-7B. The DCData was collected from a single data centre in Dongguan (2024), so generalizability to different geographical locations or data centre architectures isn't tested. The authors don't address how the framework handles catastrophic forgetting when adapting to new data centre environments. The paper also doesn't compare HyperLoad against domain-specific fine-tuned LLMs for time-series forecasting, which could provide additional insights.

## Appendix: Worked Example
Let's walk through a concrete example of how HyperLoad processes data for a 96-step forecast with 25% training data:

1. **Data Collection**: The system collects 5-minute interval data from a data centre over October-December 2024, resulting in 13,438 records of 41 variables (outdoor temperature, cooling water temperatures, etc.).

2. **Cross-Modality Input Construction**: For a single time series segment (96 time steps), the system applies reversible instance normalization to each feature column. For example, for inlet cooling water temperature (feature 3), the mean is 15.3°C and standard deviation 2.1°C, so each value is normalized as (x - 15.3)/2.1.

3. **Text Template Creation**: The Context-Aware Temporal Synthesis Template (CATS) is constructed by concatenating:
   - Domain knowledge base (background: "Data centre cooling system consists of 9 cooling pumps and 9 chillers with inlet/outlet temperature sensors"; instruction: "Predict cooling load for next 96 steps")
   - Trend description: "Temperature shows a 15% daily fluctuation with a 3-hour cycle"
   - Statistics: "Mean inlet temperature: 16.2°C, standard deviation: 2.3°C"

4. **KARI Alignment**: The time-series features (V_s) and text features (V_t) are projected into a shared latent space using LKARI loss. For a batch size of 64, the cosine similarity between aligned features is maximised with temperature τ=0.05.

5. **ADPT Implementation**: The text encoder's output (V_T) is used as a prefix (20 tokens) and concatenated with time-series features (96 tokens), resulting in an input sequence of 116 tokens. This concatenated sequence is fed into the LLM backbone.

6. **EGIA Mechanism**: The embedded device-level features (S) for 9 cooling pumps are projected into query (Q), key (K), and value (H) representations. For example, Q = S*WQ (9x768 dimension), K = S*WK (9x768), H = S*WH (9x768). The attention mechanism computes V_E = Softmax(QK^T/√768)H.

7. **Prediction**: The LLM outputs a prediction for the next 96 time steps, which is inverse-normalized to obtain final cooling load predictions.

## References

- Haoyu Jiang, Boan Qu, Junjie Zhu, Fanjie Zeng, Xiaojie Lin, Wei Zhong, "HyperLoad: A Cross-Modality Enhanced Large Language Model-Based Framework for Green Data Center Cooling Load Prediction", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37011

Tags: #data-centre-optimisation #llm-forecasting #cross-modality #adaptive-prompting #energy-efficiency
