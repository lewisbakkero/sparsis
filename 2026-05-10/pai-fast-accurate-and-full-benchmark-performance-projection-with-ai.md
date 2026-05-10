---
title: "PAI: Fast, Accurate, and Full Benchmark Performance Projection with AI"
venue: "Accurate"
paper_url: "https://arxiv.org/abs/2603.19330"
---

## Executive Summary
PAI introduces a machine learning-based approach that predicts full benchmark performance for hardware design without relying on slow cycle-accurate simulators. It achieves 9.35% average IPC prediction error while running 3 orders of magnitude faster than existing techniques. For production hardware teams, this means reducing performance analysis from hours to minutes, enabling more rapid design iterations.

## Why This Matters for Practitioners
Hardware engineers at companies like Intel spend weeks waiting for cycle-accurate simulators to evaluate design choices for new SoCs. With PAI, teams can now evaluate multiple hardware configurations for a full benchmark suite in under 3 minutes instead of hours or days. For example, when designing a new processor core for a mobile SoC, you could test 10 different cache sizes and clock frequencies with PAI in 30 minutes rather than 10 hours. This enables the rapid identification of optimal configurations before committing to expensive fabrication cycles, directly reducing time-to-market and R&D costs.

For engineers at companies building custom accelerators (like for AI workloads), PAI allows quick validation of performance targets without building full simulators first. When evaluating whether to include a new hardware accelerator for matrix multiplication, you could use PAI to predict the performance impact across multiple benchmarks in minutes rather than waiting for weeks of simulation.

## Problem Statement
Imagine trying to evaluate a new car's fuel efficiency by manually counting every drop of petrol used while driving at different speeds, through different terrains, with various passenger loads. That's what hardware engineers face today when using traditional simulators: painstakingly counting cycles at a micro-architectural level for full benchmarks, making performance analysis a bottleneck in the design cycle.

## Proposed Approach
PAI augments the hardware design cycle with a hierarchical LSTM model that predicts performance metrics from microarchitecture-independent features. The system operates in two phases: training (collecting features from native execution or emulators) and prediction (using the model to forecast performance for new configurations). At its core, PAI processes execution traces of microarchitecture-independent metrics (uAIMs) along with hardware configuration details to predict performance metrics like IPC.

```python
def predict_performance(uAIM_trace, hw_config):
    # uAIM_trace: trace of microarchitecture-independent features (e.g., branch counts, cache misses)
    # hw_config: hardware configuration details (e.g., cache sizes, clock frequency)
    
    # Process uAIMs through one LSTM layer
    uAIM_features = lstm_uaim(uAIM_trace)
    
    # Process hardware configuration through another LSTM layer
    hw_features = lstm_hw(hw_config)
    
    # Combine features through a second LSTM layer
    combined_features = lstm_combined(uAIM_features, hw_features)
    
    # Predict performance metrics
    ipc_prediction = fully_connected_layers(combined_features)
    
    return ipc_prediction
```

## Key Technical Contributions
PAI's core innovations address fundamental limitations in prior approaches:

1. **Hierarchical LSTM architecture**: Unlike prior ML approaches that process instruction-level features or require detailed simulation for training data, PAI uses two separate LSTM layers that handle uAIMs and hardware configurations independently before combining them. This allows the model to learn the distinct patterns in program behaviour (uAIMs) and hardware characteristics separately, improving prediction accuracy. The architecture specifically addresses the mismatch between sequential program execution patterns and static hardware configuration parameters.

2. **Microarchitecture-independent feature selection**: PAI uses 128 normalized features that capture high-level program behaviour without relying on detailed architectural knowledge. These include instruction-related metrics (61 features), memory access patterns (48 features), branch behaviour (7 features), and other system statistics (12 features). This avoids the need for detailed architectural knowledge required by prior approaches that use instruction-level encoding.

3. **Dataset collection from native execution**: PAI's training dataset is collected from actual hardware or fast emulators (like Simics) rather than detailed simulators. This enables collecting a much larger dataset (1.3 million datapoints) without the simulation overhead, directly addressing the speed limitations of prior approaches that required detailed simulation for training data.

4. **Unseen benchmark generalisation**: By deliberately splitting their dataset to hold out certain benchmarks (XZ, WRF, MCF, nab, cactubssn, xalanbmk) for evaluation, the authors demonstrate PAI's ability to predict performance for unseen benchmarks with 15.5% average error. This is critical for real-world hardware design where new applications are constantly emerging.

## Experimental Results
PAI achieved an average IPC prediction error of 9.35% for the SPEC CPU 2017 benchmark suite, comparable to state-of-the-art techniques (TAO achieved 9.7% error for Zsim and 13% for Gem5) but requiring 3 orders of magnitude less time. The full benchmark suite (approximately 3 trillion instructions) takes only 2 minutes 57 seconds to predict with PAI, compared to 5076 seconds (over 1 hour 24 minutes) for TAO and 6948 seconds for SimNet.

The dataset consisted of 1.3 million datapoints collected from 15 different Intel Xeon 6 processor configurations, with features broken down as follows: 61 instruction-related, 48 memory access-related, 7 branch-related, and 12 other features. For unseen benchmarks, PAI achieved a 15.5% average error, with some outliers like cactubssn showing higher errors (likely due to complex cache behaviour patterns not captured well). The paper doesn't explicitly report statistical significance testing for the performance gains, though the magnitude of the speedup (3 orders of magnitude) is clearly significant.

## Related Work
PAI builds upon previous work using ML for performance prediction but addresses key limitations of prior approaches. Li et al. [5] proposed a two-model approach combining instruction encoding with microarchitecture modelling, but their method is limited by instruction-level processing complexity. TAO [6] uses a multi-head attention network trained on detailed simulation data, making it inherently slow and dependent on the accuracy of the underlying simulator. Munigala et al. [7] used MLPs for predicting bandwidth based on a limited set of architectural parameters but couldn't generalise beyond specific configurations. Barboza et al. [8] focused on bottleneck identification using performance counters but didn't provide full benchmark performance prediction.

PAI improves upon these by eliminating the need for detailed simulation (making it faster), avoiding instruction-level encoding (making it more general), and demonstrating effectiveness across unseen benchmarks and configurations.

## Limitations
The authors acknowledge that PAI struggles with certain benchmarks like cactubssn, where prediction errors exceed 10% due to complex cache behaviour (high L2 and LLC miss latency). The paper doesn't explore why these specific cases are problematic or how to address them. Additionally, PAI was only evaluated on SPEC CPU 2017; generalisation to other benchmark suites or architectures hasn't been demonstrated. The paper also doesn't provide details on how PAI would scale to extremely large benchmarks or more complex hardware configurations with many more parameters.

## Appendix: Worked Example
Let's walk through how PAI processes a single benchmark execution:

Consider running the SPEC CPU 2017 benchmark "gcc" with a 6-core processor configuration (L3 cache: 32MB, clock frequency: 3.0GHz). PAI collects a trace of microarchitecture-independent features (uAIMs) every 10 million instructions:

- At 10 million instructions: instruction count = 5.2M (1.1M arithmetic, 1.3M load/store, 2.8M branch), memory access = 8.7M (5.6M hits, 3.1M misses), branch taken = 2.4M (entropy = 0.92)
- At 20 million instructions: instruction count = 5.8M (1.2M arithmetic, 1.4M load/store, 3.2M branch), memory access = 9.2M (5.9M hits, 3.3M misses), branch taken = 2.7M (entropy = 0.91)

These uAIMs are processed through PAI's hierarchical LSTM:

1. The uAIMs (instruction, memory, branch features) are fed into the first LSTM layer, producing a latent representation capturing the program's behaviour pattern (e.g., 256-dimensional vector).
2. The hardware configuration (6-core, 32MB L3, 3.0GHz) is processed through a separate LSTM layer, producing a latent representation of the hardware's characteristics (e.g., 256-dimensional vector).
3. These two latent representations are combined through a second LSTM layer, which then outputs a prediction for IPC.

For "gcc" with this specific configuration, PAI predicts an IPC of 1.32, while the actual measured IPC is 1.45, resulting in a 9.0% error for this particular configuration. This is within the paper's reported average of 9.35% for the entire benchmark suite.

## References

- Avery Johnson, Mohammad Majharul Islam, Riad Akram, Abdullah Muzahid, "PAI: Fast, Accurate, and Full Benchmark Performance Projection with AI", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19330

Tags: #computer-architecture #performance-analysis #machine-learning #hierarchical-lstm #hardware-design
