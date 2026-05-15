---
title: "Data-driven ensemble prediction of the global ocean"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19591"
---

## Executive Summary
FuXi-ONS is the first machine-learning ensemble forecasting system for global ocean prediction, providing probabilistic forecasts for sea-surface temperature, sea-surface height, subsurface temperature, salinity, and ocean currents at 5-day intervals up to 365 days. Unlike conventional numerical models that require multiple expensive integrations, FuXi-ONS learns physically structured perturbations and incorporates an atmospheric encoding module to stabilize long-range forecasts, running orders of magnitude faster than traditional systems while maintaining or improving accuracy. For engineers building ocean forecasting systems, this represents a major shift from computationally intensive numerical approaches to efficient, scalable probabilistic forecasting.

## Why This Matters for Practitioners
If your team is maintaining a numerical ocean forecasting system constrained by computational costs (like ECMWF OCEAN5/ORAS6 with only 5-11 ensemble members at 0.25° resolution), FuXi-ONS demonstrates a path to dramatically increase ensemble size (24 members) while reducing computational requirements, running on just 8 NVIDIA A100 GPUs versus millions of CPU hours for traditional systems. For climate risk assessment applications needing probabilistic outlooks (e.g., for marine heatwaves or ENSO events), this translates to being able to offer more nuanced uncertainty estimates without requiring a complete infrastructure overhaul. Crucially, the atmospheric encoding module provides a template for handling long-range atmospheric forcing errors, a common pain point in ocean forecasting systems that currently requires expensive numerical adjustments.

## Problem Statement
Today's ocean forecasting systems face a similar challenge to weather forecasting 30 years ago, trying to predict chaotic systems with limited computational resources. Imagine trying to predict a city's traffic flow using only a single car's route: it might be accurate for the first few minutes, but as small variations in initial conditions (like a driver changing lanes) compound, the forecast rapidly diverges from reality. Similarly, current numerical ocean models can't maintain enough ensemble members to adequately sample the ocean's complex uncertainty landscape, forcing a trade-off between resolution and ensemble size. The result is that operational systems like the UK Met Office FOAM (running at 1/12° resolution) can only afford 5-11 ensemble members, making them unable to properly characterise the full uncertainty of ocean states over multi-month lead times.

## Proposed Approach
FuXi-ONS operates as a three-component system: (1) a deterministic backbone for ocean state forecasting, (2) a learned perturbation module that generates physically structured ensemble variations (rather than using generic noise), and (3) an atmospheric encoding module that stabilizes long-range forecasts by accounting for atmospheric forcing errors. The system learns directly from historical ocean reanalysis data, producing 24 ensemble members that provide both a more accurate central forecast and a more realistic characterisation of uncertainty. Unlike traditional ensemble systems that require multiple full model integrations, FuXi-ONS generates all 24 ensemble members in a single forward pass on a GPU.

```python
def fu_xi_ons_ensemble_forecast(initial_conditions, atmospheric_forcing):
    # 1. Deterministic backbone forecast
    deterministic_forecast = deterministic_model(initial_conditions)
    
    # 2. Learn physical perturbations
    physical_perturbations = perturbation_model(deterministic_forecast)
    
    # 3. Generate ensemble members
    ensemble_members = []
    for i in range(24):
        ensemble_member = deterministic_forecast + physical_perturbations[i]
        ensemble_members.append(ensemble_member)
    
    # 4. Atmospheric encoding for stability
    stabilized_forecast = atmospheric_encoder(ensemble_members, atmospheric_forcing)
    
    return stabilized_forecast
```

## Key Technical Contributions
FuXi-ONS advances ocean forecasting through four specific technical innovations:

1. **Physically structured perturbation learning**: The system avoids generic noise-based perturbation strategies (like Perlin noise) by learning how to generate variations that respect the physical structure of ocean uncertainty. The perturbation module produces variations that evolve consistently with ocean dynamics, as evidenced by the vertical coherence in Fig. 2 where improvements extend through the water column rather than being confined to the surface layer. This is achieved through a neural network trained on historical reanalysis data to capture how uncertainty varies across depth, variables, and lead times.

2. **Atmospheric encoding module**: This component addresses the challenge of long-range atmospheric forcing errors by encoding atmospheric states into the forecast process. By incorporating atmospheric information directly into the model, FuXi-ONS reduces sensitivity to atmospheric errors that would otherwise degrade ocean forecasts at extended lead times. The module processes atmospheric data through a separate transformer layer that modifies the ensemble spread based on atmospheric conditions.

3. **Vertical structure preservation**: Unlike surface-focused probabilistic forecasts, FuXi-ONS maintains uncertainty representation across all depths. The system uses a multi-variable, multi-depth architecture that ensures the ensemble spread remains consistent with actual forecast uncertainty throughout the water column (Fig. 2). For example, the improvement in CRPS for subsurface temperature (Fig. 2) demonstrates that the system captures vertical uncertainty patterns that generic perturbation methods miss.

4. **Computational efficiency at scale**: By learning the ensemble generation process directly from historical data, FuXi-ONS eliminates the need for multiple numerical integrations. The system runs inference on 8 NVIDIA A100 GPUs to generate 24 ensemble members, achieving an order-of-magnitude speedup compared to traditional numerical ensemble systems. As noted in the paper, "FuXi-ONS improves both ensemble-mean skill and probabilistic forecast quality relative to deterministic and noise-perturbed baselines, and shows competitive performance against established seasonal forecast references... while running orders of magnitude faster than conventional ensemble systems."

## Experimental Results
FuXi-ONS demonstrated consistent improvements across multiple metrics. In deterministic evaluation (RMSE and ACC), FuXi-ONS achieved lower RMSE (0.03-0.10 across variables) and higher ACC (0.25-1.00) compared to baselines across the full annual forecast range. For probabilistic evaluation (CRPS and SSR), FuXi-ONS consistently outperformed both the deterministic baseline (FuXi-Aim) and the noise-perturbed baseline (FuXi-Aim-Perlin), with CRPS values of 0.03-0.10 and SSR values of 0.08-0.32. When compared to established seasonal forecast references like NMME, FuXi-ONS showed competitive performance in Niño3.4 index prediction (Fig. 3) and better SST forecast skill (Fig. 4), with lower RMSE (0.40-1.60) and CRPS (0.00-0.90) than NMME. The paper notes that "the advantage of FuXi-ONS is not limited to a scalar climate index, but is also reflected in the monthly SST field from which the Niño3.4 anomaly is derived."

## Related Work
FuXi-ONS builds upon recent advances in data-driven deterministic ocean forecasting but extends to the probabilistic domain, addressing a gap that has been identified in the literature. Unlike traditional numerical ensemble systems that rely on repeated model integrations (e.g., ECMWF OCEAN5/ORAS6 with 5-11 members), FuXi-ONS learns ensemble generation directly from data. It also improves upon atmospheric ensemble techniques adapted for ocean use, which typically suffer from generic perturbations that don't capture the ocean's vertical and multivariate uncertainty structure. The work demonstrates how machine learning can be extended from deterministic prediction to ensemble forecasting in the ocean, following recent developments in atmospheric ensemble AI systems (e.g., [31-33]).

## Limitations
The authors acknowledge that FuXi-ONS remains under-dispersive (SSR values below the ideal), indicating that the ensemble spread is still somewhat smaller than actual forecast uncertainty. The system was evaluated against GLORYS12 reanalysis but not against operational numerical models with the same resolution and ensemble size. The paper doesn't specify whether the improvements generalise to extreme climate events or under changing climate conditions. Additionally, the atmospheric encoding module was tested on short-to-medium lead times but may need further refinement for very long-range forecasts (e.g., beyond 365 days).

## Appendix: Worked Example
Let's walk through a simplified version of how FuXi-ONS processes a single ocean state forecast for sea-surface temperature (SST) at a 1° grid resolution with a 5-day forecast lead time:

1. **Input**: The system receives initial ocean state (including SST, salinity, currents) at a global 1° grid, along with atmospheric forcing data (temperature, wind, etc.) at the same resolution for the target forecast period.

2. **Deterministic backbone**: The deterministic model processes this input to generate a baseline SST forecast. For a specific location (e.g., 30°N, 150°W), the deterministic forecast predicts an SST of 22.4°C with an RMSE of 0.08°C compared to the reanalysis.

3. **Perturbation generation**: The perturbation module generates 24 physically structured variations of the deterministic forecast. For the same location, perturbations add variations ranging from -0.3°C to +0.5°C (based on historical patterns of SST uncertainty at that location and depth).

4. **Ensemble member formation**: The 24 ensemble members are formed by adding these perturbations to the deterministic forecast, resulting in a range of SST predictions from 22.1°C to 22.9°C.

5. **Atmospheric encoding**: The atmospheric module adjusts the ensemble based on atmospheric forcing. If the atmospheric data suggests stronger than usual wind patterns (e.g., 20% increase in wind speed), it increases the spread in the ensemble members by 10% to account for potential atmospheric impacts.

6. **Probabilistic output**: The final output is a probability distribution (mean 22.4°C, standard deviation 0.2°C) that better reflects the true uncertainty than a single deterministic forecast or generic noise-based perturbations. For this location, the CRPS is 0.05 (compared to 0.08 for FuXi-Aim-Perlin), indicating a more accurate probabilistic forecast.

## References

- Qiusheng Huang, Xiaohui Zhong, Anboyu Guo, Ziyi Peng, Lei Chen, Hao Li, "Data-driven ensemble prediction of the global ocean", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19591

Tags: #ocean-science #probabilistic-forecasting #machine-learning #uncertainty-quantification #ensemble-learning
