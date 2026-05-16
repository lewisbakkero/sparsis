---
title: "Unmasking Algorithmic Bias in Predictive Policing: A GAN-Based Simulation Framework with Multi-City Temporal Analysis"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.18987"
---

## Executive Summary

This paper introduces a GAN-based simulation framework that quantifies how racial bias propagates through the full predictive policing pipeline, from crime occurrence to police contact, using real crime data from Baltimore (2017-2019) and Chicago (2022). The framework reveals extreme year-variant bias in algorithmically directed patrols (mean annual DIR up to 15,714), demonstrating that historical enforcement patterns become embedded in predictive models and amplify through self-reinforcing feedback loops. Engineers building similar systems should implement regular bias audits to prevent these compounding disparities.

## Why This Matters for Practitioners

If you're building or maintaining predictive policing systems that use historical crime data to allocate patrol resources, this paper shows that even small historical biases can amplify into extreme racial disparities, Baltimore 2019's detected mode showed a mean annual Disparate Impact Ratio (DIR) of 15,714, meaning Black residents were detected at 15,714 times the rate of White residents. The sensitivity analysis proves that officer deployment levels have the largest effect on bias metrics (reducing from 60 to 30 officers more than doubled DIR from 0.084 to 7.713), so you should consider varying patrol budgets during system design. CTGAN debiasing can't eliminate structural disparity without policy intervention, it merely changes the direction of bias (DIR shifted from 0.513 to 3.106), so engineers should partner with sociologists and community representatives before deploying any fairness mitigation. For production systems, implement the same four bias metrics (DIR, Demographic Parity Gap, Gini Coefficient, and Bias Amplification Score) monthly to monitor for compounding bias.

## Problem Statement

Predictive policing systems today are like a broken mirror: they reflect historical enforcement patterns, amplify them through self-reinforcing feedback loops, and present the distorted results as objective crime forecasts. The system in Baltimore 2019 didn't just reflect bias, it created a feedback loop where the GAN learned to concentrate patrols in Black neighbourhoods (due to historical over-policing), which then created more detection events in those areas, feeding back into the training data. This isn't a theoretical concern; it's a practical engineering failure that compounds with each deployment cycle.

## Proposed Approach

The authors built a simulation framework coupling a Generative Adversarial Network with a Noisy-OR patrol-detection model to measure bias propagation through the full enforcement pipeline. This framework uses historical crime data to train a GAN that generates synthetic patrol locations, then applies the Noisy-OR model to compute detection probabilities based on patrol location and crime event. The system computes four bias metrics (DIR, Demographic Parity Gap, Gini Coefficient, and Bias Amplification Score) across multiple city-year-mode combinations.

```python
# GAN-Based Patrol Location Generation
def generate_patrol_locations(crime_data, num_officers=60):
    # Train GAN on crime incident coordinates
    generator = build_generator(latent_dim=100)  # 5-layer generator
    discriminator = build_discriminator()  # 4-layer discriminator
    
    # Train for 200 epochs with Adam optimizer (lr=0.0002)
    train_gan(generator, discriminator, crime_data, epochs=200)
    
    # Generate patrol locations (60 locations per simulation step)
    noise = np.random.normal(size=(num_officers, 100))
    return generator.predict(noise)
```

## Key Technical Contributions

The paper introduces a novel simulation framework that quantifies bias propagation through the full policing pipeline. Key contributions include:

1. **GAN-based spatial patrol model with calibrated detection parameters**: The authors train a 5-layer generator and 4-layer discriminator on real crime-incident coordinates, using a Noisy-OR detection model with realistic parameters (detection radius = 700 ft, per-officer detection probability = 0.85). This calibration ensures the simulation reflects real-world patrol dynamics rather than artificial bias.

2. **Longitudinal multi-city bias audit across 264 city-year-mode observations**: The framework computes bias metrics across 3 Baltimore years (2017-2019) × 2 modes (detected vs. reported) × 11 months + 1 Chicago year (2022) × 2 modes × 11 months = 264 simulation runs. This captures temporal trends that single-city or single-year studies miss, Baltimore 2019 showed extreme year-variant bias (mean DIR 15,714) whereas reported mode remained stable (mean DIR 0.61-1.22).

3. **CTGAN debiasing with structural insight**: The authors adopt CTGAN to rebalance the training data by generating equal proportions of incidents from each racial group, replacing 30% of real training incidents. Critically, they show this approach changes the direction of disparity (DIR from 0.513 to 3.106) rather than equalising detection rates, demonstrating that algorithmic debiasing cannot eliminate structural disparity without policy intervention.

## Experimental Results

The experiments reveal extreme bias in Baltimore's detected mode (mean annual DIR up to 15,714 in 2019), moderate under-detection of Black residents in Chicago (DIR = 0.22), and persistent Gini coefficients of 0.43-0.62 across all conditions. Baltimore 2019 detected mode showed a mean DIR of 15,714 (10 of 11 months above 1.0), driven by near-zero White detection rates as the GAN concentrated patrols away from White-majority areas. The sensitivity analysis showed officer count had the largest effect on bias metrics (reducing from 60 to 30 officers more than doubled DIR from 0.084 to 7.713), with patrol radius having a monotonic effect (increasing from 400 to 1500 ft increased DIR from 0.045 to 0.227) and citizen reporting probability having a non-monotonic relationship. CTGAN debiasing increased Black detection rates by +1.49 percentage points but decreased White detection rates by -5.11 percentage points, reversing the direction of disparity (DIR from 0.513 to 3.106).

## Related Work

This paper extends foundational critiques of predictive policing (Ensign et al. [7]) by moving beyond theoretical feedback loops to quantify bias propagation through the full enforcement pipeline. It improves upon Wu and Frias-Martinez [21] and Wang et al. [20] by modelling the full path from crime occurrence to police contact rather than focusing on crime prediction alone. The authors also build upon CTGAN [22] for debiasing but demonstrate its limitations in addressing structural disparity without policy intervention, unlike Ma et al. [11] who demonstrated that generative rebalancing can reduce disparate outcomes without dramatically degrading accuracy.

## Limitations

The paper only examines two cities (Baltimore and Chicago) across limited time periods, so results may not generalise to other cities with different demographic structures. The simulation assumes a fixed patrol budget (60 officers), which may not reflect real-world resource constraints. The authors acknowledge that CTGAN debiasing cannot eliminate structural disparity without accompanying policy intervention, an important limitation for engineers considering implementation. The study also relies on Part 1 crime data, which may not capture the full complexity of policing dynamics.

## Appendix: Worked Example

Let's walk through a concrete example from Baltimore 2019 detected mode:

1. **Start with the training data**: The GAN is trained on 47,822 crime incidents from Baltimore 2019 (after excluding January for GAN burn-in).
2. **Generate patrol locations**: The GAN generates 60 patrol locations per simulation step (month). In 2019, the GAN concentrated patrols in majority-Black neighbourhoods.
3. **Compute detection probabilities**: For a crime event in a Black-majority area with 1 officer within 700 ft (detection radius), the Noisy-OR model calculates:
   - P(detected) = 1 - (1 - 0.85) = 0.85
4. **Compute detection rates**: The biased condition shows:
   - Black detection rate = 3.44%
   - White detection rate = 6.70%
   - DIR = 3.44% / 6.70% = 0.513
5. **Apply CTGAN debiasing**: Replace 30% of training incidents with CTGAN-generated incidents in equal proportions across racial groups:
   - Black detection rate = 4.93% (+1.49 percentage points)
   - White detection rate = 1.59% (-5.11 percentage points)
   - DIR = 4.93% / 1.59% = 3.106
6. **Result**: The bias direction reversed (from under-detection of Black residents to over-detection), demonstrating that algorithmic debiasing alone cannot eliminate structural disparity.

## References

- Pronob Kumar Barman, Pronoy Kumar Barman, "Unmasking Algorithmic Bias in Predictive Policing: A GAN-Based Simulation Framework with Multi-City Temporal Analysis", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.18987

Tags: #public-policy #algorithmic-fairness #generative-models #predictive-policing #bias-audit
