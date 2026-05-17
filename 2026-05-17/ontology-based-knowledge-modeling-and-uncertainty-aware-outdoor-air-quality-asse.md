---
title: "Ontology-Based Knowledge Modeling and Uncertainty-Aware Outdoor Air Quality Assessment Using Weighted Interval Type-2 Fuzzy Logic"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19683"
---

## Executive Summary
This paper proposes a hybrid ontology-based framework that integrates weighted interval Type-2 fuzzy logic with semantic knowledge modelling to handle uncertainty in outdoor air quality assessment. The approach explicitly models uncertainty near AQI class boundaries and incorporates health impact-based pollutant weights, resulting in more reliable and interpretable AQI classifications compared to traditional methods that use rigid thresholds.

## Why This Matters for Practitioners
If you're building or maintaining air quality monitoring systems in urban environments, this paper challenges the fundamental assumption that crisp thresholds are sufficient for real-world air quality assessment. Traditional AQI calculations force decisions at boundaries where the data is inherently uncertain, potentially leading to misleading public health advisories. For instance, when PM2.5 levels are near the threshold between "Moderate" and "Poor" categories, your system might incorrectly classify the air quality as "Poor" due to sensor noise, triggering unnecessary public alerts. The authors' framework provides a concrete methodology to incorporate this uncertainty directly into your decision pipeline, reducing false alarms while maintaining sensitivity to actual risk increases. Engineers should reconsider how they handle boundary conditions in any system where numerical thresholds are used to make real-world decisions, especially in health-critical applications where the cost of false positives or negatives is high.

## Problem Statement
Traditional AQI calculation is like forcing a patient with a fever to choose between "healthy" or "sick" without considering the gradual progression from mild to severe fever. When pollutant concentrations approach category boundaries (e.g., PM2.5 at 35 μg/m³ between "Moderate" and "Poor" classifications), the system can't capture the natural uncertainty of real-world environmental conditions, resulting in misleading binary classifications that don't reflect the true nature of the data. This becomes especially problematic in rapidly urbanizing areas where air quality fluctuates near critical thresholds.

## Proposed Approach
The framework combines four core components: data preprocessing, interval Type-2 fuzzy inference, weighted rule firing, and ontology-based semantic reasoning. Raw data from monitoring stations undergoes preprocessing to handle missing values before being transformed into interval membership values using Type-2 fuzzy sets. These interval values then feed into a rule base where each rule's activation strength is weighted by pollutant health impact. Finally, the results are semantically represented in a knowledge graph enabling explainable reasoning about air quality classifications.

```python
def air_quality_assessment(pollutant_data):
    # Preprocess data (handle missing values, normalize to CPCB standards)
    clean_data = preprocess(pollutant_data)
    
    # Fuzzify using interval Type-2 membership functions
    fuzzy_data = fuzzify_interval_type2(clean_data)
    
    # Generate fuzzy rules based on pollutant concentrations
    rules = generate_rules(fuzzy_data)
    
    # Calculate pollutant weights using IT2-FAHP
    weights = it2_fahp_weights(pollutant_data)
    
    # Apply weighted rule firing
    aqi_class = weighted_rule_firing(fuzzy_data, rules, weights)
    
    # Store results in ontology for semantic reasoning
    store_in_ontology(aqi_class, pollutant_data)
    
    return aqi_class
```

## Key Technical Contributions
The paper introduces several novel mechanisms for uncertainty-aware air quality assessment:

1. **Interval Type-2 Fuzzy Sets with Explicit Footprint of Uncertainty**: Unlike Type-1 fuzzy sets that use a single membership function, the authors model uncertainty through a region between upper and lower membership functions (UMF and LMF). For example, PM2.5 at 35 μg/m³ has a membership of [0.6, 0.9] for "Moderate" and [0.8, 0.95] for "Poor" (based on CPCB breakpoints), explicitly capturing the uncertainty around the boundary. This region, the Footprint of Uncertainty (FOU), enables the system to recognise when sensor noise or spatiotemporal variability might cause misclassification.

2. **Health-Impact-Based Pollutant Weighting via IT2-FAHP**: The authors implement a novel application of Interval Type-2 Fuzzy Analytic Hierarchy Process (IT2-FAHP) to determine pollutant importance weights based on epidemiological evidence. They establish a dominance order: PM2.5 > PM10 > CO > O3 > NO2 > SO2 > NH3, which is then used to weight rule firings. This differs from traditional fuzzy systems that assume equal influence for all pollutants during inference, resulting in more health-relevant AQI assessments.

3. **OWL-Based Ontology with SWRL Reasoning for Explainability**: The framework extends the Semantic Sensor Network (SSN) ontology to create a comprehensive air quality knowledge representation. This includes formal definitions for pollutants, monitoring stations, AQI categories, and regulatory standards. Semantic reasoning using SWRL rules enables the system to infer not just AQI categories but also health risks and recommended mitigation actions, providing explainable decision support that's missing in purely numerical approaches.

4. **Integration of Uncertainty Modelling with Semantic Reasoning**: The framework uniquely combines the uncertainty modelling from the fuzzy logic component with the semantic reasoning capabilities of the ontology. For example, when the system classifies air quality as "Moderate," it can simultaneously infer "PM2.5 levels are near the boundary between Moderate and Poor categories, with elevated NO2 contributing to the classification," providing transparency that's absent in black-box machine learning approaches.

## Experimental Results
The paper states that their framework "improves AQI classification reliability and uncertainty handling compared with traditional crisp and Type-1 fuzzy approaches" using CPCB air quality datasets. However, the provided text does not include specific numerical results (such as accuracy improvements, F1 scores, or statistical significance measures) that would allow for quantitative comparison with baseline methods. The authors do not report the exact dataset size or the specific baselines used for comparison.

## Related Work
This work builds on three key research threads: fuzzy logic-based air quality assessment (which was shown to be more flexible than traditional crisp methods), Type-2 fuzzy inference systems (which handle uncertainty in membership functions), and ontology-based semantic modelling (for structured knowledge representation). The authors position their contribution as the first to combine all three elements, Interval Type-2 fuzzy logic, health-impact-based weighting, and ontology-based semantic reasoning, into a cohesive framework for air quality assessment. This differs from prior work that either used Type-1 fuzzy systems without uncertainty modelling or semantic approaches without weighted reasoning.

## Limitations
The authors don't explicitly state limitations in the provided excerpt, but the paper focuses exclusively on CPCB air quality data from Indian monitoring stations, which suggests the framework may not directly generalise to other regulatory frameworks or pollutant profiles. The paper also doesn't discuss computational complexity or scalability for real-time monitoring systems with high data throughput requirements, which would be critical for production deployment. Additionally, while the ontology enables explainable reasoning, the paper doesn't evaluate whether these explanations improve decision-making for public health officials or end-users.

## Appendix: Worked Example
Let's walk through a concrete example of the framework's operation using actual values from CPCB standards:

Consider a monitoring station measuring:
- PM2.5: 35 μg/m³
- PM10: 70 μg/m³
- NO2: 35 ppb
- SO2: 25 ppb
- O3: 60 ppb
- CO: 1.5 ppm
- NH3: 0.5 ppm

**Step 1: Fuzzification with Interval Type-2 Membership Functions**
For PM2.5 at 35 μg/m³ (CPCB "Moderate" range is 30-60 μg/m³), the interval Type-2 membership function yields:
- Moderate: [0.6, 0.9] (UMF: [0.7, 0.9], LMF: [0.5, 0.8])
- Poor: [0.8, 0.95] (UMF: [0.85, 0.95], LMF: [0.75, 0.85])

This interval explicitly captures the uncertainty around the boundary between Moderate and Poor categories.

**Step 2: Weight Calculation using IT2-FAHP**
The health impact weights are:
- PM2.5: 0.35
- PM10: 0.25
- NO2: 0.15
- SO2: 0.10
- O3: 0.10
- CO: 0.05
- NH3: 0.05

These weights reflect epidemiological evidence of relative health impacts.

**Step 3: Rule Evaluation with Weighted Firing**
Consider the rule: "IF PM2.5 is Moderate AND PM10 is Satisfactory AND NO2 is Good THEN AQI is Moderate."

The weighted firing strength for this rule is:
0.35 (PM2.5 weight) × 0.75 (PM2.5 Moderate membership) + 0.25 (PM10 weight) × 0.6 (PM10 Satisfactory membership) + 0.15 (NO2 weight) × 0.8 (NO2 Good membership) = 0.3875

**Step 4: AQI Classification**
After evaluating all applicable rules and aggregating results, the system determines the AQI class as "Moderate" with a confidence interval reflecting the uncertainty (e.g., [0.7, 0.85]). The ontology layer then infers "PM2.5 levels are near the boundary between Moderate and Poor categories, with elevated NO2 contributing to the classification," providing explainable reasoning that supports public health decision-making.

## References

- Md Inzmam, Ritesh Chandra, Sadhana Tiwari, Sonali Agarwal, Triloki Pant, "Ontology-Based Knowledge Modeling and Uncertainty-Aware Outdoor Air Quality Assessment Using Weighted Interval Type-2 Fuzzy Logic", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19683

Tags: #urban-environment #fuzzy-logic #knowledge-representation #decision-support #health-impact-analysis
