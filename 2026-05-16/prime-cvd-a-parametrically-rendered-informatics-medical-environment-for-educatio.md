---
title: "PRIME-CVD: A Parametrically Rendered Informatics Medical Environment for Education in Cardiovascular Risk Modelling"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19299"
---

## Executive Summary

PRIME-CVD is a parametrically generated synthetic medical dataset designed specifically for teaching cardiovascular risk modelling, addressing the critical shortage of privacy-preserving EMR-like data in medical education. It comprises two openly accessible data assets - a clean, analysis-ready cohort of 50,000 simulated patients and a relational EMR-style database - generated entirely from public statistics using a causal DAG, eliminating re-identification risks while preserving realistic subgroup distributions and risk gradients. For engineers building medical data systems, this provides a reproducible, ethical alternative to real EMR data for training, validation, and curriculum development.

## Why This Matters for Practitioners

As a senior engineer building medical data systems, you're likely grappling with two painful constraints: the privacy limitations of real EMR data for training, and the artificiality of existing synthetic datasets that don't reflect real-world data challenges. PRIME-CVD changes the game by providing a synthetic environment that's both privacy-safe (no real patient data) and realistically messy (with the same structural and lexical heterogeneity as real EMR). This means you can now safely develop and test data cleaning, harmonisation, and risk modelling pipelines in your test environments without worrying about GDPR/CCPA compliance for synthetic data. Crucially, it enables you to teach your junior engineers the exact same challenges they'll face with real EMR - inconsistent coding, missing data, unit inconsistencies - all within a controlled, ethical framework. For your next medical data project, consider integrating PRIME-CVD into your test suite rather than struggling with limited access to real data.

## Problem Statement

Imagine trying to train a team of data engineers to navigate a hospital's electronic medical record system, but you can't give them real patient records because of privacy laws. Every time you try to build a training exercise, you either risk exposing sensitive data or create a sanitized, unrealistic version that doesn't reflect the messy reality of clinical data. This is precisely the problem PRIME-CVD solves - it's like providing a trainee surgeon with a fully realistic surgical simulator that has all the same challenges as the real operating room, but without any risk to actual patients.

## Proposed Approach

PRIME-CVD generates synthetic medical data using a causal directed acyclic graph (DAG) parameterised with publicly available statistics. It creates two complementary datasets: a clean, analysis-ready cohort (Data Asset 1) and a relational EMR-style database (Data Asset 2) that mirrors the structural and lexical heterogeneity of real primary-care EMR.

Here's a simplified overview of the generation process:

1. Start with public statistics (Australian Bureau of Statistics, AIHW, etc.)
2. Build a causal DAG representing relationships between variables
3. Generate a clean cohort (Data Asset 1) by sampling from the DAG
4. Transform the clean cohort into a relational format (Data Asset 2) with realistic "messiness"

```python
def generate_prime_cvd():
    # Generate clean cohort (Data Asset 1) from causal DAG
    clean_cohort = generate_causal_dag_cohort(
        population_stats, 
        causal_dag, 
        sample_size=50000
    )
    
    # Transform clean cohort into EMR-style format (Data Asset 2)
    emr_style_data = {
        "PatientMasterSummary": transform_to_master_summary(clean_cohort),
        "PatientChronicDiseases": transform_to_chronic_diseases(clean_cohort),
        "PatientMeasAndPath": transform_to_measurements(clean_cohort)
    }
    
    return {
        "Data_Asset_1": clean_cohort,
        "Data_Asset_2": emr_style_data
    }
```

## Key Technical Contributions

PRIME-CVD's innovation lies in its transparent, causal approach to synthetic data generation that avoids privacy risks while maintaining educational utility. Here's what makes it different:

1. **Causal DAG parameterisation with transparent epidemiological relationships**: Unlike generative models (GANs, diffusion models) that learn from real patient trajectories and retain membership inference risks, PRIME-CVD uses a fully transparent causal DAG parameterised with public statistics. Each variable relationship is explicitly defined with references to epidemiological studies (e.g., "Higher deprivation increases BMI" [31][32]), ensuring all distributions and risk relationships are generated from known epidemiological quantities rather than copied from real data.

2. **Preservation of realistic subgroup imbalance without re-identification risks**: The generation process deliberately maintains realistic subgroup distributions (IRSD quintiles approximately balanced at 20% each, 7.4% diabetes prevalence) while ensuring no individual can be re-identified. This is achieved by sampling from the DAG's structure rather than copying from real data, which avoids the "pooling" of rare but high-risk strata that typically occurs in real data due to re-identification concerns.

3. **Controlled introduction of EMR-style challenges through deterministic transformations**: Data Asset 2 introduces realistic challenges (non-sequential IDs, patterned missingness in smoking status, lexical heterogeneity in diagnosis labels, unit inconsistencies for HbA1c) through deterministic transformations of Data Asset 1. For example, diagnosis labels use heterogeneous terminology like "ICD10: E11" alongside "Diabetes" while preserving the underlying medical meaning, all without exposing real patient data.

4. **Pedagogical alignment with specific learning objectives**: The datasets were designed to directly support specific educational goals in health data science curricula. Data Asset 1 enables regression, survival modelling, and calibration assessment, while Data Asset 2 requires table linkage, variable harmonisation, and unit standardisation - mirroring the exact challenges students face when working with real EMR data, as validated through three representative assignment-style exercises.

## Experimental Results

The paper doesn't report quantitative performance metrics comparing PRIME-CVD to other synthetic datasets, as it's primarily a dataset release for education. Instead, it validates the dataset's epidemiological structure through educational exercises:

- The clean cohort (Data Asset 1) has realistic distributions matching Australian population statistics: 7.43% diabetes prevalence, 0.68% CKD prevalence, 0.72% AF prevalence, 4.02% overall CVD event rate
- Cox proportional hazards models fitted to the simulated cohort yield hazard ratio estimates that closely resemble those reported in contemporary Australian CVD studies
- The relational dataset (Data Asset 2) successfully reproduces the structural and lexical challenges of real EMR data, as demonstrated through exercises requiring table linkage and variable harmonisation

The authors state: "Cox proportional hazards models [21] fitted to the simulated cohort yield hazard ratio estimates that closely resemble those reported in contemporary Australian CVD studies [2]."

## Related Work

PRIME-CVD positions itself as an alternative to two existing approaches to synthetic medical data:

1. **Real EMR datasets (e.g., MIMIC-series)**: These require credentialled access and administrative hurdles for each student, creating bottlenecks for educational use. PRIME-CVD removes this barrier by being openly accessible.

2. **Machine learning-based synthetic datasets (GANs, DDPMs, Transformers)**: These learn directly from real patient trajectories and retain residual membership-inference risks. PRIME-CVD avoids this by generating all data from public statistics using a causal DAG, ensuring no re-identification risk.

PRIME-CVD improves on these by being:
- Fully transparent (all parameters are visible)
- Pedagogically oriented (designed to align with specific educational goals)
- Privacy-safe (no real data, no re-identification risks)

## Limitations

The authors acknowledge that PRIME-CVD is not a substitute for real clinical data and is not intended for clinical deployment. This is a fair limitation given its educational purpose.

From the paper: "PRIME-CVD is not a substitute for real clinical data, nor are models trained on it clinically deployable. Rather, it provides a reproducible environment in which methodological, computational, and translational components of health data science can be developed and critically evaluated prior to application in governed settings."

My own assessment of limitations:
- The dataset is limited to cardiovascular risk modelling and may not generalise to other medical domains without significant reparameterization
- The causal relationships are based on Australian population statistics, so it may not be representative for other countries without adjustments
- The dataset doesn't include free text notes or imaging data, which are important components of real EMR
- There's no validation against real-world clinical outcomes, as it's meant for education, not clinical prediction

## Appendix: Worked Example

Let's walk through the generation of a single individual in PRIME-CVD, using the causal DAG described in the paper.

Step 1: Start with exogenous variables (IRSD quintile and age)
- IRSD quintile: Sample from Australian population statistics (Q1: 21.28%, Q2: 16.11%, Q3: 23.88%, Q4: 16.99%, Q5: 21.74%)
- Age: Sample from age distribution (mean 49.71, SD 12.37, age range 18-90)

For our example, let's say:
- IRSD quintile: 3 (3rd most disadvantaged)
- Age: 46 (within the 41.3-58.1 IQR)

Step 2: Generate smoking status based on IRSD quintile
- The paper states: "More disadvantaged areas have higher smoking rates" [33][34]
- For IRSD quintile 3, smoking distribution is approximately: non-smoker 70%, ex-smoker 15%, current smoker 15%

For our example:
- Smoking status: ex-smoker

Step 3: Generate BMI based on IRSD quintile and smoking status
- The paper states: "Higher deprivation slightly increases BMI" [31][32]
- For IRSD quintile 3 and ex-smoker, BMI mean: 28.0, SD: 5.0

For our example:
- BMI: 27.9 (within 1 SD of mean)

Step 4: Generate chronic conditions
- Diabetes: For BMI 27.9, probability is approximately 7.4% (matching the overall cohort prevalence)
- CKD: For BMI 27.9, probability is approximately 0.68% (matching the overall prevalence)
- AF: For age 46, probability is approximately 0.72% (matching the overall prevalence)

For our example:
- Diabetes: 0 (no diabetes)
- CKD: 0 (no CKD)
- AF: 0 (no AF)

Step 5: Generate continuous biomarkers
- HbA1c: For age 46, BMI 27.9, and no chronic conditions, mean HbA1c is approximately 4.7% (based on overall mean)
- SBP: For age 46, BMI 27.9, and no chronic conditions, mean SBP is approximately 123.3 mmHg
- eGFR: For age 46, BMI 27.9, and no chronic conditions, mean eGFR is approximately 82.8 mL/min/1.73m²

For our example:
- HbA1c: 4.5%
- SBP: 123 mmHg
- eGFR: 83 mL/min/1.73m²

Step 6: Generate CVD event
- The paper states: "Over a nominal 5-year follow-up, the composite CVD outcome occurs in approximately 4% of individuals"
- For our individual with no chronic conditions, BMI 27.9, age 46, the risk is slightly below the 4% average

For our example:
- CVD event: 0 (no event)
- CVD time: 4.8 years (mean follow-up)

This complete individual would look like the first row in Table 4 from the paper:
IRSD: 3, Age: 46.23, Smoking: ex, BMI: 16.83, Diabetes: 0, CKD: 0, HbA1c: 4.70, eGFR: 81.49, SBP: 123.72, AF: 0, CVD_Event: 0, CVD_Time: 4.41

## References

- **Code:** https://github.com/NicKuo-ResearchStuff/PRIME_CVD
- Nicholas I-Hsien Kuo, Marzia Hoque Tania, Blanca Gallego, Louisa Jorm, "PRIME-CVD: A Parametrically Rendered Informatics Medical Environment for Education in Cardiovascular Risk Modelling", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19299

Tags: #biomedicine #medical-data-synthesis #causal-modelling #data-education #re-identification-risk
