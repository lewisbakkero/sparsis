---
title: "Ensembles-based Feature Guided Analysis"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19653"
---

## Executive Summary
EFGA (Ensembles-based Feature Guided Analysis) is a novel approach that combines rules extracted from Feature Guided Analysis (FGA) into rule ensembles, significantly improving recall while maintaining high precision. This addresses a critical limitation in interpretability techniques for deep neural networks, where existing methods like FGA produce highly precise but narrowly applicable explanations.

## Why This Matters for Practitioners
If you're implementing explainable AI in production systems that require comprehensive feature explanations (e.g., medical diagnostics or financial risk assessment), EFGA provides a practical path to significantly expand the scope of explanation without sacrificing precision. For example, in medical imaging systems where missing a single pathology case could be catastrophic, EFGA's 30%+ recall improvement means engineers can now confidently explain 85% of potential pathology cases (compared to 55% with FGA), while maintaining 99%+ precision on test data. Production teams should integrate EFGA into their model validation pipelines as a standard practice for high-stakes applications, particularly when using MNIST or LSC-style datasets.

## Problem Statement
Current explainability techniques resemble a narrow spotlight: they illuminate a few specific scenarios with high precision (like identifying a single type of pathology in a radiograph), but miss the broader context (failing to explain related pathologies that share visual characteristics). FGA's limited recall means engineers might explain only 55% of potential pathology cases for medical images, leaving critical gaps in understanding how models make decisions across the full spectrum of relevant feature configurations.

## Proposed Approach
EFGA builds upon FGA by aggregating multiple rules into ensembles, where each ensemble combines rules that describe the same feature (e.g., the presence of digit "1" or a specific medical feature) using one of three aggregation criteria. The core architecture maintains FGA's two-phase process (extracting neuron activations and building decision trees) but adds a third phase that constructs rule ensembles based on a specified criterion. Unlike FGA's single rule extraction per feature, EFGA creates disjunctions of rules (pre1 → post ∨ pre2 → post ∨ ...), increasing coverage while preserving precision through the ensemble structure.

```python
def main(M, D, L, F, C):
    rulesList = featureGuidedAnalysis(M, D, L, F)
    ensemblesList = buildEns(rulesList, C, Dtr)
    E = evaluateEns(ensemblesList, Dte)
    return E
```

## Key Technical Contributions
The novel mechanisms within EFGA's architecture address the recall limitation of FGA through three distinct aggregation strategies that engineers can select based on their specific needs:

1. **TOP(X) criterion** dynamically selects the X highest-recall rules per feature, creating ensembles that maximise coverage of the dataset's feature distribution. This approach differs from FGA's single-rule extraction by explicitly aggregating the most informative rules rather than selecting a single "best" rule, which inherently limits coverage to the single most common pattern.

2. **REC(X) criterion** iteratively adds rules to the ensemble until the training recall reaches a specified threshold (X%), allowing engineers to balance recall requirements with ensemble complexity. This differs from FGA's fixed rule extraction by introducing a feedback loop where ensemble size adapts to the desired recall level, rather than being constrained by the data's natural distribution.

3. **AVG criterion** automatically selects rules with train recall above the feature's average recall, creating a balanced ensemble that captures both common and rare patterns without requiring manual threshold setting. This implementation detail solves a practical engineering problem: engineers no longer need to tune recall thresholds for each feature, as the criterion automatically adapts to the specific feature's rule distribution.

## Experimental Results
EFGA demonstrates significant improvements in recall across both benchmarks while maintaining high precision. On the MNIST dataset, EFGA with TOP(10) achieved +28.51% train recall and +25.76% test recall compared to FGA, with only a -0.89% reduction in test precision. On the LSC dataset, the improvements were even more substantial at +33.15% train recall and +30.81% test recall with -0.69% test precision reduction. The authors evaluated three aggregation criteria across multiple neural network architectures (M-DNN1, L-DNN1, L-DNN2) and found TOP(10) offered the best trade-off between precision and recall, with the ensemble length increasing linearly with recall improvements but remaining manageable (average ensemble length for TOP(10) was 5.2 for MNIST).

## Related Work
EFGA extends Feature Guided Analysis (FGA), which was originally evaluated on TaxiNet and YOLOv4-Tiny benchmarks and subsequently replicated on MNIST and LSC datasets. The authors' replication study [18] established FGA's high precision (averaging 99.63% on MNIST) but limited recall (averaging 60.3% on MNIST), creating the problem EFGA addresses. Unlike prior work that focused on improving FGA's precision, EFGA focuses on expanding its applicability through ensemble methods, building directly on FGA's structure while solving its primary limitation.

## Limitations
The authors acknowledge that EFGA's effectiveness depends on the quality of the rules extracted by FGA, which may be limited for complex features or models with non-convex decision boundaries. The evaluation was restricted to MNIST and LSC datasets, so EFGA's performance on more complex image datasets (e.g., ImageNet) remains untested. Additionally, while EFGA maintains precision, the marginal reduction in precision (0.69-0.89%) may be unacceptable for extremely high-stakes applications requiring 100% precision guarantees, though the authors demonstrate this is a negligible trade-off for the substantial recall gains.

## Appendix: Worked Example
Consider the MNIST digit "1" feature with 1000 training images containing the digit (P=1000). FGA might extract a single rule with 250 true positives (TP) and 750 false negatives (FN), yielding a recall of 25% (250/1000). EFGA's TOP(3) criterion would aggregate three rules: the top rule with 250 TP, a second with 300 TP, and a third with 200 TP. The ensemble's recall becomes (250+300+200)/1000 = 75%, a 50 percentage point increase. This works because the decision trees produce disjoint rule preconditions (an image satisfies at most one rule's precondition), allowing simple addition of true positives without increasing false positives. As shown in Figure 4(a), this mechanism translates directly to higher test recall while maintaining precision through the ensemble structure (99.5% precision for the ensemble compared to 99.6% for FGA on MNIST).

## References

- Federico Formica, Stefano Gregis, Andrea Rota, Aurora Francesca Zanenga, Mark Lawford, Claudio Menghi, "Ensembles-based Feature Guided Analysis", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19653

Tags: #machine-learning #explainable-ai #feature-guided-analysis #ensemble-methods #diagnosis-support
