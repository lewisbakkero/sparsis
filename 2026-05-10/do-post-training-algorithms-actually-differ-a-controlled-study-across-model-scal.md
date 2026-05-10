---
title: "Do Post-Training Algorithms Actually Differ? A Controlled Study Across Model Scales Uncovers Scale-Dependent Ranking Inversions"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19335"
---

## Executive Summary
This paper presents a controlled, large-scale comparison of 8 post-training algorithms across 4 model scales (0.5B-7B), 3 evaluation domains, and 20 variants of DPO. It reveals that algorithm rankings invert across scale (e.g., SGRPO at 1.5B vs. SimPO at 7B), that none of 20 DPO variants significantly outperform vanilla DPO after statistical correction, and that algorithm choice matters primarily on the trained task within the training distribution.

## Why This Matters for Practitioners
If you're selecting a post-training algorithm for production deployment, this paper fundamentally changes your approach. At 1.5B models, SFT outperforms all preference methods (54.4% vs. 49.1% on GSM8K), but at 7B, SimPO becomes the best option (85.8% vs. 76.4% for SFT). This scale-dependent inversion means you must validate algorithm performance at your target scale. Never assume that results from small models transfer to larger ones. Use vanilla DPO (not variants) as your default for most tasks, and prioritise model scale (50 pp impact) and training paradigm (10 pp) over loss function modifications (1 pp impact). For 7B+ models with LoRA, SimPO consistently outperforms other methods and offers the best accuracy per GPU-hour (0.55 GPU-h/pp).

## Problem Statement
The post-training algorithm landscape resembles a chaotic marketplace where vendors claim their version of 'better alignment' is superior, but each product is tested on different models, datasets, and evaluation protocols. Like choosing a car based on a brochure describing how it performs on a racetrack with a different engine, practitioners have been selecting algorithms without understanding how their choices scale across model sizes or transfer to different tasks. This paper exposes that the 'best' algorithm for a 1.5B model can be the 'worst' for a 7B model, a fundamental mismatch that leads to misallocated engineering effort.

## Proposed Approach
The authors created OXRL, a unified framework implementing 51 post-training algorithms with identical infrastructure, enabling controlled comparisons. The framework standardises model loading, data pipelines, distributed training, and evaluation, varying only the loss function. They evaluated 8 algorithms across 4 model scales, with 20 DPO variants at 1.5B (100 runs with 5 seeds each), and three evaluation domains (GSM8K, MATH, general reasoning). The core innovation is the framework's design to eliminate codebase-as-confound, ensuring only the loss function differs between methods.

```python
class BaseAlgorithm:
    def loss_function(self, x, y_w, y_l, reference_model):
        """
        Each algorithm must implement this method.
        All other components (model loading, data pipeline, training) are identical.
        """
        # Custom loss implementation goes here
        raise NotImplementedError
```

## Key Technical Contributions
The paper's key contributions go beyond the framework to reveal critical insights about the post-training landscape. Understanding these mechanisms is essential for making informed algorithm selection decisions.

1. **Scale-Dependent Algorithm Ranking Inversion**: The paper demonstrates that the optimal algorithm changes dramatically with model scale. At 1.5B, online RL (SGRPO) achieves 58.0% ± 0.57 on GSM8K, but at 7B, SimPO becomes the best method (85.8%), while SFT collapses to 76.4%. This inversion isn't due to LoRA regularisation (confirmed via 2×2 factorial), but rather to increased model capacity enabling reference-free methods to learn better format compliance.

2. **Loss Function Modifications Yield Negligible Gains**: The authors tested 20 DPO variants across 100 runs at 1.5B. None significantly outperformed vanilla DPO after Bonferroni correction (α = 0.05/19 ≈ 0.0026), with the sole significant result being SimPO (−11.5 pp, p < 10⁻⁴). This confirms that loss function engineering is a low-impact area compared to scale or training paradigm.

3. **Task-Specific Algorithm Leverage**: The GSM8K spread (19.3 pp) collapses to 0.54 pp on MATH (36× compression) and 0.47 pp on general-domain benchmarks (41×). This means algorithm choice matters primarily on the training distribution, with no method transferring gains across tasks.

See Appendix for a step-by-step worked example of how format compliance drives the 7B ranking inversion.

## Experimental Results
The study comprised 240 training runs on H100 GPUs, with key findings:

- **Core Algorithm Comparison**: At 1.5B, SGRPO (58.0% ± 0.57) outperformed SFT (54.4% ± 0.59) by 3.6 pp and DPO (49.1% ± 0.61) by 8.9 pp. At 7B, SimPO (85.8%) surpassed DPO (83.8%) by 2.0 pp and SFT (76.4%) by 9.4 pp.
- **DPO Variant Taxonomy**: Of 20 DPO variants, none significantly outperformed vanilla DPO (mean 49.76% ± 2.27) after statistical correction. ORPO achieved the highest accuracy (53.89% ± 0.60, +4.12 pp, p = 0.013, Cohen's d = 2.48) but still failed Bonferroni correction.
- **Cross-Task Generalisation**: The GSM8K spread (19.3 pp) collapsed to 0.54 pp on MATH (36× compression) and 0.47 pp on general-domain benchmarks (41×).
- **Compute Efficiency**: At 7B, SimPO offers the best accuracy-per-GPU-hour (0.55 GPU-h/pp), while SGRPO at 1.5B requires 10× the GPU-h/pp of SFT.

The paper explicitly acknowledges the hidden determinism bug in PyTorch's DistributedSampler that eliminated seed-dependent variance in previous studies, affecting the credibility of other distributed training studies.

## Related Work
This work positions itself as the first large-scale, controlled comparison of post-training algorithms. It builds on the RLHF pipeline [Ouyang et al., 2022] and DPO [Rafailov et al., 2023] but addresses a critical gap: how rankings change across scale, which DPO variants matter, and the compute-performance tradeoff between online and offline methods. Unlike Xu et al. [2024] (2-3 methods) or Spangher et al. [2025] (17 algorithms at single scale), this study spans multiple scales and includes rigorous statistical testing. It also complements recent theoretical work on GRPO, DPO equivalence [Wu et al., 2025] by showing empirical results that partially contradict this equivalence (SGRPO outperforms DPO by 8.9 pp at 1.5B).

## Limitations
The study uses a single model family (Qwen 2.5) up to 7B parameters, limiting generalisability to other architectures. SGRPO is absent at 3B/7B due to RL requiring 2 GPUs per run, restricting factorial design. The authors use synthetic preference data and greedy decoding rather than human-preference data or sampling-based evaluation, which might yield different rankings. The paper doesn't test code generation or instruction-following tasks, though MBPP configurations are released for community extension. The hidden determinism bug in DistributedSampler suggests other distributed training studies may require verification.

## Appendix: Worked Example
Let's walk through the mechanism behind the 7B ranking inversion using GSM8K data. At 1.5B, SFT produces 54.4% accurate responses with 6.2% format non-compliance (e.g., answers like "42" instead of "42 is the answer"). The model generates sufficient correct solutions to create a positive feedback loop between policy quality and training data quality.

By 7B, the model has sufficient capacity to learn format compliance autonomously. SimPO's reference-free training encourages the model to self-generate correctly formatted answers. At 7B, SimPO produces strictly formatted answers 85.8% of the time (4.9 pp higher than flexible extraction), while SFT conceals 3.9 pp of accuracy behind format non-compliant outputs. For a sample of 100 questions:

- SimPO: 85 correct answers with strict formatting (e.g., "42" → "42 is the answer"), 4.9 pp higher than flexible extraction
- SFT: 76 correct answers but only 72.5% with strict formatting (e.g., "42" without the full sentence), hiding 3.9 pp of accuracy

This format compliance mechanism explains why SimPO becomes the best method at 7B: the model's improved capacity enables it to learn proper formatting without relying on reference models. The paper confirms this by showing that format compliance correlates inversely with reference model reliance (SimPO > DPO > IPO > KTO at 7B).

## References

- **Code:** https://github.com/warlockee/oxRL.
- Xiaoyi Li, "Do Post-Training Algorithms Actually Differ? A Controlled Study Across Model Scales Uncovers Scale-Dependent Ranking Inversions", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19335

Tags: #machine-learning #large-language-models #post-training-alignment #model-scale #algorithm-selection
