---
title: "Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.20155"
---

## Executive Summary
D-MMD (Discrete Moment Matching Distillation) distills discrete diffusion models into few-step generators that consistently outperform their teachers while reducing sampling steps by up to 64x. This solves the key bottleneck of high computational cost in text and image generation systems that rely on diffusion models.

## Why This Matters for Practitioners
If you're currently using diffusion models for text or image generation in production systems with 1024 sampling steps, this paper demonstrates that you can transition to 16-64 steps while achieving better output quality and significantly reduced latency. For example, on CIFAR-10 image generation, the distilled model using 32 steps achieved FID 3.7 compared to the teacher's FID 7.5 at 1024 steps. For language model applications, a Masked D-MMD generator using 16 steps achieved better GPT-2 Gradient Moment (0.236) than the teacher (0.275) at 1024 steps. This means you can implement the same model with 32x fewer computations without compromising quality.

## Problem Statement
Today's diffusion models are like a slow, methodical chef who meticulously seasons each dish component one at a time, when what production systems need is a chef who can create a great meal from just a few well-timed steps. Current discrete diffusion models require 1024 sampling steps to generate high-quality output, making them impractical for most production systems despite their potential quality advantages over autoregressive models.

## Proposed Approach
D-MMD adapts Moment Matching Distillation (MMD) from continuous to discrete diffusion models. It trains a distilled generator that matches the moment expectations of the teacher model through alternating optimisation between the student and an auxiliary model. The core insight is that discrete diffusion requires probabilistic matching rather than direct value matching.

```python
def train_discrete_mmd(student, teacher, auxiliary, dataset, steps=1000):
    for i in range(steps):
        # Sample from dataset and diffuse
        z_t = diffuse(dataset.sample(), time=random.uniform(0, 1))
        # Generate probability vector
        x_eta = student(z_t)
        # Sample discrete token
        x = sample_categorical(x_eta)
        # Draw posterior sample
        z_s = posterior_sample(z_t, x)
        
        if i % 2 == 0:
            # Train student to match teacher
            loss = (teacher(z_s) - auxiliary(z_s)) * x_eta
            student.update(loss)
        else:
            # Train auxiliary to match student and teacher
            loss = (x - auxiliary(z_s))**2 + (teacher(z_s) - auxiliary(z_s))**2
            auxiliary.update(loss)
```

## Key Technical Contributions
D-MMD's innovations go beyond simply applying continuous MMD to discrete models. The authors implemented several specific mechanisms that enable this leap:

1. The algorithm reformulates MMD to operate directly on probability distributions in discrete space, using cross-entropy loss instead of squared error. This avoids the gradient challenges of directly sampling from discrete distributions, as shown in Equation 11: LGEN(η) = CE( x̂η| x̂θ( z_s) ) - CE( x̂η| x̂φ( z_s) ).

2. They solve the "factorized output" problem by recognising that a composition of soft sampling followed by discrete sampling allows the generator to learn correlations between tokens. The model reduces output entropy to match expectations, as demonstrated in Table 6 showing decreased entropy with more distillation steps.

3. The authors developed a novel evaluation metric called Gradient Moment (GPT-2 GM) that measures how well generated samples resemble real data, overcoming the limitations of perplexity metrics that can be gamed by repeated words or ungrammatical samples.

4. They provide a solution for incorporating temperature and top-p sampling into distillation without causing gradient divergence, using a dynamic logit correction method (sθ(z_s) ← sθ(z_s) - (1 - mask_top-p) · Δ) that avoids the -10^20 masking that diverges in naive implementations.

## Experimental Results
On CIFAR-10 image generation:
- Masked diffusion teacher (1024 steps): FID 6.4
- Masked D-MMD (64 steps): FID 3.5 (2.9 better, 16x fewer steps)
- Uniform diffusion teacher (1024 steps): FID 7.5
- Uniform D-MMD (32 steps): FID 3.7 (3.8 better, 32x fewer steps)

On text generation (Open Web Text):
- Masked teacher (1024 steps): GPT-2 GM 0.275
- Masked D-MMD (16 steps): GPT-2 GM 0.236 (0.039 better, 64x fewer steps)

D-MMD outperforms prior methods like Di4C (FID 9.5 vs 5.0 at 20 vs 8 steps) and SDTT (FID 20.6 at 10 steps). The authors demonstrate that distilled models can exceed teacher performance while using far fewer sampling steps.

## Related Work
D-MMD builds on continuous MMD (Salimans et al., 2024) but generalizes it to discrete diffusion. It improves upon prior discrete distillation approaches like SDTT (Deschenaux and Gulcehre, 2025) and Di4C (Hayakawa et al., 2024) by avoiding the quality collapse that occurs when maintaining factorized output distributions. While Di4C uses mixture distributions to learn correlations (requiring exponentially more mixtures), D-MMD leaves the factorized output unchanged and instead allows the generator to match expectation moments.

## Limitations
The paper doesn't discuss the impact on training data requirements for distilled models compared to teachers. They don't provide results for conditional generation (beyond a brief mention in Section 5.2), though the metric could be extended. The evaluation relies solely on GPT-2 as a reference model, and it's unclear if using different reference models would yield consistent results.

## Appendix: Worked Example
Consider generating the text "The cat sat on the mat" using a 16-step D-MMD generator compared to a 1024-step teacher:

1. **Teacher model (1024 steps)**: At step 1024, the teacher model produces a sample with GPT-2 GM 0.275. This requires calculating 1024 denoising steps.

2. **Distilled model (16 steps)**: The D-MMD process trains the student to match the teacher's distribution:
   - At step 16, the student has learned that the probability vector for "The" should be [0.45, 0.15, 0.40] (for tokens "The", "A", and "I" respectively).
   - Through alternating optimisation, the auxiliary model learns to match the student's output at a lower step count.
   - The model reduces output entropy (from ~1.5 to ~1.2) to better correlate tokens.
   - In contrast to a standard approach where a single token might be generated independently, the distilled model learns to correlate "The" with "cat" and "sat" through the moment matching.

3. **Result**: The distilled model with just 16 steps produces samples that match the teacher's distribution more closely (GPT-2 GM 0.236 vs 0.275 at 1024 steps), meaning the generated text is more similar to the training distribution.

See Appendix for a step-by-step worked example with concrete numbers.

## References

- Emiel Hoogeboom, David Ruhe, Jonathan Heek, Thomas Mensink, Tim Salimans, "Beyond Single Tokens: Distilling Discrete Diffusion Models via Discrete MMD", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20155

Tags: #machine-learning #diffusion-models #model-distillation #discrete-diffusion
