---
title: "Full-Stack Domain Enhancement for Combustion LLMs: Construction and Optimization"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19268"
---

## Executive Summary

The paper proposes a full-stack domain-enhanced LLM workflow specifically designed for combustion science, addressing the critical problem of hallucinations and physical inconsistency in general-purpose LLMs. This workflow integrates automated domain corpus construction, incremental pre-training, instruction fine-tuning, and verifiable reward-based reinforcement learning to ensure models internalise physical laws rather than merely learning statistical patterns. For senior engineers building production systems in engineering domains with strict physical constraints, this approach provides a concrete methodology to reduce hallucinations and improve reliability in domain-specific applications.

## Why This Matters for Practitioners

If you're building production systems that require physical consistency (like combustion simulation, materials design, or fluid dynamics), this paper shows that simply using RAG or fine-tuning on domain data isn't enough. The authors demonstrate their model achieves 43.8% accuracy on FlameBench, outperforming even the best RAG method (GLM-4 + RAG at 32.09%) by 11.71 percentage points.

This means for engineering domains with strict physical constraints, you shouldn't just think about adding domain knowledge (RAG) but also need to explicitly optimise for physical consistency during training. Specifically, the RLVR stage (reinforcement learning with verifiable rewards) is what enables the significant jump from 35.1% to 43.8% accuracy. If you're currently using RAG for domain-specific applications, this suggests you should explore incorporating a verifiable reward mechanism during training rather than relying solely on retrieval at inference time.

## Problem Statement

General-purpose LLMs struggle with combustion science not because they're "stupid," but because they're trained on text that doesn't encode the physical constraints governing combustion processes. Imagine trying to build a bridge using only a recipe book for making cakes - the model might describe all the ingredients and steps correctly, but it wouldn't know that bridges need to withstand gravity or that materials have specific tensile strengths. Similarly, LLMs trained on general text can describe combustion equations but can't ensure their reasoning respects conservation laws. This leads to "hallucinations" like predicting reaction pathways that violate chemical kinetics or efficiency estimates that contradict energy conservation.

## Proposed Approach

The authors propose a full-stack workflow that integrates four components:
1. Construction of a large-scale, combustion-specific corpus
2. Multi-stage model adaptation (incremental pre-training, supervised fine-tuning)
3. Reinforcement learning with verifiable rewards (RLVR)
4. Standardised evaluation using FlameBench

The core innovation is not just collecting domain-specific data, but designing training stages that explicitly enforce physical consistency throughout the model's development.

```python
def rlvr_optimization(model, flamebench):
    """
    Implements reinforcement learning with verifiable rewards for combustion reasoning.
    
    Parameters:
    - model: The current model state
    - flamebench: The domain-specific benchmark
    
    Returns:
    - Optimised model with improved physical consistency
    """
    # Binary reward function that penalises physical inconsistencies
    def verify_reward(response):
        if not verify_conservation_laws(response) or not verify_chemical_kinetics(response):
            return -1.0  # Penalty for physical inconsistency
        return 1.0       # Reward for physically consistent response
    
    # Train model using reward function
    for step in range(7000):  # 7K complex reasoning instances
        responses = model.generate(flamebench.samples)
        rewards = [verify_reward(r) for r in responses]
        model.update_policy(rewards)
        
        # Maintain training stability with KL divergence constraint
        if step % 100 == 0:
            model.apply_kl_constraint()
    return model
```

## Key Technical Contributions

The authors' key contribution is not just a new dataset, but a full-stack workflow that addresses the dual challenges of knowledge acquisition and physical consistency through carefully designed training stages:

1. **Domain Corpus Construction with Hybrid Token Distribution**: The authors constructed a 30B-token corpus with a specific 5B:25B ratio of combustion-specific to general-domain tokens, rather than just collecting all combustion-domain text. This balance ensures the model maintains general language capabilities while acquiring domain-specific knowledge. The distribution (25.7% PDFs, 20% combustion, 16.1% math, etc.) was carefully optimised for combustion reasoning tasks.

2. **Multi-Stage Training with RLVR for Physical Consistency**: The key innovation is the RLVR stage, which explicitly penalises physical inconsistencies during training rather than just relying on retrieval at inference time. As the paper states, "A binary reward function is defined to explicitly penalise violations of domain knowledge and physical consistency." The authors show that this stage alone boosts accuracy from 35.1% (SFT-Combustion) to 43.8% (RLVR-Opt), demonstrating that physical consistency must be enforced during training, not just at inference time.

3. **FlameBench: A Domain-Specific Verification Benchmark**: Unlike generic benchmarks, FlameBench was constructed from "high-information-density fragments extracted from peer-reviewed literature, dissertations, and domain-specific code repositories," with questions "grounded in a unique source reference to ensure reproducibility and verifiability." This benchmark provides the necessary ground truth to define and verify physical consistency in combustion reasoning.

See Appendix for a step-by-step worked example of the RLVR process with concrete numbers.

## Experimental Results

The authors evaluated their model on FlameBench, a domain-specific benchmark for combustion science reasoning, using multiple-choice accuracy as the primary metric.

Key results:
- Qwen-8B (base model): 26.8% accuracy
- CPT (continued pre-training): 33.3% accuracy (+6.5 points)
- SFT-General (general SFT): 33.5% accuracy (+0.2 points)
- SFT-Combustion (combustion-specific SFT): 35.1% accuracy (+1.6 points)
- RLVR-Opt (RLVR optimisation): 43.8% accuracy (+8.7 points)

Comparing against closed-source models:
- GPT-5: 15.60% accuracy
- GLM-4: 32.64% accuracy
- Gemini Pro: 32.10% accuracy
- DeepSeek-R1: 28.37% accuracy
- Average: 27.18% accuracy

The RLVR-Opt model outperforms the best RAG-based approach (RAG + GLM-4, 32.09%) by 11.71 percentage points and outperforms the best base model (GLM-4, 32.64%) by 11.16 percentage points. The paper does not report statistical significance testing, but the results show a clear monotonic improvement across training stages.

## Related Work

The authors position their work as building on the limited prior work in domain-specific LLMs (e.g., AlphaFold for protein structure prediction, BioGPT for biomedical text), but note that "despite the central role of combustion science in energy and aerospace applications, the development of domain-specific foundation models for combustion remains largely unexplored."

They also build on domain adaptation methods (continued pre-training, SFT, RAG) but highlight their limitation in combustion science: "incremental pre-training alone does not enforce physical consistency, SFT provides limited supervision for tightly coupled dynamical reasoning, RAG struggles with integrated multi-physics reasoning."

## Limitations

The authors acknowledge that FlameBench is specific to combustion science, so the benchmark might not generalise to other engineering domains with different physical constraints. They also note that the RLVR stage required careful design of the binary reward function based on physical rules, which "would require domain experts to define these rules for other domains."

The paper doesn't report on scalability of their approach to other domains or how the domain-specific corpus construction could be automated for other engineering domains. The evaluation is limited to a single benchmark (FlameBench), with no comparison to human performance.

## Appendix: Worked Example

Let's walk through a specific combustion reasoning task from FlameBench. Consider the question: "Given a methane-air mixture at 1000K and 1 atm, with a stoichiometric ratio of 1:1.8, what is the expected flame velocity in m/s, assuming a steady-state diffusion flame?"

**Initial model response (SFT-Combustion):** "The flame velocity is 0.8 m/s based on the given conditions."

**Verification process:**
1. Conservation of mass: The model didn't reference mass conservation (fails)
2. Chemical kinetics: The model didn't reference the Arrhenius equation (fails)
3. Energy conservation: The model didn't account for heat release (fails)

**Reward calculation:** -1.0 (physical inconsistency detected)

**After RLVR optimisation:** "Based on the Arrhenius equation for methane combustion (k = A exp(-Ea/RT)), with A = 1.0×10^13 s^-1, Ea = 100 kJ/mol, and the given conditions (T = 1000K), the flame velocity is approximately 0.75 m/s. This calculation respects mass conservation (1 mole CH4 + 1.8 moles O2 → 1 mole CO2 + 2 moles H2O) and energy conservation (heat release of 800 kJ/mol)."

**Verification process:**
1. Conservation of mass: Reference to balanced reaction (passes)
2. Chemical kinetics: Reference to Arrhenius equation (passes)
3. Energy conservation: Reference to heat release (passes)

**Reward calculation:** +1.0 (physically consistent response)

This RLVR process iteratively adjusts the model to generate responses that pass the physical consistency checks, leading to the final 43.8% accuracy on FlameBench.

## References

- Quanjia Xiao, Weimin Ouyang, Zonglin Yang, Tianhao Wu, Qingguo Zhou, Runze Mao, Zhi X. Chen, "Full-Stack Domain Enhancement for Combustion LLMs: Construction and Optimization", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19268

Tags: #ai-applications
