---
title: "RTMol: Rethinking Molecule-text Alignment in a Round-trip View"
category: "AI Applications"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/36966"
---

## Executive Summary
RTMol introduces a bidirectional alignment framework for molecule-text understanding that eliminates the need for paired datasets through self-supervised round-trip learning. It achieves up to 47% improvement in exact molecule reconstruction across LLM backbones by prioritising chemical fidelity over linguistic similarity in evaluation metrics. For engineers building chemical AI systems, this means more reliable molecule generation without expensive curation of high-quality datasets.

## Why This Matters for Practitioners
If you're building a drug discovery platform that uses LLMs for molecular generation or captioning, this paper directly challenges your current evaluation metrics. Most teams still rely on BLEU/METEOR scores that reward fluent chemistry descriptions but ignore chemical validity, RTMol's approach shows that a 47% increase in exact molecule reconstruction (from 19% to 28% for ChemDFM) is achievable without paired datasets. Specifically, you should:
1. Audit your model evaluation metrics: Replace BLEU/METEOR with chemical similarity metrics (MACCS, RDKit, Morgan fingerprints) for molecule reconstruction tests
2. Implement round-trip consistency checks in your training pipeline: Generate text from a molecule, then reconstruct the molecule from that text
3. Use RTMol's unsupervised training for molecular captioning when your dataset contains noisy descriptions (e.g., L+M-F dataset improved from 0.1% to 2.7% exact match)

## Problem Statement
Current molecular captioning systems are like a telephone game where chemists describe molecules verbally, but each translation step distorts the chemistry. The first player says "molecule with three carbon atoms, hydroxyl group on middle carbon" (Propan-2-ol), but the last player hears "molecule with three carbons, hydroxyl on first carbon" (Propan-1-ol), a fundamental chemical difference. Existing approaches treat molecule-to-text and text-to-molecule as separate games, leading to inconsistent chemistry because the evaluation metrics reward fluent descriptions that may describe the wrong molecule.

## Proposed Approach
RTMol solves the problem through a self-supervised round-trip learning framework that unifies molecule-to-text and text-to-molecule tasks. The core mechanism involves training a single LLM to serve as both the Captioner (molecule-to-text) and Generator (text-to-molecule), with their training alternating to enforce consistency. 

For a given molecule $x$, the Captioner generates a description $y \sim p_θ(y|x)$, which the Generator then uses to reconstruct a molecule $x' \sim q_φ(x'|y)$. The reconstruction score $S(x, x')$ (defined in Equation 7) evaluates three aspects:
1. Validity: Whether $x'$ is chemically valid
2. Exact match: Whether $x' = x$
3. Similarity: Using combined MACCS, RDKit, and Morgan fingerprints

The Captioner learns to maximise the reconstruction score $S(x, x')$, not matching reference captions, driving it to produce descriptions that preserve chemically relevant information. This avoids reliance on noisy datasets while directly optimising for chemical fidelity.

```python
def rtmol_round_trip_training(molecule):
    # Captioner generates description (unpaired with reference)
    caption = captioner.generate(molecule)
    
    # Generator reconstructs molecule from caption
    reconstructed = generator.generate(caption)
    
    # Calculate reconstruction score (Equation 7)
    valid = check_validity(reconstructed)
    exact = (reconstructed == molecule)
    similarity = calculate_similarity(molecule, reconstructed)
    
    score = 0 if not valid else (similarity + exact)
    
    # Use score as reward for Captioner training
    captioner.update_policy(reward=score)
    
    # Generator remains fixed during Captioner training
    return score
```

## Key Technical Contributions
RTMol fundamentally changes how we evaluate and train molecule-text alignment systems. Unlike prior approaches that rely on paired datasets and textual metrics, it introduces a chemically grounded framework that enforces bidirectional consistency.

1. **A round-trip metric that evaluates captions by chemical fidelity**: The metric $R(θ, ϕ)$ measures whether reconstructed molecules exactly match the original (Equation 2), rather than relying on n-gram overlap like BLEU. When a caption results in a molecule that matches the original (e.g., exact match = 1), the metric scores 1; otherwise, it scores 0. This directly aligns with chemical accuracy requirements.

2. **RTMol's reinforcement learning framework**: The Captioner learns through round-trip consistency with a reward signal based on reconstruction quality ($S(x, x')$), not reference captions. This enables unsupervised training for molecular captioning, models can learn from unlabeled molecular structures without requiring paired molecule-text data.

3. **Chemical similarity metrics for generation evaluation**: The framework uses three fingerprint-based similarity metrics (MACCS, RDKit, Morgan) combined into a single score (Equation 8). These metrics capture complementary chemical features (substructures, physicochemical properties, topological scaffolds) rather than focusing on linguistic similarity.

4. **Bidirectional training with coupled models**: Instead of training separate models for captioning and generation, RTMol uses a single LLM that serves both roles. The Generator acts as an evaluator for the Captioner's output, creating a closed-loop training process that enforces consistency between the two directions.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper evaluates RTMol on the ChEBI-20 dataset (33,010 molecule-text pairs) with a standard 8:1:1 split, using 100 samples for benchmarking. Key results for ChemDFM (the best-performing baseline):

- **Round-trip captioning**: Exact match improved from 19.0% to 28.0% (+47.4%), Morgan similarity from 0.457 to 0.597 (+31.1%), validity from 90.0% to 98.0% (+8.0%)
- **Text-based molecular design**: Exact match improved from 24.0% to 54.0% (+125.0%), Morgan similarity from 0.620 to 0.800 (+29.0%)
- **Chemical metrics**: For all models, RTMol consistently improved exact match, validity, and fingerprint similarities across MACCS, RDKit, and Morgan
- **Textual metrics**: BLEU and METEOR also improved (ChemDFM: BLEU from 0.603 to 0.722, METEOR from 0.734 to 0.812), showing enhanced bidirectional alignment

The authors compared against three categories of baselines:
1. General-purpose LLMs: GPT-4o, DeepSeek-V3, Gemini-2.5-Flash, ether0
2. Domain-specific models: ChemT5-0.2B, ChemDFM-8B
3. RTMol variants using these as base models

The paper does not report statistical significance testing for the improvements, so we cannot confirm if the 47% improvement is statistically significant (p < 0.05). The authors state "Experiments demonstrate that RTMol enhances bidirectional alignment performance by up to 47%," but do not provide confidence intervals or p-values.

## Related Work
RTMol builds on recent work using LLMs for molecular understanding (Li et al. 2024; Zhang et al. 2024; Sadeghi et al. 2024) but addresses three limitations they leave unaddressed: 
1. The disconnect between linguistic metrics (BLEU/METEOR) and chemical accuracy
2. The dependence on high-quality, paired datasets
3. The lack of bidirectional consistency in current frameworks

Unlike previous approaches that rely on supervised fine-tuning or contrastive learning pipelines, RTMol introduces a self-supervised round-trip learning framework that eliminates these dependencies. It also improves upon Guo et al. (2023), which identified the bidirectional inconsistency problem but did not provide a solution.

## Limitations
The paper acknowledges that RTMol's performance depends on the quality and coverage of reference descriptions. For example, on the noisy L+M-F dataset, reference captions yielded poor reconstruction (exact match <0.1% on chemical metrics), while RTMol's unsupervised approach improved results significantly.

The authors do not test RTMol on extremely large-scale datasets (beyond 240K examples in Mol-Instruct-F), so its scalability to million-scale datasets remains unverified. Additionally, the paper does not evaluate the framework's performance on molecules with complex stereochemistry or rare functional groups, which might challenge the fingerprint-based similarity metrics.

From an engineering perspective, the framework requires a substantial computational investment for the round-trip training process. The paper does not provide latency or throughput metrics for the training process, making it difficult to assess the practical cost for production systems.

## Appendix: Worked Example
Let's walk through a single molecule from the ChEBI-20 dataset using RTMol with the ChemDFM backbone:

1. **Input molecule**: Propan-2-ol (SMILES: CC(C)O)
2. **Captioner generates description**: "The molecule is a secondary alcohol with the hydroxyl group attached to the second carbon atom of the propyl chain" (based on RTMol's unsupervised training)
3. **Generator reconstructs from this description**:
   - Checks validity: CC(C)O is chemically valid (score = 1)
   - Compares to original: Exact match = 1 (score = 1)
   - Calculates similarity:
     - MACCS: 0.820
     - RDKit: 0.969
     - Morgan: 0.892
   - Similarity score = 0.820 + 0.969 + 0.892 = 2.681
   - Total reconstruction score = 1 (validity) + 1 (exact match) + 2.681 (similarity) = 4.681

4. **Captioner updates its policy**: The reward signal of 4.681 (scaled appropriately) is used to update the Captioner's parameters to produce more accurate descriptions in the future.

5. **Validation**: The reconstructed molecule (CC(C)O) matches the original molecule exactly, confirming the system's chemical fidelity (exact match = 100% as shown in Figure 3).

This process ensures that the model learns to generate descriptions that preserve chemically relevant information, specifically, the position of the hydroxyl group on the carbon chain, which is critical for distinguishing between structural isomers like Propan-2-ol and Propan-1-ol.


## References

- **Code:** https://github.com/clt20011110/RTMol
- Letian Chen, Runhan Shi, Gufeng Yu, Yang Yang, "RTMol: Rethinking Molecule-text Alignment in a Round-trip View", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/36966
