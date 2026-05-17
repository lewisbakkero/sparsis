---
title: "Automated Membership Inference Attacks: Discovering MIA Signal Computations using LLM Agents"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19375"
---

## Executive Summary
AutoMIA is a framework that uses LLM agents to automate the design and implementation of membership inference attack (MIA) signal computations. By systematically exploring a vast space of potential attack strategies, AutoMIA discovers novel MIAs that outperform existing methods by up to 0.18 in absolute AUC. For practitioners, this means better tools for assessing privacy risks in machine learning systems, particularly for models handling sensitive data.

## Why This Matters for Practitioners
If you're deploying machine learning models in production that handle sensitive user data, this paper suggests you should proactively assess membership inference risks. AutoMIA demonstrates that privacy vulnerabilities in ML systems are more pervasive than previously understood, with manual MIA design often missing dataset-specific vulnerabilities. Specifically, engineers should: 1) Integrate automated privacy evaluation tools like AutoMIA into their ML development pipelines; 2) Test different attack strategies per dataset rather than relying on one-size-fits-all approaches; 3) Consider the context-specific memorization patterns in your data when assessing privacy risks, as the paper shows that attack effectiveness varies significantly across datasets.

## Problem Statement
Designing effective membership inference attacks (MIAs) is like trying to crack a combination lock with a different mechanism for each lock. Just as a locksmith would need to manually craft a key for each unique lock, security researchers have traditionally needed to manually develop tailored MIAs for each model architecture and dataset. This process is time-consuming, requires deep domain expertise, and often misses subtle vulnerabilities that emerge from model memorization patterns specific to a given context.

## Proposed Approach
AutoMIA employs an evolutionary loop where LLM agents iteratively propose new attack strategies, implement and evaluate them, and store the results in a shared knowledge base. The system includes two classes of agents: Design agents (Explorer and Exploiter) that generate novel and optimised attack strategies, and Execution agents (Programmer, Executor, and Analyzer) that translate these strategies into code, run experiments, and analyse results. The design agents represent attack strategies in natural language for more effective exploration, while the execution agents handle the technical implementation.

```python
def automia_evolution_loop(user_config):
    DB = initialize_database()  # Shared knowledge base
    explorer = ExplorerAgent(DB)
    exploiter = ExploiterAgent(DB)
    
    if user_config.seed:
        DB.add(explorer.seed_experiment(user_config.seed))
    
    for _ in range(user_config.budget):
        new_idea = explorer.explore()
        refined_idea = exploiter.exploit()
        code = programmer.implement(refined_idea)
        results = executor.run(code)
        analysis = analyzer.analyse(results)
        DB.add((refined_idea, code, results, analysis))
```

## Key Technical Contributions
AutoMIA's core innovation lies in how it represents and evolves MIA strategies. Unlike previous approaches that evolved code directly, AutoMIA reasons about attack strategies in natural language first, then translates to code. This enables more effective exploration of the design space. The key technical contributions include:

1. **Natural Language Representation of MIA Strategies**: AutoMIA stores MIA experiments with high-level descriptions (ideas, design, code, and results), rather than just code and scores. This representation enables the agents to reason about past failures and successes more effectively - as shown in the ablation study where removing this representation caused a 0.11 AUC drop.

2. **Dual-Agent Evolution Strategy**: AutoMIA employs both exploration (finding novel approaches) and exploitation (refining existing ones) through the Explorer and Exploiter agents. The Explorer uses a novelty-guided loop with a Novelty Judge to ensure new ideas, while the Exploiter optimizes based on performance metrics. This dual approach outperforms single-strategy methods, with the ablation study showing a 0.067 AUC drop when removing the Exploiter.

3. **Context-Aware Attack Generation**: AutoMIA discovers attacks that are specifically tailored to the target context, as evidenced by the different optimal strategies found for different datasets (e.g., edit-distance for ArXiv and Pubmed, rare n-grams for GitHub). This demonstrates that LLM memorization patterns vary across data domains, necessitating dataset-specific attack strategies.

## Experimental Results
AutoMIA was tested on two settings: black-box LLMs (using the MIMIR benchmark) and gray-box VLMs (using the setting from Li et al., 2024). For black-box LLMs on the ArXiv dataset with Pythia 1.4B, AutoMIA achieved an AUC of 0.687 compared to the human-designed baseline's 0.547, representing a 0.14 AUC improvement. On the OPT-7B model, AutoMIA achieved 0.703 AUC versus the baseline's 0.542, a 0.161 AUC improvement. For gray-box VLMs on image logits, AutoMIA achieved 0.752 AUC (vs. baseline's 0.594), a 0.158 AUC improvement. These results represent up to 0.18 AUC improvements over existing MIAs. AutoMIA consistently outperformed both the human-designed baselines and OpenEvolve (a general-purpose algorithm search framework) across all datasets and target models.

## Related Work
AutoMIA builds on previous MIA research that has traditionally been manual and context-specific. It differs from AttackPilot (Wu et al., 2025), which uses LLM agents to reduce engineering effort in implementing existing MIAs rather than discovering novel ones. AutoMIA also differs from general-purpose frameworks like OpenEvolve, which evolves raw code directly, by reasoning about high-level attack ideas in natural language before implementing them. This approach is specifically tailored to MIA design rather than general code optimisation.

## Limitations
The paper acknowledges limitations in its experimental scope, focusing on two relatively recent MIA settings (black-box LLMs and gray-box VLMs) rather than a broader range of models. The authors note that while AutoMIA can discover effective MIAs from scratch without human-designed seeds (as shown in the ablation study), the initial performance may be slower in the early iterations. Additionally, the framework has not been tested on more complex MIA settings like those involving model fine-tuning or alignment.

## Appendix: Worked Example
Let's walk through how AutoMIA might discover a novel MIA signal for a black-box LLM on a code dataset. The system starts with a seed MIA based on n-gram overlap (the human-designed baseline). The Explorer agent retrieves relevant experiments from the database and generates a new design: "Use geometric edit-distance to compare generated code and ground-truth code." The Novelty Judge evaluates this against existing designs and determines it's novel. The Design Refiner then suggests adding a calibration factor to improve the edit-distance metric.

The Exploiter agent identifies this new design as a promising candidate and refines it further: "Calculate edit-distance between generated code and ground-truth code, then apply a logarithmic scale to amplify differences in low-performing cases." This refined design is implemented as code by the Programmer agent.

When executed, the new MIA achieves an AUC of 0.75 on the GitHub dataset, compared to the baseline's 0.62. The Analyzer agent notes that this approach works well for code snippets because code has more structured patterns than natural language, making edit-distance a more effective signal for code memorization. The system stores this design (including the natural language description of the geometric edit-distance approach) in the database for future reference.

## References

- Toan Tran, Olivera Kotevska, Li Xiong, "Automated Membership Inference Attacks: Discovering MIA Signal Computations using LLM Agents", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19375

Tags: #privacy #machine-learning #multi-agent #llm-agents
