---
title: "CDEoH: Category-Driven Automatic Algorithm Design With Large Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19284"
---

## Executive Summary
CDEoH introduces a category-driven evolutionary framework for automatic heuristic design that explicitly models algorithm categories while jointly balancing performance and diversity in population management. It enables parallel exploration across multiple algorithmic paradigms, significantly reducing premature convergence in LLM-based heuristic search while maintaining low computational overhead.

## Why This Matters for Practitioners
If you're building production systems that require adaptive optimisation for dynamic problem instances, such as resource allocation in cloud infrastructure or real-time routing in logistics, this paper suggests you should consider explicitly managing algorithmic diversity rather than relying on single-island evolutionary approaches. Engineers should integrate category-aware population management into their LLM-based heuristic generators, treating algorithm categories as first-class citizens in the evolutionary process. This approach prevents the common pitfall of converging to a single algorithmic paradigm during execution, which can lead to catastrophic failure when problem instances shift from the training distribution. The reflection mechanism alone can reduce algorithm discard rates by up to 35% in error-prone scenarios, directly improving development velocity.

## Problem Statement
Imagine designing a search algorithm where you're constantly hitting the same local optimum despite trying different prompts, you're like a hiker who keeps finding the same mountain pass when the trail splits into multiple valleys. Traditional LLM-based evolutionary approaches (like EoH) suffer from this 'single-island problem,' where the search space narrows to one algorithmic paradigm, limiting adaptability to new problem variations. The authors observed that heuristic algorithms naturally fall into distinct categories (greedy, brute-force, dynamic programming), each with different performance ceilings, but existing methods don't explicitly model this structure.

## Proposed Approach
CDEoH integrates LLMs into an evolutionary framework with three key components: algorithm category induction, category-based two-stage selection, and a reflection mechanism for error repair. It maintains a population of algorithms where each is categorised, ensuring diverse exploration across algorithmic paradigms while preserving high-performing individuals. The framework operates as a continuous loop of generation, evaluation, category induction, and population management.

```python
def cdeoh_evolution(population, task, max_generations):
    categories = set()
    for generation in range(max_generations):
        new_population = []
        for individual in population:
            # Generate new algorithms via innovation and refinement
            candidate1 = llm_generate(individual, "innovation")
            candidate2 = llm_generate(individual, "refinement")
            
            # Evaluate and repair via reflection if needed
            for candidate in [candidate1, candidate2]:
                if not execute(candidate):
                    repair = reflection_prompt(candidate, error)
                    candidate = llm_generate(repair)
                categories.add(induce_category(candidate))
                new_population.append(candidate)
        
        # Two-stage selection: keep top from each category, then select by performance-diversity
        selected = category_elitism(new_population, categories)
        selected += performance_diversity_selection(new_population, selected, categories)
        population = selected
    return population
```

## Key Technical Contributions
CDEoH advances LLM-based heuristic design through novel mechanisms that explicitly manage algorithmic diversity:

1. **Algorithm category induction via LLM semantic understanding**: Unlike HSEvo's encoder-based approach, CDEoH directly classifies algorithms using LLM semantic understanding of both thought and code. For example, a greedy bin-packing heuristic might be categorised as "Harmonic Fit" based on its priority function: `np.exp(-r[valid] / bins[valid]) * bins[valid] / (bins[valid] + item)`, which explicitly models proportional waste minimisation. This avoids the vector space mapping and clustering complexity of prior methods.

2. **Category-driven two-stage selection**: The first stage preserves the best individual from each discovered category via category-wise elitism (ensuring structural diversity), while the second stage selects remaining candidates using a joint performance-diversity score: `S_i = (f_i - f_min)/(f_max - f_min) + λ·1/|C(i)|`. This prevents both premature convergence (by preserving category diversity) and low-performance drift (by normalising performance scores). When λ = 0, this degenerates to pure performance selection (like EoH).

3. **Lightweight reflection mechanism**: CDEoH employs a reflection prompt that provides the LLM with the algorithm's core idea, code, and error message to guide targeted modification. This replaces the common practice of discarding potentially valuable algorithms due to minor implementation flaws, reducing discard rates by up to 35% in error-prone scenarios. The mechanism uses a fixed reflection budget and repeats only when necessary, maintaining low computational overhead.

## Experimental Results
CDEoH was evaluated on two combinatorial optimisation problems at multiple scales:

**Online Bin Packing (OBP)**:
- Outperformed EoH by 5.8% on 10kC500 instances (0.448 vs 0.483 relative gap)
- Achieved 2.378 relative gap (vs EoH's 2.476) on 1kC100 instances
- Consistently improved over baselines across all problem scales (1k-10k items)

**Traveling Salesman Problem (TSP)**:
- Reduced relative distance to LKH baseline by 8.5% on size50 (9.226 vs 10.060)
- Improved by 5.9% on size100 (12.328 vs 13.097)
- Achieved statistically significant improvements (p < 0.05) over all baselines

The authors report that CDEoH consistently achieved superior average performance across tasks and scales, with ablation studies confirming that category induction is critical for maintaining diversity.

## Related Work
CDEoH builds on LLM-based evolutionary frameworks like EoH (Liu et al., 2024) and FunSearch (Romera-Paredes et al., 2024), but addresses their core limitations: EoH's single-island evolution and FunSearch's multi-island approach with high computational overhead. Unlike HSEvo's encoder-based category management, CDEoH directly uses LLM semantic understanding for category induction. The paper explicitly positions itself against traditional hyper-heuristics (which rely on human-designed heuristics) and LLM-based heuristic selection (which identifies suitable algorithms via similarity), arguing that CDEoH's category-driven approach enables more effective exploration of the heuristic space.

## Limitations
The authors acknowledge that CDEoH's effectiveness depends on the LLM's ability to correctly induce algorithm categories, a limitation they observed in cases where algorithms had novel or atypical structures. The paper doesn't specify how well the approach generalises to problems outside combinatorial optimisation (e.g., continuous optimisation tasks). Additionally, the reflection mechanism's success rate when faced with complex, multi-stage errors wasn't fully quantified. From a practical standpoint, the framework requires careful calibration of the diversity parameter λ to balance exploration and exploitation in specific application domains.

## Appendix: Worked Example
Let's walk through CDEoH's two-stage selection process for a simplified online bin packing problem instance with 100 items:

1. **Initial population**: The current population contains 10 algorithms across 4 categories (Harmonic Fit, Brute-Force, Greedy, and Dynamic Programming), with performance scores ranging from 0.95 to 0.98 (higher is better).

2. **Category-wise elitism**:
   - For each category, select the highest-scoring algorithm:
     - Harmonic Fit: 0.98 (best in category)
     - Brute-Force: 0.96
     - Greedy: 0.97
     - Dynamic Programming: 0.95
   - These four algorithms form the category-elite set E.

3. **Performance-Diversity Selection**:
   - Calculate normalized performance scores (f_i - f_min)/(f_max - f_min):
     - Harmonic Fit: (0.98 - 0.95)/(0.98 - 0.95) = 1.0
     - Brute-Force: (0.96 - 0.95)/0.03 = 0.33
     - Greedy: (0.97 - 0.95)/0.03 = 0.67
     - Dynamic Programming: (0.95 - 0.95)/0.03 = 0.0
   - Calculate diversity score for remaining candidates (using λ = 0.5):
     - Brute-Force: 0.33 + 0.5 × 1/2 = 0.58 (2 algorithms in this category)
     - Greedy: 0.67 + 0.5 × 1/3 = 0.83
     - Dynamic Programming: 0.0 + 0.5 × 1/2 = 0.25
   - Rank remaining candidates by joint score (Brute-Force 0.58, Greedy 0.83, Dynamic Programming 0.25)
   - Select top two for the population: Greedy (0.83) and Brute-Force (0.58)

4. **Next-generation population**: The top algorithm from each category (Harmonic Fit, Brute-Force, Greedy, Dynamic Programming) plus the top two from diversity selection (Greedy, Brute-Force) form the new population of 6 algorithms (with the Greedy algorithm appearing twice due to high joint score).

## References

- Yu-Nian Wang, Shen-Huan Lyu, Ning Chen, Jia-Le Xu, Baoliu Ye, Qingfu Zhang, "CDEoH: Category-Driven Automatic Algorithm Design With Large Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19284

Tags: #optimisation #automatic-algorithm-design #llm-evolution #algorithm-diversity #combinatorial-optimisation
