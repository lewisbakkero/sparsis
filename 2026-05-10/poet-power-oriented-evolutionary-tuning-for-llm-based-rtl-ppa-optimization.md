---
title: "POET: Power-Oriented Evolutionary Tuning for LLM-Based RTL PPA Optimization"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19333"
---

## Executive Summary
POET solves two critical problems in applying LLMs to RTL code optimisation: ensuring functional correctness through deterministic verification and systematically prioritizing power reduction in multi-objective PPA trade-offs. It achieves 100% functional correctness across 40 diverse RTL designs while delivering the best power results on every benchmark, with competitive area and delay improvements.

## Why This Matters for Practitioners
If you're optimising hardware designs in production systems where power consumption directly impacts battery life or thermal management (e.g., IoT sensors or mobile SoCs), POET eliminates the functional verification bottleneck that plagues existing LLM-based approaches. Rather than manually crafting testbenches or risking functional errors when prioritising power, you can now deploy an automated pipeline that guarantees correctness while systematically achieving power reduction. For your next RTL optimisation project, skip the manual weight tuning for power vs. area trade-offs and adopt POET's verified evolutionary approach to immediately realise power savings without functional regressions.

## Problem Statement
Current LLM-based RTL optimisation resembles a chef trying to perfect a recipe while blindfolded: they generate new versions (optimised designs) but can't verify if the dish (hardware functionality) still works because the LLM hallucinates the taste (output signals). This is especially dangerous when prioritising power reduction, as a single functional error renders all PPA gains meaningless, like reducing caloric content in a poisoned dish.

## Proposed Approach
POET integrates two core components: a differential-testing verification pipeline and a power-oriented evolutionary optimiser. The verification pipeline treats the original design as a functional oracle to generate deterministic testbenches, while the optimiser uses non-dominated sorting with power-first ranking to steer searches toward low-power Pareto solutions. The evolutionary loop generates, evaluates, and selects RTL variants through LLM prompts with domain-specific operators (mutation/crossover), guided by power-aware selection.

```python
def poet_optimisation(original_design, testbench, population_size=10, generations=10):
    population = initialise_population(original_design, population_size)
    for generation in range(generations):
        offspring = []
        for _ in range(population_size):
            operator = select_operator(ucb_rewards)  # UCB for exploration/exploitation
            parent = select_parent(population, rank_based_probability)
            child = llm_mutate_or_crossover(parent, operator)
            if not verify(child, testbench):
                child = llm_repair(child)  # Up to 3 repair attempts
            if verify(child, testbench):
                offspring.append((child, evaluate_ppa(child)))
        # Power-oriented selection
        population = power_oriented_select(population + offspring, population_size)
    return population.pareto_front
```

## Key Technical Contributions
POET introduces three novel mechanisms that address both challenges simultaneously.

1. **Differential-testing verification pipeline**: Instead of relying on LLM-generated testbenches (which hallucinate outputs) or manual testbench creation, POET decomposes verification into input-side tasks (LLM handles specification extraction and test scenario generation) and output-side tasks (deterministic simulation of the original design). The LLM extracts a structured specification from the original RTL, generates test stimuli covering boundary conditions, and simulates these through the original design to produce golden output signals. This eliminates hallucination in verification while automating testbench assembly.

2. **Power-first intra-level ranking in non-dominated sorting**: While standard multi-objective optimisation (e.g., NSGA-II) sorts individuals by Pareto dominance, POET introduces power-oriented ranking *within* each Pareto level. For designs on the same Pareto front (neither dominates the other), it sorts them by ascending power (lower power = higher rank). This ensures that among equally Pareto-optimal designs, the low-power option is always selected, without requiring manual weight tuning for power.

3. **Proportional survivor selection preserving Pareto diversity**: Unlike sequential fill in NSGA-II that may drop lower Pareto levels, POET allocates selection slots proportionally to Pareto level priority using $s_k = \max(1, \lfloor N \cdot w_k \rfloor)$ where $w_k = \frac{L-k+1}{\sum_{j=1}^{L}(L-j+1)}$. This guarantees that even the lowest Pareto level retains at least one representative, preserving potentially valuable optimisation strategies while steering the search toward low-power solutions (see Appendix for worked example).

## Experimental Results
POET was evaluated on 40 RTL designs from the RTL-OPT benchmark, with all methods using GPT-4o-mini and identical LLM call counts. Key results:

- **Functional correctness**: 100% (40/40 designs), outperforming I/O prompting (72.5%), CoT (82.5%), and REvolution (90.0%).
- **Power**: Best on all 40 designs (e.g., `alu_64bit` reduced power from 1040 to 944 μW, a 9.2% improvement).
- **Area**: Best on 29/40 designs (72.5%), e.g., `adder_carry` achieved 47.61 μm² vs. original 52.14 μm² (9.9% reduction).
- **Delay**: Best on 27/40 designs (67.5%), e.g., `comparator_16bit` reduced CPD from 0.29 to 0.22 ns (24.1% improvement).

Ablation studies confirmed each component's necessity (see Figure 2): omitting differential testing dropped functional correctness to 72.5%, removing power ranking reduced power improvements by 38.7%, and disabling proportional allocation caused loss of 7.9% of Pareto-optimal solutions.

## Related Work
POET positions itself against three categories: (1) LLM-based verification methods (e.g., VeriOpt, LLM-VeriPPA) that rely on manual testbenches or unreliable LLM-generated testbenches; (2) Evolutionary approaches (e.g., REvolution, VFlow) that use weighted-sum fitness requiring manual tuning; and (3) Structure-aware optimisers (e.g., RTLRewriter, SymRTLO) that focus on local transformations without global multi-objective search. POET improves over all by combining deterministic verification with power-aware evolutionary search, eliminating the need for manual weight tuning while guaranteeing functional correctness.

## Limitations
The authors acknowledge limitations in scalability: the method assumes designs are verified within a fixed number of repair attempts (R=3), which may not suffice for extremely complex circuits. The benchmark (RTL-OPT) contains only 40 designs, so generalisation to larger industrial-scale RTL is untested. Additionally, the paper doesn't quantify the cost of synthesising each candidate (e.g., time per design), which could impact practical adoption.

## Appendix: Worked Example
Let's walk through the power-first ranking mechanism using `alu_64bit` (original power: 1040 μW):

1. **Population after generation 5**:  
   - Pareto Level 1 (best): `D1` (power=944 μW, area=1321 μm², CPD=2.55 ns), `D2` (power=952 μW, area=1305 μm², CPD=2.60 ns)  
   - Pareto Level 2 (next best): `D3` (power=965 μW, area=1280 μm², CPD=2.75 ns), `D4` (power=970 μW, area=1310 μm², CPD=2.65 ns)  

2. **Non-dominated sorting**:  
   Level 1 contains `D1` and `D2` (neither dominates the other).  
   Level 2 contains `D3` and `D4`.  

3. **Power-first intra-level ranking**:  
   - Level 1: `D1` (944 μW) > `D2` (952 μW) → *D1* selected before *D2*  
   - Level 2: `D3` (965 μW) > `D4` (970 μW) → *D3* selected before *D4*  

4. **Proportional slot allocation** (N=10, L=2):  
   $w_1 = \frac{2-1+1}{(2-1+1)+(2-2+1)} = \frac{2}{3}$ → $s_1 = \lfloor 10 \cdot \frac{2}{3} \rfloor = 6$  
   $w_2 = \frac{1}{3}$ → $s_2 = \lfloor 10 \cdot \frac{1}{3} \rfloor = 3$  
   *Level 1 gets 6 slots, Level 2 gets 3*.  

5. **Selection**:  
   Top 6: `D1`, `D2`, `D3` (Level 1: 2 slots used, Level 2: 1 slot used) → *D3* is selected as Level 2's highest-ranked candidate.  
   Remaining 4 slots: Next 4 from Level 1 (but Level 1 only had 2).  
   Final population: `D1`, `D2`, `D3`, `D4` (with `D1` and `D2` prioritised), steering toward low-power solutions.

See Section 2.3.3 for the full ranking mechanism.

## References

- Heng Ping, Peiyu Zhang, Zhenkun Wang, Shixuan Li, Anzhe Cheng, Wei Yang, Paul Bogdan, Shahin Nazarian, "POET: Power-Oriented Evolutionary Tuning for LLM-Based RTL PPA Optimization", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19333

Tags: #hardware-optimisation #llm-applications #multi-objective-optimisation #evolutionary-algorithms #functional-verification
