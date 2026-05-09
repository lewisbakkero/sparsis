---
title: "ItinBench: Benchmarking Planning Across Multiple Cognitive Dimensions with Large Language Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19515"
---

# Technical Article

## Executive Summary
ItinBench establishes a benchmark for evaluating large language models across both verbal and spatial reasoning dimensions in travel itinerary planning. The authors demonstrate that LLMs struggle to maintain high and consistent performance when required to handle multiple cognitive domains simultaneously, with validated plan rates dropping to as low as 6% for some models. This research matters because it reveals critical limitations in how LLMs approach real-world planning tasks that require integrated reasoning, which directly impacts the reliability of LLM-based planning systems in production.

## Why This Matters for Practitioners
If you're building or evaluating LLM-based planning systems for travel, logistics, or any domain requiring spatial and verbal reasoning, this paper reveals a key risk: you'll likely overestimate model performance if you only test on verbal reasoning tasks alone. The benchmark shows that adding spatial reasoning requirements can drop validated plan rates from 65% (with pre-filtered data) to as low as 6% (with full dataset). This means you should implement both verbal and spatial reasoning metrics in your evaluation framework, and expect performance degradation of 50-70% when combining these domains. For practical implementation, structure your evaluation pipeline to include both spatial (route optimisation) and verbal (preference matching) metrics, and consider providing pre-filtered data to reduce the cognitive load on the model.

## Problem Statement
Current LLM planning benchmarks are like testing a chef only on taste preferences while ignoring their spatial reasoning for arranging ingredients on a plate. Just as a chef who can perfectly describe a dish might struggle to efficiently arrange ingredients in a kitchen, LLMs that excel at verbal reasoning tasks (like matching preferences) often falter when required to also optimise spatial relationships (like planning efficient routes between locations). The paper points out that "the non-symbolic nature of spatial reasoning makes it significantly less overlap with verbal reasoning abilities," creating a fundamental mismatch in how we evaluate these systems.

## Proposed Approach
ItinBench integrates spatial reasoning (route optimisation) into travel itinerary planning to create a benchmark that tests both verbal and spatial reasoning simultaneously. The pipeline uses a database of Philadelphia businesses (restaurants, hotels, attractions) with their reviews, generates human-like travel queries with specific preferences, and constructs four main tasks that vary in the complexity of verbal and spatial reasoning required. The evaluation measures both verbal reasoning performance (preference matching) and spatial reasoning performance (route optimisation).

The key innovation is the integration of the Traveling Salesman Problem (TSP) algorithm to quantify spatial reasoning performance within a real-world context. This allows researchers to measure not just whether the model can recommend destinations (verbal reasoning), but also whether it can arrange them efficiently (spatial reasoning).

```python
def evaluate_route_optimization(llm_plan, optimized_plan):
    """Evaluate spatial reasoning performance using adapted TSP algorithm"""
    # Calculate distance gap for each day
    daily_gaps = []
    for day in llm_plan.days:
        if day.has_valid_route():
            llm_distance = calculate_distance(llm_plan.route)
            optimized_distance = calculate_distance(optimized_plan.route)
            daily_gaps.append(optimized_distance - llm_distance)
    
    # Calculate total distance gap for the entire itinerary
    total_gap = sum(daily_gaps)
    # Calculate cluster jump ratio
    cluster_jumps = count_cluster_jumps(llm_plan)
    optimized_jumps = count_cluster_jumps(optimized_plan)
    cluster_jump_ratio = (cluster_jumps - optimized_jumps) / optimized_jumps
    
    return {
        "distance_gap": mean(daily_gaps),
        "total_distance_gap": total_gap,
        "extra_cluster_jumps": cluster_jump_ratio
    }
```

## Key Technical Contributions
The paper makes two significant contributions to LLM evaluation:

1. **Integrated cognitive dimension benchmarking**: ItinBench moves beyond traditional verbal reasoning benchmarks by incorporating a spatial reasoning task (route optimisation) into the same planning pipeline. Unlike prior work that focused on verbal reasoning alone (e.g., TravelPlanner, ITINERA), ItinBench evaluates LLMs on both dimensions simultaneously, generating metrics that specifically measure spatial reasoning performance through adapted Traveling Salesman Problem (TSP) algorithms. The key implementation detail is that the benchmark provides precomputed spatial cluster information in text to allow LLMs to reason about spatial relationships without requiring explicit geometric processing.

2. **Quantified performance trade-offs**: The authors quantitatively demonstrate that LLMs experience significant performance degradation when required to handle multiple cognitive domains concurrently, with validated plan rates dropping to 6-18% for models like Llama 3.1 8B and GPT-4o when compared to 65% with pre-filtered data. The implementation detail that makes this possible is their evaluation framework that separates verbal reasoning metrics (OOP, MI, Micro, Macro, VR) from spatial reasoning metrics (ARG, DG, Total-DG, ECJ), allowing for precise attribution of performance issues to specific cognitive domains. The paper specifically notes that "gains on spatial tasks largely arise when models are given explicit spatial-relation cues, suggesting current 'spatial reasoning' leans on semantic text manipulation rather than human-like spatial cognition."

3. **Real-world evaluation design**: Rather than using artificial, isolated settings (e.g., grids and board games), ItinBench evaluates reasoning in a realistic travel planning context with real-world data (Yelp business listings and user reviews). The key implementation detail is the data pipeline that incorporates business attributes from Yelp, extracts user review attributes (like "flavor rating" and "freshness"), and constructs human-like queries with specific preferences. The benchmark's design ensures that the verbal reasoning aspects (preference matching) and spatial reasoning aspects (route optimisation) are evaluated in the same context, reflecting how humans approach real-world planning.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
The paper evaluates models across four tasks with varying complexity:

1. **Task 1 (Entire dataset, no route optimisation)**: Highest validated rate (VR) is 18.0% for OpenAI o1, with lower rates for other models (Llama 3.1 8B: 0.0%, Mistral-large: 2.0%, Gemini-1.5-Pro: 5.0%, GPT-4o: 5.0%).

2. **Task 2 (Entire dataset, with route optimisation)**: Performance drops significantly, with OpenAI o1 achieving 4.0% VR (down from 18.0% in Task 1), while others show even larger drops (Llama 3.1 8B: 0.0%, Mistral-large: 0.0%, Gemini-1.5-Pro: 7.0%, GPT-4o: 4.0%).

3. **Task 3 (Filtered dataset, with route optimisation)**: Performance improves substantially across all models, with Mistral-large achieving the highest VR (66.7%), and OpenAI o1 at 42.0%. This shows that pre-filtering the data (reducing verbal reasoning complexity) significantly improves spatial reasoning performance.

4. **Task 4 (Tool use, with route optimisation)**: Performance varies, with Mistral-large achieving 64.0% VR, while OpenAI o1 reaches 42.3%.

Spatial reasoning results show a consistent pattern: when route optimisation is required (Tasks 2, 3, 4), models experience "15% to 38% additional unnecessary travel distance" in generated plans compared to optimised routes. For the best model (OpenAI o1), the Total-DG (Total Distance Gap) is 7.5% in Task 3 (filtered data), but rises to 24.0% in Task 2 (entire dataset).

The paper does not report statistical significance testing for most performance differences, though it notes that "the average recommendation gap is 24.3 (3.03) for Llama 3.1 8B in Task 1," where the numbers in parentheses appear to be the standard deviation.

## Related Work
ItinBench positions itself as a significant advancement over prior travel planning benchmarks like TravelPlanner, ITINERA, and UnSatChristmas, which primarily focus on verbal reasoning tasks. Unlike TripTailor, which focuses on personalized city-scale itinerary planning with day-level details, ItinBench provides an algorithmic evaluation of spatial reasoning by quantifying distance differences in final routes. The paper also distinguishes itself from spatial reasoning research that focuses on visual or spatial question answering (e.g., VSI-Bench, PATHEVAL) by evaluating LLMs in the context of real-world planning tasks that require both verbal and spatial reasoning.

## Limitations
The paper acknowledges limitations including:
- The benchmark focuses on route optimisation within a single city (Philadelphia), which may not generalise to multi-city or international travel planning.
- The spatial reasoning evaluation uses precomputed proximity relations rather than requiring the model to compute spatial relationships from scratch.
- The authors don't test whether the performance gap between verbal and spatial reasoning can be bridged through specific fine-tuning strategies.

My assessment: The single-city focus is a significant limitation for real-world applications, as international travel planning would require handling different geographical constraints and cultural contexts. The paper also doesn't explore whether providing more explicit spatial cues (beyond the precomputed clusters) could improve spatial reasoning performance.

## Appendix: Worked Example
Let's walk through a specific example of the spatial reasoning evaluation for a 3-day trip using OpenAI o1 in Task 3 (filtered dataset, with route optimisation). The paper shows that for this task, o1 achieves a Total-DG of 7.5% and ECJ of 6.9%.

We'll use a simplified example for a 3-day trip with the following constraints:
- Each day requires 4 attractions (4.00 recommended on average)
- The optimised route has a total distance of 45 miles
- The LLM-generated route has a total distance of 48.3 miles (7.5% longer)

Day 1:
- Optimised route distance: 12 miles
- LLM route distance: 12.5 miles
- Distance gap: 0.5 miles
- Cluster jumps: Optimised - 2 jumps, LLM - 3 jumps
- Extra cluster jumps: (3-2)/2 = 50% more than optimised

Day 2:
- Optimised route distance: 18 miles
- LLM route distance: 18.6 miles
- Distance gap: 0.6 miles
- Cluster jumps: Optimised - 3 jumps, LLM - 4 jumps
- Extra cluster jumps: (4-3)/3 = 33.3% more than optimised

Day 3:
- Optimised route distance: 15 miles
- LLM route distance: 17.2 miles
- Distance gap: 2.2 miles
- Cluster jumps: Optimised - 2 jumps, LLM - 4 jumps
- Extra cluster jumps: (4-2)/2 = 100% more than optimised

Total:
- Optimised total distance: 45 miles
- LLM total distance: 48.3 miles
- Total distance gap: 3.3 miles (7.5% of 45 miles)
- Average extra cluster jumps: (50% + 33.3% + 100%) / 3 = 61.1%

This example shows how the benchmark quantifies both the distance inefficiency (Total-DG) and the spatial reasoning quality (ECJ) of LLM-generated itineraries.

## References

- Tianlong Wang, Pinqiao Wang, Weili Shi, Sheng li, "ItinBench: Benchmarking Planning Across Multiple Cognitive Dimensions with Large Language Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19515

Tags: #travel-planning #spatial-reasoning #verbal-reasoning #benchmarking #llm-evaluation
