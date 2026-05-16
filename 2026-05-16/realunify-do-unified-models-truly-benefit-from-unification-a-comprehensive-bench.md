---
title: "RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2509.24897"
---

## Executive Summary
RealUnify introduces the first comprehensive benchmark to evaluate whether visual understanding and generation capabilities in unified AI models genuinely benefit from architectural unification through bidirectional synergy. The benchmark reveals current unified models struggle to effectively leverage this synergy, achieving only 37.5% average accuracy on the most challenging tasks where understanding should enhance generation.

## Why This Matters for Practitioners
If you're architecting production systems that require both visual understanding and generation capabilities (like visual question answering with image synthesis), this paper suggests that simply adopting a unified model is insufficient for high-accuracy systems. For complex real-world applications where models must reason before generating images or reconstruct visuals to answer questions, you should consider either: (1) using specialized models with carefully engineered integration (as the oracle "Gemini 2.5 Pro + GPT-Image-1" approach achieved 72.7% accuracy), or (2) developing new training strategies to foster true capability synergy rather than merely functional coexistence. This is particularly critical for applications where accuracy is paramount, such as medical image interpretation with visual explanation generation.

## Problem Statement
Current evaluation frameworks for unified models are like testing a chef who can cook and plate food separately but never actually serving a dish that combines both skills. They assess understanding and generation in isolation (like evaluating cooking ability alone and plating ability alone) or merely combine them (like putting food on plates without considering how the plate affects the dish), but fail to test whether the model can genuinely leverage understanding to guide generation (like using knowledge of ingredients to create a better dish) or use generation to enhance understanding (like mentally visualising how ingredients would combine to solve a recipe problem).

## Proposed Approach
RealUnify evaluates bidirectional capability synergy between visual understanding and generation through a dual-evaluation framework. It comprises 1,000 human-annotated instances across 10 categories and 32 subtasks, structured around two core axes:
1. **Understanding Enhances Generation (UEG)**: Understanding (reasoning) guides image generation
2. **Generation Enhances Understanding (GEU)**: Generation (mental reconstruction) supports visual reasoning

The benchmark uses a dual-evaluation protocol:
- **Direct evaluation**: Tests end-to-end capability synergy
- **Stepwise evaluation**: Decomposes tasks into understanding-then-generation (UEG) or generation-then-understanding (GEU) phases

```python
def evaluate_model(model):
    # Direct evaluation: end-to-end assessment of understanding + generation
    direct_results = []
    for task in realunify_tasks:
        if task.type == "UEG":
            image = model.generate(image_prompt=task.prompt)
            direct_results.append(verify_image(image, task.expected))
        elif task.type == "GEU":
            answer = model.answer(image=image, question=task.question)
            direct_results.append(verify_answer(answer, task.expected))
    
    # Stepwise evaluation: decomposed understanding + generation
    stepwise_results = []
    for task in realunify_tasks:
        if task.type == "UEG":
            understanding = model.understand(task.prompt)
            image = model.generate(image_prompt=understanding)
            stepwise_results.append(verify_image(image, task.expected))
        elif task.type == "GEU":
            image = model.generate(image_prompt=task.input_image)
            answer = model.answer(image=image, question=task.question)
            stepwise_results.append(verify_answer(answer, task.expected))
    
    return {
        "direct": average(direct_results),
        "stepwise": average(stepwise_results)
    }
```

## Key Technical Contributions
The benchmark's unique value lies in how it specifically tests for synergy rather than merely combining capabilities. 

1. **Task design requiring intricate interplay**: Unlike previous benchmarks that focused on aesthetics or textual relevance, RealUnify's tasks explicitly require understanding to guide generation (e.g., "Three birds, one blue and one gray, are lined up on a telephone pole. The blue bird is not in the middle, and the adjacent birds are different colours" before generating an image), rather than simply combining separate understanding and generation tasks.

2. **Dual-evaluation protocol for precise diagnosis**: The protocol enables distinguishing whether performance issues stem from weak core abilities or failure to integrate capabilities. For instance, when decomposing UEG tasks into "understanding-then-generation," BAGEL improved from 32.7% to 47.7%, revealing that models possess the required knowledge but cannot seamlessly integrate it.

3. **Human-annotated tasks spanning 32 subtasks**: The benchmark goes beyond previous efforts by having domain experts manually curate all tasks, with cross-checking by three additional reviewers to ensure data correctness, rather than relying on automated methods or (M)LLM annotations.

4. **Focus on real-world task complexity**: Tasks require mathematical computation before generation (e.g., "Code: num = int(input()); if num > 0: print("A pair of shoes")") and visual tracking of multi-step transformations (e.g., "Turn all black segments into orange, then turn all yellow into orange, then turn all green into red"), reflecting actual production challenges where models must solve complex problems.

## Experimental Results
The paper evaluated 12 unified models (11 open-source, 1 proprietary) and 6 specialized baselines using RealUnify's dual-evaluation framework.

- **Direct evaluation on UEG tasks**: Best open-source model (BAGEL) achieved 37.5% accuracy, while proprietary model (Nano Banana) reached 63.0%.
- **Direct evaluation on GEU tasks**: All models performed poorly (specific accuracy percentage not provided in abstract).
- **Stepwise evaluation on UEG tasks**: Decomposing into "understanding-then-generation" improved performance across models (BAGEL from 32.7% to 47.7%).
- **Stepwise evaluation on GEU tasks**: Decomposing into "generation-then-understanding" caused performance degradation, suggesting models default to understanding shortcuts rather than effectively leveraging generation.
- **Oracle model**: Combining specialized models (Gemini 2.5 Pro for understanding, GPT-Image-1 for generation) achieved 72.7% accuracy on UEG tasks, establishing a high-performance upper bound that current unified models fall far short of.

The results are robust and consistent across all evaluated models, with statistical significance not explicitly measured but inferred from the consistent performance patterns across multiple models.

## Related Work
Previous benchmarks like MME-Unify and UniEval assessed understanding and generation capabilities but evaluated them in isolation or merely combined tasks without testing for true synergy. T2I-CoReBench and WISE explored whether understanding enhances generation quality but did not explicitly test whether success depends on interaction between capabilities. RealUnify advances these efforts by focusing exclusively on bidirectional capability synergy, with tasks specifically designed to require intricate interplay between understanding and generation.

## Limitations
The benchmark evaluates only visual understanding and generation, so results may not generalise to other modalities like audio or text. The study limited itself to 12 unified models (11 open-source, 1 proprietary), which may not represent the full spectrum of available models. The paper doesn't explicitly address how the benchmark would scale to larger model sizes or different architectural paradigms. The stepwise evaluation protocol is resource-intensive compared to direct evaluation, which may limit its adoption in production settings.

## Appendix: Worked Example
Let's walk through a specific UEG task from RealUnify: "Three birds, one blue and one gray, are lined up on a telephone pole. The blue bird is not in the middle, and the adjacent birds are different colours." The model must first understand the constraints (blue bird not in middle, adjacent birds different colours) before generating an image.

1. **Start with understanding phase**: The model analyzes the prompt and determines the correct arrangement: Gray, Blue, Gray (since blue can't be in the middle, and adjacent birds must be different colours).
2. **Stepwise generation phase**: The model uses this understanding to generate the image directly from the inferred arrangement.
3. **Direct generation phase**: The model attempts to generate the image directly from the original prompt without explicit intermediate understanding.

In stepwise evaluation, BAGEL achieved 47.7% accuracy on this task by first understanding the arrangement (Gray-Blue-Gray) and then generating the image. In direct evaluation, it achieved only 32.7% because the model struggled to simultaneously understand the constraints and generate the correct image without this intermediate step. This demonstrates that the model possesses the required knowledge but fails to integrate it seamlessly for end-to-end tasks.

## References

- **Code:** https://github.com/FrankYang-17/RealUnify
- Yang Shi, Yuhao Dong, Yue Ding, Yuran Wang, Xuanyu Zhu, Sheng Zhou, Wenting Liu, Haochen Tian, Rundong Wang, Huanqian Wang, Zuyan Liu, Bohan Zeng, Ruizhe Chen, Qixun Wang, Zhuoran Zhang, Xinlong Chen, Chengzhuo Tong, Bozhou Li, Qiang Liu, Haotian Wang, Wenjing Yang, Yuanxing Zhang, Pengfei Wan, Yi-Fan Zhang, Ziwei Liu, "RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2509.24897

Tags: #computer-vision #multimodal-ai #capability-synergy #benchmarking #unified-models
