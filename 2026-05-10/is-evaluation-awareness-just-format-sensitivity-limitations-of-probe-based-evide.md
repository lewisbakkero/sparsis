---
title: "Is Evaluation Awareness Just Format Sensitivity? Limitations of Probe-Based Evidence under Controlled Prompt Structure"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19426"
---

## Executive Summary
This paper demonstrates that probe-based evidence for "evaluation awareness" in large language models (LLMs) primarily tracks benchmark format rather than evaluation context. The authors construct a controlled 2×2 dataset to isolate format and context variables, finding that standard probes misclassify deployment prompts rewritten into benchmark format as "evaluation" 94.5% of the time. For engineering teams building LLM safety systems, this means probe-based diagnostics may be unreliable for distinguishing real-world deployment from evaluation scenarios.

## Why This Matters for Practitioners
If you're building safety monitoring systems that rely on probe-based evaluation awareness signals, you must account for prompt format as a confounding variable. The paper shows that deployment prompts converted into standard benchmark format (e.g., multiple-choice questions) are indistinguishable from actual evaluation prompts to standard probes. This means your safety system might falsely flag normal user interactions as "evaluation mode" simply because they match a common benchmark format. Engineering teams should implement format controls in their diagnostic pipelines and avoid using standard probe configurations for safety monitoring without formal validation against format-shifted prompts.

## Problem Statement
Imagine a teacher grading an essay: the paper's format (structured with clear sections) might make a student's work seem more "academic" than its actual content. Similarly, LLMs might respond to the rigid structure of benchmark questions (like multiple-choice formats) as if they're in "evaluation mode," not because of the context but because of the format itself. This confound has led to potentially misleading safety diagnostics where models appear to distinguish evaluation from deployment contexts when they're actually just recognising standard benchmark templates.

## Proposed Approach
The authors create a 2×2 experimental design that independently varies usage context (evaluation vs. deployment) and prompt format (benchmark structured vs. casual free-form), while controlling for length and genre. They construct four dataset quadrants:
- Bench-Eval: Canonical benchmark prompts (MMLU, ARC, GSM8K)
- Bench-Deploy: Deployment prompts rewritten into benchmark format
- Casual-Eval: Conversational evaluation prompts (MuTual dataset)
- Casual-Deploy: Deployment prompts in conversational style (ShareGPT)

The key insight is that standard probe training (Bench-Eval vs. Casual-Deploy) conflates format and context, while "paired training" (Bench-Eval ∪ Casual-Eval vs. Bench-Deploy ∪ Casual-Deploy) decorrelates format from context.

```python
def construct_2x2_dataset():
    # Load original datasets
    mmlu = load_dataset('mmlu')
    sharegpt = load_dataset('sharegpt')
    mutual = load_dataset('mutual')
    
    # Create Bench-Eval: standard benchmarks
    bench_eval = [format_to_mmlu(p) for p in mmlu.prompts]
    
    # Create Bench-Deploy: deployment prompts rewritten to benchmark format
    bench_deploy = []
    for prompt in sharegpt.prompts:
        # Route to appropriate benchmark category
        category = route_prompt(prompt)
        # Rewrite to benchmark format
        rewritten = rewrite_to_benchmark_format(prompt, category)
        bench_deploy.append(rewritten)
    
    # Create Casual-Eval: conversational evaluation proxy
    casual_eval = [get_first_turn(p) for p in mutual.prompts]
    
    # Create Casual-Deploy: natural deployment prompts
    casual_deploy = [get_first_turn(p) for p in sharegpt.prompts]
    
    # Apply histogram matching for length control
    length_matched = histogram_match(bench_eval, casual_deploy)
    
    return {
        'Bench-Eval': bench_eval,
        'Bench-Deploy': bench_deploy,
        'Casual-Eval': casual_eval,
        'Casual-Deploy': casual_deploy
    }
```

## Key Technical Contributions
The authors introduce a rigorous methodology for disentangling format sensitivity from context awareness in LLM probing.

1. **Controlled format-transplant technique**: They develop a rewrite pipeline using DeepSeek-V3 to convert deployment prompts into benchmark formats without introducing evaluation framing. This preserves task intent while modifying surface structure, enabling precise format manipulation without confounding context changes.

2. **Length-matching protocol**: They implement histogram matching across character-length bins to eliminate prompt length as a confounding variable, ensuring probes cannot exploit length distribution differences between datasets.

3. **Genre control implementation**: They introduce formal rewriting to maintain linguistic register (academic vs. conversational) while altering format, demonstrating that probe sensitivity to format persists regardless of stylistic changes.

4. **Paired training protocol**: Instead of standard single-contrast training (Bench-Eval vs. Casual-Deploy), they use pooled data (Bench-Eval ∪ Casual-Eval vs. Bench-Deploy ∪ Casual-Deploy) with standard train/test splits. This decorrelates format from context during training, enabling the model to learn context sensitivity independently.

## Experimental Results
The paper's core finding is that standard probes fail to generalise: Bench-Deploy prompts (deployment rewritten in benchmark format) were misclassified as "Evaluation" 94.5% of the time (Layer 24, Table 2). In contrast, paired training (decorrelating format from context) reduced this error to 7.2% while correctly identifying Casual-Eval as evaluation (100% accuracy). These results hold across all layers (8, 31), with consistent patterns shown in Table 4 (Standard) and Table 5 (Paired). Crucially, neither standard nor paired probes flagged safety benchmarks (AdvBench, HarmBench) as evaluation (0% classification as Evaluation, Table 6), indicating evaluation awareness probes should not be interpreted as general-purpose safety monitors.

## Related Work
This work situates itself within the emerging field of evaluation awareness in LLMs. It directly addresses limitations in prior probe-based studies by Needham et al. (2025) and Nguyen et al. (2025), who reported probe generalisation across datasets without controlling for format-context correlation. It also challenges Chaudhary et al. (2025), who observed scaling laws for evaluation awareness probes, by demonstrating that these signals are format-dependent rather than context-dependent. The authors position their methodology as a methodological correction to standard probing practices, not a refutation of evaluation awareness itself.

## Limitations
The study uses only linear probes and a single backbone (Llama-3.1-8B-Instruct), so it doesn't rule out non-linear representations of evaluation awareness. Some dataset instantiations rely on proxy constructions (e.g., using MuTual as a conversational evaluation proxy), which may not perfectly represent actual evaluation contexts. The authors explicitly note that their results don't prove evaluation awareness is absent, it's that standard probing methods fail to isolate it from format sensitivity. The paper also doesn't address how larger models might handle this confound differently, though the layer-wise consistency suggests this is a pervasive structural bias.

## Appendix: Worked Example
Let's walk through the Bench-Deploy misclassification using their dataset construction process. Take a ShareGPT deployment prompt: "Can you help me solve this math problem? I have 3 apples and buy 5 more, how many do I have?"

1. **Routing**: The routing prompt categorises this as GSM8K (arithmetic reasoning).
2. **Rewrite**: The GSM8K-style rewrite becomes: "Q: If you have 3 apples and buy 5 more, how many do you have? Options: A. 5 B. 6 C. 7 D. 8"
3. **Length control**: The rewritten prompt (72 characters) is histogram-matched to Bench-Eval's median length (248 characters) by selecting similar-length ShareGPT prompts.
4. **Probe classification**: The standard probe (trained on Benchmark vs. Casual) classifies this as "Evaluation" 94.5% of the time because the format (multiple-choice question with numbered options) matches the benchmark structure, not because the prompt is an evaluation scenario.

This demonstrates how a natural user query can be misclassified as "evaluation mode" simply due to its adoption of a standard benchmark format, with the probe learning to associate that format with evaluation context rather than the actual usage intent.

## References

- Viliana Devbunova, "Is Evaluation Awareness Just Format Sensitivity? Limitations of Probe-Based Evidence under Controlled Prompt Structure", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19426

Tags: #artificial-intelligence #large-language-models #safety-testing #model-probing #format-sensitivity
