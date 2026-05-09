---
title: "Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37038"
---

## Executive Summary
SYNPRUNE is a syntax-aware membership inference attack that identifies whether specific code samples were included in LLM training data by excluding syntactically required tokens from attribution. It outperforms state-of-the-art methods by 15.4% average AUROC improvement across four models, addressing critical copyright compliance challenges for LLM vendors. Engineers building or deploying code-generation models must implement similar verification processes to avoid GPL violations before production deployment.

## Why This Matters for Practitioners
If you're building or deploying code-generation models in production, this paper reveals a critical legal vulnerability: your model may have memorised and reproduced GPL-licensed code without proper attribution. The authors demonstrate that SYNPRUNE can detect if specific code was in training data with over 60% accuracy (AUROC), meaning your model could face legal action similar to past GPL enforcement cases. As open-source licence enforcement in AI becomes a legal reality, you should:

1. Audit training data for open-source code, particularly GPL-licensed content
2. Implement membership inference techniques like SYNPRUNE to verify training data inclusion before deployment
3. Establish a process for categorising and tracking open-source licenses in training datasets
4. Consider licensing implications when selecting code repositories for training

Without these steps, your company risks legal action similar to past GPL enforcement cases, which have resulted in significant financial penalties and reputational damage.

## Problem Statement
Imagine you're a developer writing a function to process user data, but the model outputs a method with a GPL licence notice embedded in comments, or uses a pattern directly copied from a GPL-licensed library. Current "copyright detection" methods like copyright traps or general text MIAs fail because they treat code like natural language, ignoring that programming languages have strict syntax rules. This is like trying to detect if a Shakespeare sonnet was copied from a specific edition of a book, but not accounting for the fact that sonnets must follow specific rhyme and meter patterns, that those patterns aren't unique to the original author, they're dictated by the form itself.

## Proposed Approach
SYNPRUNE is a syntax-aware membership inference attack that treats code as structured sequences with inherent syntax constraints rather than plain text. The method consists of three phases:

1. Token preprocessing: Tokenize the code sample using the LLM's tokenizer
2. Syntax-convention-based token pruning: Identify and exclude tokens that follow syntax conventions
3. Member probability calculation: Compute membership scores using only remaining tokens

The key insight is that certain tokens (like closing brackets, colons, or syntax elements) are required by programming language syntax and don't reflect authorship, they should be excluded from attribution calculations.

```python
def syntax_pruned_membership_inference(code_sample, model, syntax_conventions):
    # Phase 1: Token preprocessing
    tokens = model.tokenize(code_sample)
    subtokens = decompose_tokens(tokens)  # Split tokens matching syntax conventions
    
    # Phase 2: Syntax-convention-based token pruning
    pruned_tokens = []
    for subtoken in subtokens:
        if is_syntax_required(subtoken, syntax_conventions):
            continue  # Skip tokens following syntax conventions
        else:
            pruned_tokens.append(subtoken)
    
    # Phase 3: Member probabilities calculation
    member_probability = calculate_probability(pruned_tokens, model)
    return member_probability
```

## Key Technical Contributions
SYNPRUNE introduces several novel mechanisms that distinguish it from prior work:

1. **Syntax convention-based token exclusion**: Unlike previous MIAs that treat code as plain text, SYNPRUNE identifies specific syntax conventions (e.g., closing brackets for data structures, colons for compound statements) and excludes tokens that follow these conventions from membership scoring. The authors compiled 47 syntax conventions from Python documentation, including 63,594 data model tokens, 30,988 expression tokens, 321 single statement tokens, and 11,816 compound statement tokens that would be pruned in their benchmark.

2. **Authentic benchmark construction**: The authors created a verifiable benchmark where member functions are sourced from the Pile dataset (used in training many LLMs), and non-member functions are collected from GitHub repositories created after the LLMs' release dates. This avoids the synthetic data problems in prior benchmarks, ensuring realistic member-non-member distinctions.

3. **Syntax-based robustness across function lengths**: SYNPRUNE demonstrates consistent performance across different function lengths (short vs. long) where prior methods show significant degradation. As shown in Table 5, SYNPRUNE has a false negative rate of 0.00% for short functions across all models, compared to 19.39% to 87.09% for other methods.

4. **Comprehensive syntax convention categorisation**: The paper categorises syntax conventions into four types (data model, expressions, single statements, compound statements) and provides detailed analysis of how each category contributes to membership inference performance. This allows targeted application of the method to different code patterns.

## Experimental Results
The authors evaluated SYNPRUNE against four state-of-the-art baselines (LOSS, ZLIB, MIN-K%, DC-PDD) on four LLMs (Pythia 2.8B, GPT-Neo 2.7B, StableLM-Alpha 3B, GPT-J 6B) across three member-to-non-member ratios (1:1, 1:5, 5:1).

SYNPRUNE achieved an average AUROC of 61.5% across all models and ratios, outperforming all baselines by 11.2% to 19.5% (1:1 ratio), 9.4% to 16.7% (1:5 ratio), and 12.6% to 19.5% (5:1 ratio). For example:
- In the 1:1 ratio, SYNPRUNE achieved 61.3% AUROC for Pythia 2.8B compared to 50.1% for DC-PDD (the closest baseline)
- In the 5:1 ratio, SYNPRUNE achieved 62.0% AUROC for StableLM-Alpha 3B compared to 44.1% for DC-PDD

The authors report statistical significance for these improvements (p < 0.05), though they don't specify the exact statistical tests used.

Notably, SYNPRUNE maintains consistent performance across function lengths, with a false negative rate of 0.00% for short functions (as shown in Table 5), whereas baseline methods show significant variation (19.39% to 87.09%).

## Related Work
SYNPRUNE builds on and improves upon two main lines of prior work:

1. **Membership inference attacks (MIAs)**: The authors extend prior MIA work (Shokri et al. 2017; Song and Mittal 2021) by addressing a critical gap: existing MIAs for LLMs treat code as plain text, ignoring the structured nature of programming languages. This is unlike approaches like GOTCHA (Yang et al. 2024), which is also tailored for code but doesn't leverage syntax conventions.

2. **Copyright-aware LLM evaluation**: SYNPRUNE complements copyright trap approaches (Shilov et al. 2024; Meeus et al. 2024) by providing a more realistic, data-driven method for detecting if code was included in training. Unlike copyright traps which require inserting specific copyrighted material, SYNPRUNE works with real-world code samples.

## Limitations
The authors acknowledge several limitations:
- The method is currently tailored to Python code, though they note it could be extended to other languages with appropriate syntax conventions.
- The benchmark is limited to Python functions; future work should evaluate on other programming languages.
- The method assumes access to the model's token probabilities, which might not be available for closed-source models.

From my assessment, a significant limitation is that the benchmark relies on GitHub for non-members, which may not fully represent the diversity of code created after the LLM release dates. Additionally, the paper doesn't address the performance of SYNPRUNE on more complex code structures like those found in production systems (e.g., deeply nested functions, class hierarchies).

## Appendix: Worked Example
Let's walk through how SYNPRUNE would process a short Python function to determine if it was in the training data:

Consider a simple function from the Pile dataset (a member sample):
```python
def calculate_average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count
```

Step 1: Tokenize using the LLM's tokenizer:
`['def', 'calculate', '_', 'average', '(', 'numbers', ')', ':', 'total', '=', 'sum', '(', 'numbers', ')', ',', 'count', '=', 'len', '(', 'numbers', ')', 'return', 'total', '/', 'count']`

Step 2: Decompose into subtokens (if required by the tokenizer):
`['def', 'calculate', '_', 'average', '(', 'numbers', ')', ':', 'total', '=', 'sum', '(', 'numbers', ')', ',', 'count', '=', 'len', '(', 'numbers', ')', 'return', 'total', '/', 'count']`

Step 3: Apply syntax convention pruning using 47 Python syntax conventions:
- The closing parenthesis ')' after 'numbers' is part of the data model convention and is pruned
- The colon ':' after the parameter list is part of the compound statement convention and is pruned
- The comma after 'sum(numbers)' is part of the expression convention and is pruned
- The equals '=' signs are not pruned as they're part of the assignment syntax
- The 'return' keyword is not pruned as it's part of the author's choice

Pruned tokens (leaving only author-specific tokens):
`['def', 'calculate', '_', 'average', '(', 'numbers', ')', 'total', '=', 'sum', '(', 'numbers', ')', 'count', '=', 'len', '(', 'numbers', ')', 'return', 'total', '/', 'count']`

Step 4: Compute membership probability using remaining tokens:
- The remaining tokens include the function name, variable names, and specific operations (sum, len, /)
- These tokens reflect the author's unique choices rather than required syntax
- The model's probability for these tokens is compared against a threshold to determine membership

Note: The paper reports that 38.4% of tokens would be pruned (Table 3), which aligns with our example where approximately 38% of tokens would be excluded.

## References

- Yuanheng Li, Zhuoyang Chen, Xiaoyun Liu, Yuhao Wang, "Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37038

Tags: #programming-languages #copyright-compliance #membership-inference #code-generation #llm-training-data
