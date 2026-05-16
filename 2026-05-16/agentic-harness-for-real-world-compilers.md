---
title: "Agentic Harness for Real-World Compilers"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.20075"
---

## Executive Summary
llvm-autofix introduces the first agentic harness designed to assist LLM agents in understanding and fixing compiler bugs within LLVM, one of the most widely used compiler infrastructures. The system addresses a critical gap where frontier models experience a 60% performance decline when tackling compiler bugs compared to common software bugs. For senior engineers maintaining compiler infrastructure in production systems, this work provides a practical framework for integrating LLMs into compiler debugging workflows without requiring deep compiler expertise.

## Why This Matters for Practitioners
If you're responsible for maintaining LLVM-based toolchains in production systems (such as those powering ML frameworks or Rust), this paper demonstrates that simply using general-purpose LLM agents like mini-SWE-agent will fail to resolve 60% of compiler bugs compared to standard software bugs. Instead of trying to shoehorn general LLM tools into compiler debugging, the authors demonstrate that a compiler-specific harness like llvm-autofix can increase resolution rates by 22% across multiple models. For your team, this means: (1) consider building compiler-specific tooling instead of relying on general LLM agents, (2) focus on creating reproducible bug reports with minimal reproducers (like those in llvm-bench), and (3) be aware that even with the right tools, the hardest compiler bugs remain challenging (only GPT 5 resolved 21% of hard issues).

## Problem Statement
Fixing compiler bugs is like trying to repair a complex clock without seeing the internal gears - you only get the ticking sound and the broken face, but not the mechanisms inside. Unlike standard software bugs that often come with clear descriptions, compiler crashes (like stack traces) or miscompilations (like input-output pairs) provide limited context to LLMs. The paper shows that this lack of description leads to a 60% performance drop for frontier models when compared to general software bug fixing.

## Proposed Approach
The authors propose a three-part harness called llvm-autofix, consisting of compiler-specific tools, a benchmark for reproducible LLVM bugs, and a minimal agent for fixing these bugs. The tools provide an agent-friendly interface to LLVM's internals, the benchmark (llvm-bench) contains 334 reproducible bugs with varying difficulty levels, and the minimal agent (llvm-autofix-mini) follows a "Setup → Reason → Generate → Validate" workflow to produce and validate patches.

```python
def llvm_autofix_mini_agent(bug):
    # Stage I: Setup
    setup_environment(bug)
    pause_at_breakpoint(bug)
    
    # Stage II: Reason (ReAct loop)
    while not root_cause_found:
        inspect_internal_state()
        analyze_root_cause()
    
    # Stage III: Generate (ReAct loop)
    while not patch_validated_online:
        generate_patch()
        test_patch_online()
    
    # Stage IV: Validate (offline)
    return validate_patch_offline()
```

## Key Technical Contributions
The paper's key technical contributions are:

1. **Compiler-specific tooling that abstracts complexity**: The harness provides tools like `reproduce()`, `debug()`, and `test()` that handle environment setup, reproducibility checks, and validation. For example, `reproduce()` validates whether a bug can be triggered faithfully using the reproducer, eliminating the need for agents to build and run LLVM themselves. This tooling abstracts away the complexity of compiler infrastructure, allowing agents to focus on core aspects of bug localization and repair.

2. **Targeted benchmark construction with difficulty splits**: The authors built llvm-bench by automatically collecting and validating 334 reproducible LLVM bugs, categorised into easy (76.3%), medium (13.2%), and hard (10.5%) based on the number of files needing changes. Unlike general benchmarks, this focuses specifically on middle-end compiler bugs (like crashes and miscompilations) that are most relevant to compiler engineers, with each issue accompanied by around 1.4 reproducers and 722 robust regression tests.

3. **Minimal agent leveraging runtime information**: Unlike general agents that rely only on static information, llvm-autofix-mini uses LLVM's runtime information from the reproducer to dynamically debug and identify root causes. The agent pauses LLVM at strategic breakpoints (before crashing functions for crash bugs, before first transformations for miscompilation bugs) and inspects internal states to identify errors. This approach is specifically tailored to the debugging workflow of compiler engineers who typically look at stack traces and intermediate states.

## Experimental Results
The evaluation showed that frontier models experienced a 60% average performance decline when moving from general software bugs (SWE-bench Verified) to LLVM bugs (llvm-bench live), with DeepSeek V3.2 achieving the best rate of 38.9% on llvm-bench live compared to its 60% on SWEV. The minimal agent llvm-autofix-mini outperformed the standard agent (mini-SWE-agent) by approximately 22% across models, with GPT 5 resolving 51.5% of issues (vs. 21.0% with mini-SWE-agent). However, even with the improved agent, resolution rates remained low for the hardest bugs (only GPT 5 resolved 21% of hard issues). The authors also performed expert review, revealing that the true capability of frontier models consistently remains below 22% on the hardest bugs.

## Related Work
The authors position their work against existing platforms like SWE-bench and SWE-agent, which effectively connect LLMs to standard bash tools for general software engineering tasks but exhibit limited efficacy in compiler engineering. Unlike these general-purpose tools, llvm-autofix is designed specifically for compiler debugging, addressing the unique challenges of compiler bugs (sparse descriptions, need for domain expertise) that make them fundamentally different from common software bugs.

## Limitations
The authors' own limitations include focusing only on middle-end LLVM bugs (not frontend or backend), and the benchmark currently focusing on crash and miscompilation bugs (not performance issues). The paper also acknowledges that even with their harness, the hardest compiler bugs remain challenging (only GPT 5 resolved 21% of hard issues), and LLVM's regression tests may be insufficient for validating model-generated patches. The authors note that the harness is currently limited to LLVM, though they plan to extend it to other compilers.

## Appendix: Worked Example
Consider a crash bug where LLVM crashes during compilation of a minimal reproducer (LLVM IR program) with the following stack trace: `#15 slpvectorizer::BoUpSLP::getEntryCost`. The agent first uses the `reproduce()` tool to validate the bug: it builds LLVM with the bug-containing version and runs the reproducer with `opt`, confirming the crash occurs and providing a cleaned stack trace. The agent then pauses at the crashing function using `debug()` and inspects the internal state with `eval(expr=WidePhi)`, revealing an incorrect pointer comparison (`%gep44 = getelementptr i8, ptr null, i64 %0` vs `%gep45 = getelementptr i8, ptr null, i64 %1`). With the root cause identified, the agent generates a patch to fix the comparison and uses `test()` to validate it against the reproducer and relevant regression tests from the SLPVectorizer component. The offline validation step then ensures the patch passes all tests before being accepted. This example demonstrates the workflow for an easy bug, but the paper notes that for hard bugs requiring changes across multiple files (e.g., 47.6 edited lines on average), resolution rates drop significantly (to 21% for GPT 5 on hard issues).

## References

- Yingwei Zheng, Cong Li, Shaohua Li, Yuqun Zhang, Zhendong Su, "Agentic Harness for Real-World Compilers", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20075

Tags: #systems #compiler-optimisation #agent-based-systems #tool-integration #llm-assisted-debugging
