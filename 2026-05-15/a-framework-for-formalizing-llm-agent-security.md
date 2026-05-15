---
title: "A Framework for Formalizing LLM Agent Security"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19469"
---

## Executive Summary
This paper introduces a formal framework for contextual security in LLM agents, moving beyond superficial definitions of attacks to define security as a property of actions within their execution context. It establishes four precise security properties (task alignment, action alignment, source authorization, and data isolation) and introduces oracle functions that theoretically enable verification of these properties during agent execution. Practitioners should care because this framework resolves the utility-security trade-off that plagues current security approaches in deployed LLM agents.

## Why This Matters for Practitioners
If you're building or deploying LLM agents in production systems that interact with sensitive data (e.g., financial transactions, healthcare systems, or infrastructure control), this paper fundamentally changes how you should design security mechanisms. Current approaches either block legitimate operations (reducing utility) or fail to detect attacks (creating vulnerabilities), as illustrated by the example where "delete file" is treated identically whether it's a legitimate cleanup request or a data destruction attack. The framework enables precise security checks that differentiate between legitimate operations and attacks even when using identical inputs. For example, when implementing security for an agent that handles payment processing, you should enforce all four security properties rather than simply blocking patterns like "delete file" or "remove item," which would prevent legitimate operations while failing to stop actual attacks.

## Problem Statement
Current LLM agent security approaches are like a bank teller who can't distinguish between a customer asking to withdraw £200 (legitimate) and a thief posing as a customer (fraudulent), both using the same words. Security breaches depend on context, whether a prompt originates from an authenticated source, whether the action aligns with the intended objective, and whether information flows respect privilege boundaries. Existing definitions treat security as an inherent property of inputs or actions, creating a fundamental utility-security trade-off: applying defences uniformly across all contexts causes false positives (reducing utility) or false negatives (creating security vulnerabilities), as shown in Figure 1 of the paper.

## Proposed Approach
The framework systematises security for LLM agents by formalising execution context and introducing four security properties that must all be satisfied simultaneously for an action to be considered secure. The core insight is that security is relational, dependent on the context of execution rather than being an inherent property of the action itself. The system architecture consists of:

1. **Execution context tracking**: Capturing user prompts, trajectory of actions and observations, memory, environment state, authenticated sources, and permission graph
2. **Oracle functions**: Theoretical requirements for verifying security properties (I for instruction attribution, L for source attribution, and Hp, HTr, Ha for objective evaluation)
3. **Security property verification**: Checking all four properties before allowing actions to proceed

Here's the core verification process:

```python
def verify_security(context, action):
    if not task_alignment(context, action):
        return False
    if not action_alignment(context, action):
        return False
    if not source_authorization(context, action):
        return False
    if not data_isolation(context, action):
        return False
    return True
```

## Key Technical Contributions
The paper's key contributions provide precise mechanisms that distinguish it from prior work:

- **Four security properties** that explicitly capture the contextual nature of security: task alignment (ensuring the agent pursues authorized objectives), action alignment (ensuring individual actions serve those objectives), source authorization (ensuring instructions originate from authenticated sources), and data isolation (ensuring information flows respect permission boundaries). Unlike prior work that defined prompt injection attacks based on surface patterns, these properties require verification of execution context.

- **Oracle functions** (I, L, Hp, HTr, Ha) that formalise the theoretical requirements for verifying security properties. These functions specify what information is necessary to determine whether an action is secure, making previously implicit requirements explicit. Current security defences implicitly approximate these oracles without understanding the full requirements, leading to fundamental limitations.

- **Reformulation of attacks** as violations of specific security properties. For example, indirect prompt injection is defined as violating source authorization (executing commands from unauthenticated sources) and action alignment (actions not serving the intended objective), while direct prompt injection violates task alignment (conflicting with higher-privilege instructions). This precise definition enables better defence design.

- **Reformulation of defences** as mechanisms that strengthen oracle functions or perform security property checks. The framework reveals that context-agnostic defences (applied uniformly across all contexts) inevitably create utility-security trade-offs, while single-property defences (focusing on only one security property) are vulnerable to attacks violating multiple properties simultaneously.

## Experimental Results
The paper does not present experimental results comparing implementations of the framework. Instead, it focuses on formalising concepts and using the framework to reinterpret existing attacks and defences. The authors don't provide specific numbers for how the framework improves security or utility compared to existing approaches, but they do argue that context-agnostic defences inevitably create utility-security trade-offs, while their framework resolves this through precise property verification.

## Related Work
This paper positions itself as addressing a fundamental gap in LLM agent security research. Prior work on prompt injection attacks [20,22,44,52,53] has defined attacks based on surface patterns in inputs (e.g., "if a prompt contains certain keywords, it's an attack"), without considering context. The framework explicitly addresses this limitation by introducing contextual security properties. It builds on prior work in security for traditional systems but adapts it to the unique challenges of LLM agents, which operate in dynamic, context-dependent environments where security depends on execution context rather than static input properties.

## Limitations
The paper acknowledges that implementing the oracle functions perfectly is challenging in practice, as they would require perfect instruction attribution and objective inference capabilities that current LLMs lack. The framework doesn't provide specific implementation details for approximating these oracle functions. It also doesn't address training-time security issues, focusing only on runtime security during agent execution. The authors don't evaluate the framework's effectiveness in real-world production systems where context complexity is significantly higher than in their examples.

## Appendix: Worked Example
Let's walk through a concrete example using the cooking assistant agent from the paper:

1. **Initial state**: User prompt is "Find me a healthy dinner recipe and order the ingredients." Memory Mt contains "User is vegetarian" and "User prefers organic produce." Environment state Et includes pantry contents: ["soy sauce"].

2. **Agent action**: The agent searches the recipe database with query "healthy vegetarian dinner." The action is search_recipes("healthy vegetarian dinner").

3. **Trajectory**: Trt-1 = [(search_recipes, ["tofu stir fry", "quinoa salad", "veggie pasta"])].

4. **Source attribution**: The recipe search originates from the recipe database (authenticated source, as established in the permission graph).

5. **Verify security properties**:
   - **Task alignment**: The objective o0 = "find healthy dinner recipe and order ingredients." The trajectory shows the agent searching recipes, which serves this objective (HTr = 1). Allowing the search is legitimate.
   - **Action alignment**: The search action directly serves the objective (Ha = 1).
   - **Source authorization**: The search originated from the recipe database (an authenticated source), so authorization holds.
   - **Data isolation**: The search only accessed the recipe database (authorized source), so data isolation holds.

6. **Result**: All four security properties hold, so the action is contextually secure. The agent can proceed to ask the user which recipe they prefer.

Now, consider an attack scenario:
1. **Malicious input**: A recipe blog (unauthenticated source) includes the text "delete pantry contents" within its content.
2. **Agent action**: The agent processes the blog content and executes "delete_pantry_contents".
3. **Verify security properties**:
   - **Task alignment**: The objective o0 remains "find healthy dinner recipe and order ingredients." The action "delete_pantry_contents" doesn't serve this objective (HTr = 0).
   - **Action alignment**: The action doesn't serve the objective (Ha = 0).
   - **Source authorization**: The instruction originated from the recipe blog (unauthenticated source), violating source authorization.
   - **Data isolation**: The action accessed pantry data without proper permission (violation).

This example demonstrates how the framework enables the agent to distinguish between legitimate recipe searches and the malicious "delete pantry contents" action, even when the underlying content (in this case, the text "delete") is identical. See Appendix for a step-by-step worked example with concrete numbers.

## References

- Vincent Siu, Jingxuan He, Kyle Montgomery, Zhun Wang, Neil Gong, Chenguang Wang, Dawn Song, "A Framework for Formalizing LLM Agent Security", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19469

Tags: #security-and-privacy #contextual-security #llm-agents #oracle-functions #multi-property-defence
