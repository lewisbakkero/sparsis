---
title: "Stepwise: Neuro-Symbolic Proof Search for Automated Systems Verification"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19715"
---

## Executive Summary
Stepwise automates proof generation for interactive theorem proving (ITP) in software verification through a neuro-symbolic framework that combines fine-tuned LLMs with symbolic tools. It achieves 77.6% proof success rate on seL4 theorems, substantially surpassing prior LLM-based approaches, while reducing expert proof-writing effort by 71.1% in collaborative settings. Engineers building safety-critical systems should care because this enables more scalable formal verification without requiring deep ITP expertise.

## Why This Matters for Practitioners
If you're maintaining a safety-critical system with formal verification requirements, Stepwise directly addresses the two most significant barriers to adoption: the manual effort required to write proofs (20 person-years for seL4) and the lack of LLM effectiveness for domain-specific verification tasks. You can immediately integrate this approach into your verification workflow to automatically generate proof steps for existing theorem statements, reducing the time spent on routine proof construction by 70% or more while maintaining rigorous correctness guarantees. For teams with limited ITP expertise, this framework effectively lowers the barrier to entry for formal verification by automating the most tedious aspects of proof construction.

## Problem Statement
Current approaches to formal software verification resemble manual proof-writing by a single expert in a highly specialized field, where each theorem requires bespoke reasoning about domain-specific lemmas and tactics. Like a master clockmaker trying to replicate a complex timepiece by hand, existing LLM-based approaches struggle to generate complete, correct proofs due to two fundamental limitations: they lack the specialized knowledge required for verification domains (like the correct application of the wp tactic in seL4), and they suffer from insufficient high-quality training data (seL4 contains only ~20K theorems, with most proofs written in a procedural style where semantics are implicit in the theorem prover's state).

## Proposed Approach
The Stepwise framework operates as a best-first tree search over proof states, repeatedly querying a fine-tuned LLM for the next candidate proof step. On the neuro side, it fine-tunes LLMs using datasets of proof state-step pairs; on the symbolic side, it incorporates ITP tools to repair rejected steps, filter and rank proof states, and automatically discharge subgoals when search progress stalls. This synergy enables data-efficient LLM adaptation and semantics-informed pruning of the search space.

```python
def stepwise_proof_search(initial_state, max_iterations=100):
    tree = Tree([initial_state])
    for _ in range(max_iterations):
        current_state = tree.get_highest_scoring_node()
        candidate_steps = llm.generate_next_step(current_state)
        for step in candidate_steps:
            try:
                new_state = isabelle.execute(step, current_state)
                if new_state.is_valid():
                    tree.add_node(new_state)
                    if new_state.has_no_subgoals():
                        return new_state.get_proof_script()
            except ExecutionError as e:
                if e.reason == "tactic_mismatch":
                    candidate_steps.extend(repair_tactic(step))
                elif e.reason == "undefined_fact":
                    candidate_steps.extend(repair_premises(step))
        tree.filter_states()
        tree.rank_and_expand()
    return sledgehammer.complete_proof()
```

## Key Technical Contributions
Stepwise's technical innovations address the two key challenges in LLM-based theorem proving:

1. **Proof-state-aware step generation**: Rather than generating complete proofs from scratch, the framework generates one step at a time based on the current proof state (available hypotheses and target goal), which the authors demonstrate is more reliable than multi-step prediction. This approach leverages the fact that theorem proving can be modelled as a Markov decision process where the next step depends solely on the present state.

2. **Symbolic repair mechanisms**: For failed steps, Stepwise implements two explicit repair strategies: tactic repair (recombining premises with a curated set of frequently used tactics) and premise repair (replacing undefined facts with semantically similar ones from the proof context using edit distance), which transforms failed attempts into viable candidates.

3. **Hybrid filtering and ranking**: The framework uses Nitpick and QuickCheck to detect counterexamples and duplicate states before ranking remaining candidates using the LLM's log probabilities. This approach eliminates 52.3% of unprovable states and 44.2% of duplicate states in the "signed overflow" theorem example, dramatically improving search efficiency.

4. **Data-efficient training**: By extracting internal proof states from existing human-authored proofs (a five-step proof yields five training instances), Stepwise creates a dataset that's more suitable for training than surface-level proof text alone, addressing the data paucity challenge.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
Stepwise achieved 77.6% success rate on the seL4 benchmark, the highest among all evaluated approaches, significantly surpassing prior work: FVEL (8.2% success rate), Selene (20% on benchmark theorems), and standalone Sledgehammer (42.3%). For multi-step proofs (defined as proofs requiring more than one step), Stepwise solved 71% of theorems compared to FVEL's 12% and Selene's 18%. In an AI-human collaboration setting, Stepwise reduced expert proof-writing effort by 71.1% on average. The framework demonstrated strong generalisation across four additional Isabelle benchmarks (X86 Semantics, IEEE Floating Point, SATSolverVerification, and Code2Inv), confirming its applicability beyond the seL4 domain.

## Related Work
Stepwise builds upon but significantly extends prior work in AI for formal verification, particularly FVEL (which reported <10% success rate on seL4) and Selene (which achieved 20% success on a small benchmark set). Unlike retrieval-augmented approaches that require high-quality proof corpora (which are often unavailable in new verification contexts), Stepwise's neuro-symbolic framework works directly with the proof state and uses symbolic tools to fill data gaps. It also advances beyond standalone LLM fine-tuning by incorporating symbolic verification tools for pruning and repair, addressing the two key challenges of domain expertise and data paucity.

## Limitations
The authors acknowledge that Stepwise currently requires a fine-tuned LLM trained on existing human-authored proofs, though they propose future "bootstrapping" where training data could be extracted from successful search paths. The framework was evaluated primarily on Isabelle/HOL, so its applicability to other ITPs (like Coq or Lean) requires further investigation. The paper doesn't report on the computational cost of the tree search, though the authors note that the search space is pruned significantly through symbolic filtering.

## Appendix: Worked Example
Consider the theorem `cap_swap_valid_objs[wp]` from seL4, which states that swapping two valid capability slots preserves the absence of dangling pointers in the address space. The human-written proof consists of three steps: `apply (simp add: cap_swap_def)`, `apply wp`, and `apply (wp set_cap_valid_objs | simp split del: if_split)+`.

The automated process begins with the initial proof state containing the theorem and its hypotheses. The LLM generates candidate steps, and the first attempt fails with an "undefined fact: hoare_seq_ext" error. The system identifies this as a premise error and applies premise repair, substituting the closest matching premises from the library (using edit distance) for the undefined fact. The revised step becomes `apply (wp set_cap_valid_objs | simp split del: if_split)+`.

This step is executed, yielding a new proof state. Nitpick checks this state and finds no counterexamples (unlike 52.3% of states in the "signed overflow" example). The LLM ranks this state based on its cumulative log probability (score: 0.7), which is higher than other candidates (e.g., a state with score 0.2). The system expands the highest-scoring state, generating another step that completes the proof.

The final proof script requires only 3 steps compared to the 10+ steps a human might write, and the framework identifies the correct sequence in 4 search iterations, demonstrating its efficiency.

## References

- Baoding He, Zenan Li, Wei Sun, Yuan Yao, Taolue Chen, Xiaoxing Ma, Zhendong Su, "Stepwise: Neuro-Symbolic Proof Search for Automated Systems Verification", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19715

Tags: #formal-methods #theorem-proving #llm-integration #symbolic-ai #verification-tools
