---
title: "PDDL Axioms Are Equivalent to Least Fixed Point Logic (Extended Version)"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2510.14412"
---

## Executive Summary
This paper establishes that PDDL axioms, a key feature for expressing indirect relationships in planning domains, are formally equivalent to least fixed point logic (LFP). The authors prove that both the restricted variant (AP0), which limits negative predicate occurrences to basic predicates, and the more general variant (AP), which allows negative occurrences of derived predicates under stratifiability, express exactly the same logical queries as LFP. This theoretical clarification resolves a longstanding ambiguity in planning literature about PDDL's expressive power.

## Why This Matters for Practitioners
If you're designing planning systems or working with PDDL-based planners like Fast Downward, this paper clarifies that you can safely use the more expressive AP variant (permitting negative derived predicates when stratifiable) without theoretical limitations. You no longer need to restrict yourself to the AP0 variant when writing complex domain models. The paper also provides a practical compilation technique to eliminate negative occurrences of derived predicates from axioms, which is crucial for planning systems that require stratified representations. For engineers implementing PDDL parsers or translators, this means you can now safely transform AP axioms into AP0 form without losing expressiveness, simplifying integration with existing planning systems that expect the restricted format.

## Problem Statement
Today's planning systems face a subtle but critical ambiguity: the PDDL standard restricts negative predicate occurrences in axiom bodies to only basic predicates (AP0), while many planning papers use a more general variant (AP) that permits negative occurrences of derived predicates as long as the axiom set is stratifiable. This creates confusion about whether the more expressive AP variant is actually more powerful. Like trying to use a standard USB-C connector with a device that expects USB-2.0 but claiming it's "more powerful" because it can handle legacy cables, this paper proves that both variants are actually equivalent, resolving the theoretical confusion.

## Proposed Approach
The authors conduct a formal comparison of three logical formalisms: PDDL axiom programs (AP and AP0), least fixed point logic (LFP), and stratified Datalog. They prove that both AP and AP0 are equivalent to LFP in expressive power. This theoretical analysis clarifies that negative occurrences of derived predicates in axioms don't increase expressiveness when the set of axioms is stratifiable. The paper also introduces a practical compilation technique that transforms any AP axiom program into an AP0 program without changing the meaning of the axioms, which is particularly useful for planning systems that require stratified representations.

```python
def compile_axioms(axiom_program):
    """Compiles an AP axiom program into an equivalent AP0 program.
    
    Args:
        axiom_program: A stratified axiom program (AP), possibly containing negative occurrences of derived predicates.
        
    Returns:
        An equivalent semipositive axiom program (AP0) with all negative occurrences of derived predicates eliminated.
    """
    # Step 1: Identify all derived predicates and their dependencies
    derived_predicates = get_derived_predicates(axiom_program)
    
    # Step 2: For each derived predicate P, create a new positive-only axiom
    new_axioms = []
    for P in derived_predicates:
        # Create a new axiom for the positive version of P
        positive_axiom = create_positive_axiom(P, axiom_program)
        new_axioms.append(positive_axiom)
        
        # Replace negative occurrences of P in other axioms with the positive version
        for axiom in new_axioms:
            if "¬" in axiom.body and P.name in axiom.body:
                axiom.body = replace_negative_with_positive(axiom.body, P)
    
    return SemipositiveAxiomProgram(new_axioms)
```

## Key Technical Contributions
The paper's theoretical contributions clarify the relationship between planning formalisms with surprising precision.

1. The authors prove that both AP (the general variant permitting negative derived predicate occurrences under stratifiability) and AP0 (the restricted variant limiting negative occurrences to basic predicates) are equally expressive as least fixed point logic (LFP), establishing AP = AP0 = LFP. This resolves a long-standing ambiguity in the planning literature.

2. They demonstrate that the commonly used Fast Downward planning system's preprocessing step, which translates tasks into stratified Datalog form, can lead to non-stratifiable programs when certain axioms are present. This explains why the system sometimes requires special handling for complex axioms.

3. The paper provides a concrete compilation technique that eliminates negative occurrences of derived predicates from PDDL axioms while preserving their meaning. This is implemented through a systematic transformation of axioms that replaces negative occurrences of derived predicates with positive equivalents, leveraging the stratification property to maintain semantic equivalence. See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
This is a theoretical paper and does not include empirical results, experiments, or comparisons against baselines. The authors focus exclusively on formal proof of equivalence between logical formalisms. The paper doesn't report performance metrics, dataset sizes, or statistical significance for claims as these are not applicable to theoretical results.

## Related Work
The paper positions itself as a clarification of the theoretical foundation for PDDL axioms, building on the work of Thiébaux, Hoffmann, and Nebel (2005) who established that axioms increase expressive power but didn't consider the difference between the restricted and general axiom variants. It also clarifies the relationship to stratified Datalog, which is widely used in planning systems like Fast Downward (Helmert, 2006, 2009). The authors note that previous work by Thiébaux et al. incorrectly asserted that PDDL axioms could be rewritten to stratified Datalog without loss of expressiveness, when in fact the more general PDDL axioms are strictly more expressive than stratified Datalog, this paper shows they are equivalent to LFP instead.

## Limitations
The paper is purely theoretical and doesn't address practical implementation challenges in planning systems. It doesn't provide performance data about the compilation technique or how it affects planning search efficiency. The authors don't address whether the compilation process might increase the size of the planning problem representation, though they acknowledge this as a potential trade-off. The paper's focus on theoretical equivalence doesn't consider the practical implications for users who might prefer simpler axiom formulations.

## Appendix: Worked Example
Let's walk through the compilation process for a simple AP axiom program containing a negative occurrence of a derived predicate, using the example from the paper.

Consider the following AP axiom program with two strata:
```
Stratum 1: 
  path(x, y) ← E(x, y) ∨ ∃z(E(x, z) ∧ path(z, y))
Stratum 2:
  acyclic() ← ∀x ¬path(x, x)
```

The second axiom has a negative occurrence of the derived predicate `path` in its body, which is invalid in AP0 but permitted in AP. To compile this into AP0, we follow the paper's method:

1. For the derived predicate `path`, we create a positive-only version `path+(x, y)` by removing the negative occurrences from its definition:
   ```
   path+(x, y) ← E(x, y) ∨ ∃z(E(x, z) ∧ path+(z, y))
   ```

2. We replace the negative occurrence of `path` in the second axiom with the new positive predicate:
   ```
   acyclic() ← ∀x ¬path+(x, x)
   ```

3. To maintain equivalence, we add an additional axiom that defines the relationship between `path` and `path+`:
   ```
   path(x, y) ← path+(x, y)
   path+(x, y) ← path(x, y)
   ```

This transformed program now has all negative occurrences of derived predicates eliminated. The compiled program is semipositive (single-stratum) and equivalent to the original AP program. For a planning domain with universe size \(n = 4\) (as in the paper's example), the fixed-point computation for `path+` would require at most \(n-1 = 3\) iterations to reach the fixed point, while the original `path` would have required the same number of iterations due to the negative occurrence being handled through the positive version.

## References

- Claudia Grundke, Gabriele Röger, "PDDL Axioms Are Equivalent to Least Fixed Point Logic (Extended Version)", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2510.14412

Tags: #automated-planning #logical-reasoning #fixed-point-logic #datalog #pddl
