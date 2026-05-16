---
title: "On Sample-Efficient Generalized Planning via Learned Transition Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2602.23148"
---

## Executive Summary
This paper introduces a state-centric approach to generalised planning that explicitly models domain dynamics through learned transition functions, rather than predicting action sequences directly. By using Weisfeiler-Leman graph embeddings for size-invariant state representation and neuro-symbolic decoding, the method achieves better out-of-distribution generalisation with 95-99% fewer parameters than Transformer-based baselines. Engineers building planning systems for scalable domains should consider this approach to avoid state drift in long-horizon tasks.

## Why This Matters for Practitioners
If you're maintaining a production planning system for logistics or robotics that must handle problem instances larger than those seen during training (e.g., scaling from 5 to 20 objects), this paper demonstrates that explicitly modelling domain dynamics can prevent catastrophic failure. Specifically, for Blocksworld domains, a 1.1M-parameter XGBoost model using Weisfeiler-Leman embeddings achieved 50% extrapolation success (vs 13% for a 25M-parameter Transformer baseline) while requiring 96% fewer training instances. This means you can deploy smaller, more efficient models without sacrificing robustness to larger problem sizes, reducing both inference latency and maintenance costs in production.

## Problem Statement
Current Transformer-based planners like PlanGPT bypass explicit transition modelling, directly predicting action sequences. This leads to "state drift" where the model loses track of the world state during long planning horizons, analogous to a navigation system trained only on city maps for small towns failing when routing through a metropolis: it might correctly turn left at the first intersection (in-distribution) but generate contradictory turns later (out-of-distribution) because it never learned the underlying road network dynamics (i.e., how street layouts evolve from one block to the next).

## Proposed Approach
The approach formulates generalised planning as a transition-model learning problem: a neural model predicts successor states (st+1) given current state (st) and goal (g), rather than direct action prediction. State representations use Weisfeiler-Leman graph embeddings for size invariance, then a residual transition model predicts the next state embedding. Finally, neuro-symbolic decoding matches predicted embeddings to valid symbolic successors using the domain's transition function (γ), recovering executable actions. This enforces symbolic validity at every step while learning domain dynamics.

```python
def generate_plan(initial_state, goal, domain, transition_model):
    state = initial_state
    plan = []
    while not goal in state:
        # Predict next state using residual transition
        next_embed = state_embedding(state) + transition_model(state_embedding(state), goal_embedding(goal))
        
        # Find nearest valid successor state via symbolic search
        valid_successors = [domain.transition(state, action) for action in domain.applicable_actions(state)]
        next_state = min(valid_successors, key=lambda s: distance(state_embedding(s), next_embed))
        
        # Recover action that caused this transition
        action = [a for a in domain.actions if domain.transition(state, a) == next_state][0]
        plan.append(action)
        state = next_state
    return plan
```

## Key Technical Contributions
The paper makes three distinct technical contributions:
1. **Residual transition modelling**: The core innovation uses a delta prediction (st+1 = st + Δ) instead of direct state prediction, explicitly encoding frame axioms (e.g., "predicates not affected by actions remain unchanged"). In Blocksworld, this increased extrapolation success from 25% to 50% by reducing regression variance for sparse STRIPS transitions (where most predicates remain stable).
2. **Size-invariant relational embeddings**: Weisfeiler-Leman (WL) graph kernels convert variable-sized relational states into fixed-dimensional embeddings (D depends only on domain, not object count). Fixed-size factored encodings (which assign slots per object) fail at extrapolation (0% success), while WL embeddings achieve 50% success in Blocksworld by preserving permutation invariance and size independence.
3. **Neuro-symbolic decoding interface**: The predicted successor embedding is matched against all valid symbolic successors (via domain operators), guaranteeing symbolic validity at every timestep. This corrects neural prediction errors on the fly, unlike action-centric methods that accumulate drift without validation.

## Experimental Results
The paper evaluated four IPC domains (Blocksworld, Gripper, Logistics, VisitAll) with object-count splits (training: 4, 7 blocks; extrapolation: 9, 17 blocks). Key results:
- **Blocksworld extrapolation**: WL-XGB (delta) achieved 50% success vs 13% for SymT (state-of-the-art action-centric baseline).
- **Model size**: State-centric models used 1.1M, 2.1M parameters (LSTM) or 128K, 819K nodes (XGBoost), versus 25, 220M parameters for baselines.
- **Training data**: State-centric models trained on 9 raw instances (no augmentation), while SymT used augmented data.
- **VisitAll extrapolation**: WL-XGB (delta) achieved 100% success vs 64% for SymT.
- **Gripper extrapolation**: SymT still outperformed state-centric approaches (79% vs 42%) due to domain-specific causal structures, but state-centric models used 96% fewer parameters.

## Related Work
The paper positions itself against two paradigms: (1) Action-centric sequence prediction (Plansformer, PlanGPT) that bypasses explicit transition modelling, and (2) Heuristic-based state-space learning (e.g., STRIPS-HGN). It improves over these by explicitly learning transition dynamics with size-invariant representations, avoiding state drift. It also relates to model-based RL (Ha & Schmidhuber, 2018) but applies to symbolic planning with neuro-symbolic validation.

## Limitations
The authors acknowledge failure in hierarchical domains like Logistics (0% extrapolation success) due to "deep multi-layer causal coupling across object types" that cannot be preserved by one-step successor matching. This limitation is structurally inherent to the approach, not a data issue. Additionally, the paper doesn't validate whether residual modelling generalizes to non-STRIPS domains.

## Appendix: Worked Example
Consider a Blocksworld test instance with 10 blocks (extrapolation beyond training size of 7 blocks). Initial state: blocks A,B,C on table; goal: stack A on B and C on D (D present in test instance).
1. **State encoding**: The state-goal pair (st, g) is converted to a WL graph embedding (D=1024 dimensions) via 3 iterations of colour refinement. Object counts don't affect embedding size.
2. **Transition prediction**: The residual model predicts: st+1 = st + f(st, g). For moving A to B, f(st, g) = [0.2, -0.1, 0.3, ...] (1024-dimensional delta vector).
3. **Neuro-symbolic decoding**: Valid successors (Succ(st)) are enumerated: {stack A on B, stack B on A, ...}. The embedding of "stack A on B" (ϕ(s′)) has the smallest Euclidean distance to st + f(st, g).
4. **Action recovery**: The unique action causing this transition is "stack A on B", which is executed. The process repeats until goal is reached (e.g., 5 steps for this instance).

See Section 3.4 for the residual formulation and neuro-symbolic decoding mechanism.

## References

- **Code:** https://github.com/ai4society/state-centric-gen-planning
- Nitin Gupta, Vishal Pallagani, John A. Aydin, Biplav Srivastava, "On Sample-Efficient Generalized Planning via Learned Transition Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2602.23148

Tags: #automated-planning #transition-modelling #weisfeiler-leman #neuro-symbolic
