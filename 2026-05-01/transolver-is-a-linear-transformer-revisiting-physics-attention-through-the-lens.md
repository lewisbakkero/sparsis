---
title: "Transolver Is a Linear Transformer: Revisiting Physics-Attention Through the Lens of Linear Attention"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37003"
---

## Executive Summary
LinearNO reimagines Transolver's Physics-Attention as a canonical linear attention mechanism, achieving state-of-the-art PDE solving performance while reducing parameters by 40% and computational cost by 36.2%. For engineers deploying physics-informed neural operators, this means better accuracy with lighter models that scale more efficiently to complex simulations.

## Why This Matters for Practitioners
If you're building production systems for computational fluid dynamics or structural analysis that rely on Neural Operators, this paper shows you can replace Transolver's Physics-Attention with LinearNO to get better results with fewer resources. Specifically, for any system solving PDEs with grid-based representations, implement the asymmetric attention projections (φ(Q) and ψ(K) learned independently) and remove the slice attention mechanism. This directly reduces inference costs, by 36.2% in computational cost and 40% in parameters, without sacrificing accuracy. For example, at the AirfRANS aerodynamic design benchmark, LinearNO reduced prediction errors by 60% over Transolver while cutting resource needs, making it more practical for real-time engineering applications.

## Problem Statement
Current approaches to PDE solving using neural operators often use attention mechanisms that scale quadratically with problem size, like trying to match every piece of a jigsaw puzzle against every other piece. Transolver's "Physics-Attention" attempted to solve this by slicing the problem into subgroups and doing attention within slices, but this created more friction than it resolved. The authors found that the symmetric slicing process (where slices are created and then decoded using the same mechanism) actually limited the model's ability to capture distinct physical states, like trying to solve a puzzle by grouping pieces by colour but then being unable to compare pieces across colour groups.

## Proposed Approach
LinearNO reframes Physics-Attention as linear attention through two key transformations: a generalisation step that removes the symmetry constraint between attention projections, and a simplification step that eliminates unnecessary slice attention. This creates a canonical linear attention mechanism that's both theoretically grounded and empirically superior. The network architecture follows Transolver's encoder-decoder structure but replaces Physics-Attention with LinearNO's two-step transformation.

```python
def linear_no(features):
    # Generalisation: asymmetric projections
    phi_Q = softmax(linear_Q(features))  # N x M
    psi_K = softmax(linear_K(features).T)  # M x N
    
    # Simplification: no slice attention, use identity
    return phi_Q @ (psi_K @ linear_V(features))
```

## Key Technical Contributions
The authors' two-step transformation fundamentally changes how attention works in PDE solving:

1. They reveal that Physics-Attention's slice attention operation was unnecessary due to the symmetric design constraint. By relaxing this symmetry (generalisation), they allow φ(Q) and ψ(K) to be learned independently, which creates more diverse slice weights and a more saturated attention matrix rank. This is why Figure 4 shows LinearNO's slice-weight matrix (Figure 4a) has more distinct patterns than Transolver's (Figure 4b), enabling more effective physical state representation.

2. They prove LinearNO is a Monte Carlo approximation of the integral kernel operator, a core theoretical foundation for Neural Operators. This means LinearNO maintains discretization-invariance while achieving better accuracy, unlike approaches that sacrifice theoretical grounding for efficiency. The authors verify this through Theorem 1, showing LinearNO converges to a continuous integral kernel operator as problem size increases.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
LinearNO achieves state-of-the-art results on six standard PDE benchmarks (Airfoil, Pipe, Plasticity, NS, Darcy, Elasticity), with over 10% relative improvement on Pipe, Plasticity, NS and Elasticity compared to the previous best method (Transolver++). On the AirfRANS aerodynamic design benchmark, LinearNO achieves a Spearman's correlation coefficient of 0.9992 for lift coefficient prediction, outperforming Transolver by 60% (Table 2). It reduces parameters by 40.0% and computational cost by 36.2% on average across benchmarks (Table 3), with results consistent across all six standard tasks.

The paper doesn't explicitly state statistical significance for the improvements, but provides relative L2 errors across multiple datasets and baselines, including FNO, U-FNO, GEO-FNO, Transolver, and Transolver++. The authors use relative L2 error (with lower values better) and Spearman's correlation coefficient as evaluation metrics.

## Related Work
LinearNO builds on Neural Operator research (Kovachki et al. 2023), which aims to solve PDEs by learning mappings between function spaces while maintaining discretization-invariance. It extends Transformer-based PDE solvers (OFormer, FactFormer, Transolver) by revealing that Physics-Attention is essentially a special case of linear attention. Unlike methods that compress sequence length through latent spaces (LNO, AROMAP), LinearNO derives a more direct linear attention mechanism through theoretical analysis of the existing approach.

## Limitations
The paper focuses on standard PDE benchmarks but doesn't test on more complex problems with multiple physical phenomena or non-standard boundary conditions. The ablation study (Table 4) shows that the generalisation step improves performance in most cases, but the authors don't provide analysis of how LinearNO would scale to problems with millions of grid points. The paper also doesn't compare against graph-based neural operator approaches, which could represent alternative architectures for handling spatial relationships.

## Appendix: Worked Example
Let's walk through a single step of LinearNO's mechanism using the Airfoil dataset with N=1024 grid points and M=64 slices (consistent with the paper's Figure 4).

1. Input: Feature matrix HN ∈ R^1024×64 (estimated feature dimension dh=64, paper doesn't specify)
2. Generalisation step:
   - Compute φ(Q) = Softmax(linear_Q(HN)) → produces 1024×64 matrix
   - Compute ψ(K) = Softmax(linear_K(HN)^T) → produces 64×1024 matrix
   - For grid point 1, φ(Q)[1,:] might be [0.05, 0.02, ..., 0.15] (summing to 1)
   - For slice 1, ψ(K)[:,1] might be [0.08, 0.01, ..., 0.12] (summing to 1)
3. Simplification step:
   - Compute intermediate result: ψ(K) @ linear_V(HN) → produces 64×64 matrix
   - Compute final output: φ(Q) @ intermediate result → produces 1024×64 matrix
4. Effect: For grid point 1, LinearNO considers all 64 slices (not just its slice) in a weighted way that reflects physical relevance, unlike Transolver's slice attention which would only consider information within the slice.
5. Result: The output H'N has improved accuracy (0.0049 L2 error vs. Transolver's 0.0053 on Airfoil) with fewer parameters.

## References

- **Code:** https://github.com/HiPRL/LinearNO
- Wenjie Hu, Sidun Liu, Peng Qiao, Zhenglun Sun, Yong Dou, "Transolver Is a Linear Transformer: Revisiting Physics-Attention Through the Lens of Linear Attention", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37003

Tags: #scientific-computing #neural-operators #linear-attention
