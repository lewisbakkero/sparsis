---
title: "Alternating Diffusion for Proximal Sampling with Zeroth Order Queries"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19633"
---

## Executive Summary
This paper introduces a diffusion-based approximate proximal sampler that operates solely with zeroth-order information of the potential function, eliminating the need for rejection sampling while permitting flexible step sizes. The method converges significantly faster than existing implementations and handles disconnected target distributions where gradient-based approaches fail. For practitioners building sampling systems in production, this means up to 10× faster convergence with deterministic runtime budgets without requiring gradient information.

## Why This Matters for Practitioners
If you're implementing Markov chain Monte Carlo (MCMC) sampling in production systems for Bayesian inference or generative modelling, this paper directly addresses two painful pain points: the unpredictability of rejection sampling and the need for small step sizes that slow convergence. For a system currently using proximal sampling with rejection sampling (like RGO-based implementations), switching to this method would reduce iteration counts from ~950 to ~100 while eliminating the stochastic runtime cost of rejection sampling. For systems targeting non-convex, multi-modal distributions where gradient information is unavailable (e.g., when f(x) is non-differentiable or expensive to compute), this zeroth-order method provides a viable alternative to gradient-based approaches like MALA or ULA. Specifically, you should consider integrating this method when your target distribution has disconnected modes or when you can't compute gradients reliably.

## Problem Statement
Current proximal sampling implementations are like navigating through a city using only a map that shows one street at a time: you can see and follow the current route precisely (the Gaussian convolution), but you're stuck at intersections where you need to make a turn (the reverse step), requiring you to repeatedly ask for directions (rejection sampling). The larger the city block (step size), the more times you get lost and have to ask again, wasting time and energy. This is why proximal sampling implementations have been limited to small step sizes, resulting in many iterations to converge. The authors' breakthrough is like having navigation software that can estimate the best turn without stopping to ask for directions - it calculates the route based on current position and map data, allowing you to take larger steps confidently.

## Proposed Approach
The method alternates between forward heat flow (adding Gaussian noise) and approximate reverse dynamics (denoising using a Gaussian-mixture-based score estimator), creating a cycle that converges to the target distribution. The key innovation is treating the intermediate particle distribution as a Gaussian mixture to enable direct Monte Carlo score estimation without gradient information or rejection sampling.

```python
def alternating_diffusion(f, x0, h, K, T, M):
    """Alternating Diffusion for Proximal Sampling with Zeroth Order Queries
    
    Args:
        f: Potential function (zeroth-order oracle)
        x0: Initial particles (N x d)
        h: Step size
        K: Outer iterations
        T: Diffusion steps for reverse dynamics
        M: Number of interim samples for score estimation
        
    Returns:
        xK: Particles after K iterations
    """
    x = x0
    for k in range(K):
        # Forward step: Gaussian perturbation (heat flow)
        y = x + np.sqrt(h) * np.random.randn(*x.shape)
        
        # Initialize diffusion
        z = y + np.sqrt(h) * np.random.randn(*y.shape)
        
        # Reverse dynamics: approximate score using Gaussian mixture
        for t in range(T, 0, -1):
            sigma_t = np.sqrt(h - (t-1)*(h/T))
            # Compute score estimator via Monte Carlo
            for i in range(z.shape[0]):
                # Draw M samples from Gaussian mixture
                z_samples = []
                for j in range(M):
                    z_sample = np.random.multivariate_normal(
                        mean=estimate_mean(z[i], y), 
                        cov=sigma_t**2 * np.eye(z.shape[1])
                    )
                    z_samples.append(z_sample)
                # Evaluate f at samples
                f_vals = [f(z_sample) for z_sample in z_samples]
                # Compute weights
                weights = np.exp(-np.array(f_vals))
                weights /= np.sum(weights)
                # Euler-Maruyama step
                z[i] = z[i] + (h/T) * np.sum(weights * (np.array(z_samples) - z[i]) / (sigma_t**2)) + np.sqrt(h/T) * np.random.randn(z.shape[1])
        
        x = z
    return x
```

## Key Technical Contributions
The paper makes two key technical contributions that differentiate it from prior approaches:

1. **Gaussian Mixture Score Estimation**: Instead of using rejection sampling to implement the reverse step in proximal sampling, they treat the intermediate particle distribution as a Gaussian mixture. This allows direct estimation of the score function using only evaluations of the potential function f. Specifically, they derive an expression for the surrogate score as an expectation over a Gaussian-mixture posterior (Equation 11), which can be approximated via Monte Carlo sampling. This eliminates the need for gradient information (unlike Langevin methods) and rejection sampling (unlike RGO implementations), with the computational cost scaling linearly with particles N and Monte Carlo samples M.

2. **Theoretical Convergence Guarantee with Flexible Step Sizes**: They prove that their method inherits the exponential convergence rate of proximal sampling under isoperimetric conditions when the score estimation error is controlled. The key insight is that the discretization error (from splitting step size h into T substeps) scales as O(h/T), allowing them to choose T = O(h) while maintaining the same overall computational cost. This means they can take larger step sizes h that reflect global properties of the target distribution (like inter-mode distances) rather than being limited to local smoothness of f, which they empirically demonstrate leads to faster convergence in both Gaussian Lasso mixture and two-tori experiments.

For the second contribution, see Appendix for a step-by-step worked example with concrete numbers showing how the error analysis works in practice.

## Experimental Results
In the Gaussian Lasso mixture experiment (d=5), their method with step size h=1/10 converged to comparable KL divergence in ~100 iterations, while RGO baseline with h=1/135 required ~950 iterations (≈9500 RGO updates). With particle interactions (N=100), the method achieved the same accuracy as RGO in about 10× fewer iterations (100× faster when accounting for thinning). Figure 3 shows the method matching the reference distribution by ~250 iterations, while RGO required ~950 iterations for the same accuracy.

In the uniform distribution experiment over two disjoint tori, In-and-Out converged to the near torus (T1) but failed to reach the distant torus (T2), while their method gradually drove particles toward T2 (Figure 4). The paper doesn't specify exact metrics for the two-tori experiment, but the visual results clearly demonstrate successful exploration of disconnected modes where gradient-based methods fail.

## Related Work
The paper positions itself against two lines of work: traditional proximal sampling (Liang & Chen, 2023b) that relies on rejection sampling for the reverse step, and diffusion-based Monte Carlo methods (Huang et al., 2024a; He et al., 2024) that estimate scores via learned models or auxiliary samplers. Unlike diffusion-based pushforward methods (which always start from a Gaussian), the authors' approach alternates between forward and reverse dynamics, removing the restriction of restarting from a Gaussian base. It also differs from ZOD-MC (He et al., 2024), which uses a proximal sampler within an inner loop for score estimation, as they invert this structure to directly implement proximal sampling.

## Limitations
The paper acknowledges that the theoretical analysis assumes large N and M for accurate score estimation, which may not hold for small particle counts. They don't test the method on high-dimensional problems (d > 10) in the experiments, though the theoretical framework suggests it should scale. The paper also doesn't address how to choose the optimal step size h for a given target distribution, though they mention that larger h is favorable when CLSI ≫ h. The experiments focus on low-dimensional problems (d=5), so the method's scalability to high-dimensional settings remains untested.

## Appendix: Worked Example
Let's walk through the Gaussian mixture score estimation mechanism with concrete numbers. Suppose we have:
- Target distribution πX(x) ∝ exp(-f(x)) where f(x) = ||x||² (for simplicity in demonstration)
- Current particles at iteration k: Xk = {x1, x2} = {(0,0), (1,0)} in d=2
- Step size h = 1
- For the reverse dynamics, we need the score estimate at time t with σ²_t = 0.5

First, the algorithm forms the intermediate distribution πY as:
πY(y) ∝ ∫ exp(-||x||² - ||x-y||²/2) dx

For our two particles, the surrogate distribution ˆqk+1 is constructed as:
ˆqk+1(x | Yk+1/2, Xk) ∝ Σj exp(-f(x) - ||x-yj||²/(2h)) * ˆqk+1/2(yj | Xk)

With ˆqk+1/2(y | Xk) = (1/N) Σi N(y; xi, hId), we have:
ˆqk+1/2(y | Xk) = 0.5 N(y; (0,0), Id) + 0.5 N(y; (1,0), Id)

For a specific y = (0.5, 0), we calculate:
- N(y; (0,0), Id) = 1/√(2π) exp(-0.125) ≈ 0.37
- N(y; (1,0), Id) = 1/√(2π) exp(-0.125) ≈ 0.37
- So ˆqk+1/2(y | Xk) = 0.5*0.37 + 0.5*0.37 = 0.37

Then, for the score estimator at z = (0.5, 0):
- Draw M=5 samples from the posterior distribution (Equation 13)
- For each sample x0, compute f(x0) = ||x0||²
- Calculate weights using exp(-f(x0))
- Compute the weighted average of (x0 - z)/σ²_t

For example, with five samples:
- x0 samples: [(0.4, 0.1), (0.6, -0.2), (0.5, 0.3), (0.45, -0.1), (0.55, 0.2)]
- f(x0) values: [0.17, 0.36, 0.34, 0.22, 0.34]
- Weights: [exp(-0.17), exp(-0.36), exp(-0.34), exp(-0.22), exp(-0.34)] ≈ [0.84, 0.70, 0.71, 0.80, 0.71]
- Normalised weights: [0.20, 0.17, 0.17, 0.19, 0.17]
- (x0 - z) values: [(-0.1, 0.1), (0.1, -0.2), (0.0, 0.3), (-0.05, -0.1), (0.05, 0.2)]
- Weighted average: [(-0.1*0.20 + 0.1*0.17 + 0*0.17 -0.05*0.19 + 0.05*0.17), (0.1*0.20 -0.2*0.17 + 0.3*0.17 -0.1*0.19 + 0.2*0.17)] ≈ [(-0.02 + 0.017 + 0 - 0.0095 + 0.0085), (0.02 - 0.034 + 0.051 - 0.019 + 0.034)] ≈ [-0.014, 0.042]

This gives the score estimate used in the Euler-Maruyama step to approximate the reverse dynamics without gradient information.

## References

- Hirohane Takagi, Atsushi Nitanda, "Alternating Diffusion for Proximal Sampling with Zeroth Order Queries", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19633

Tags: #machine-learning #sampling-algorithms #zeroth-order-optimisation #diffusion-models #nonconvex-optimisation
