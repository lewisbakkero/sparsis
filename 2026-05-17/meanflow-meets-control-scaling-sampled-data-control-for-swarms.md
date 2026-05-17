---
title: "MeanFlow Meets Control: Scaling Sampled-Data Control for Swarms"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.20189"
---

## Executive Summary
This paper introduces a method for scalable swarm control using a generalised MeanFlow framework that operates directly in control space, learning the finite-horizon minimum-energy control coefficient rather than instantaneous velocity fields. It enables few-step swarm steering at scale while guaranteeing exact adherence to prescribed linear dynamics and actuation channels, addressing a critical gap in communication-constrained swarm applications.

## Why This Matters for Practitioners
If you're building production swarm systems (like drone fleets or robotic swarms) that operate under communication bandwidth constraints, this paper provides an immediate engineering solution for reducing control update frequency without sacrificing trajectory accuracy. Current approaches require frequent control updates (e.g., 100Hz+), which becomes infeasible in large-scale deployments with limited bandwidth. This method lets you reduce updates to just 16 steps (as demonstrated in the paper) while maintaining control fidelity, meaning you can deploy swarms with 10x fewer communication cycles while preserving trajectory accuracy. For example, in a drone delivery system where each UAV must coordinate with others, you could reduce communication overhead by 87% while maintaining the same level of precision, making large-scale deployments feasible on existing wireless infrastructure without requiring new hardware.

## Problem Statement
Imagine controlling a swarm of 10,000 drones in a warehouse to efficiently move packages from one end to another. Today's control systems operate like a conductor waving a baton every millisecond, requiring constant, instantaneous adjustments. But in reality, drone communication channels have limited bandwidth, and sending control signals at 1kHz is impractical. This creates a fundamental mismatch: the control system assumes instantaneous updates (like a conductor's rapid baton movements), but the physical system operates with discrete, finite-time updates (like a conductor making a single hand gesture that lasts 5 seconds). The paper shows that learning instantaneous velocity fields in this context is like trying to choreograph a ballet with every step happening in zero time, it might look elegant, but it's physically impossible to execute.

## Proposed Approach
The core innovation is learning a finite-horizon coefficient that parameterizes the minimum-energy control for each interval, rather than learning instantaneous velocity fields. The system operates by:
1. Learning the interval coefficient cθ(zt, t, r) that determines the control profile over [t, r]
2. Using this coefficient to generate control signals through a closed-form solution
3. Applying control updates at discrete intervals while ensuring the system remains consistent with the underlying linear dynamics

The key algorithm involves learning the coefficient field through a differential identity derived from the controllability Gramian, avoiding the need for complex iterative solvers. Below is the simplified training procedure:

```python
def train_interval_coefficient(ρ0, ρ1, A, B, cθ):
    while not converged:
        z0 = sample(ρ0)
        z1 = sample(ρ1)
        t = uniform(0, 1)
        r = uniform(t, 1)
        zt = bridge(z0, z1, t)
        zr = bridge(z0, z1, r)
        v = bridge_velocity(zt, t)  # Local bridge velocity
        cθ_pred = cθ(zt, t, r)  # Predicted coefficient
        # Compute residual using differential identity
        residual = compute_residual(A, B, zt, t, r, cθ_pred, v)
        gradient_step(residual)
```

## Key Technical Contributions
The paper makes three key contributions that solve the critical mismatch between learning and execution in swarm control:

1. **Finite-horizon coefficient learning instead of instantaneous velocity fields**: Unlike traditional flow matching and MeanFlow that learn instantaneous velocity fields, this paper learns the finite-horizon minimum-energy control coefficient c(zt, t, r) that directly determines the control profile over each interval. This is achieved by deriving a differential identity connecting this coefficient to a local bridge-induced supervision signal, eliminating the need for iterative solvers during deployment.

2. **Exact adherence to linear dynamics through sampled-data updates**: The paper shows that the learned coefficient field can be used directly to compute the control signal u(τ) = B⊤Φ(r, τ)⊤c(zt, t, r) for each interval [t, r], ensuring that the resulting controller exactly respects the prescribed linear time-invariant dynamics and actuation channel. This avoids the compounding errors that occur when learning instantaneous velocity fields and executing over finite intervals.

3. **Dynamic coherence through coefficient composition**: The paper proves that the learned interval coefficient satisfies an additivity property across adjacent windows (W(t, r)c(zt, t, r) = Φ(r, s)W(t, s)c(zt, t, s) + W(s, r)c(zs, s, r)), ensuring that the finite-horizon steering action remains consistent under temporal subdivision. This dynamic coherence means that the system can be safely decomposed into multiple time intervals without losing consistency with the underlying dynamics.

## Experimental Results
The paper demonstrates the method on two case studies with concrete quantitative results:

1. **2D Planar Swarm Navigation**: The swarm evolves from initial distribution "AYKJ" (authors' initials) to target distribution "DCJK" (authors' surnames) using 16 steps. In the drift-free case (A=0, B=I), the method achieves a trajectory that closely follows the direct path between endpoints, with the induced correspondence largely aligned with the horizontal direction. When introducing rotational drift (A = [[0, -ω], [ω, 0]]), the method successfully adapts the path to follow the rotational dynamics, with the induced correspondence shifting from horizontal to vertical.

2. **3D Spatial Swarm Maneuvering**: The swarm moves from an initial pyramid to a target torus in 3D space using 16 steps. All examples used the same number of steps (16), but different dynamics:
   - Drift-free (A=0, B=I): The swarm deformation was direct, with minimal intermediate complexity
   - Planar rotation (x-y plane): The intermediate paths followed the expected rotational trajectory
   - Full 3D rotation: The intermediate paths took a fundamentally different trajectory, demonstrating how underlying dynamics reshape the feasible paths

The paper doesn't report specific quantitative metrics like error rates or comparison to baselines in the provided text, but it shows through visualizations that the method consistently maintains the specified dynamics while achieving the desired swarm configuration.

## Related Work
The paper positions itself within the growing intersection of flow matching and control theory, building on recent work in MeanFlow [24] which addressed the mismatch between learning instantaneous velocity fields and executing over finite time windows. The authors acknowledge that most flow-based generative models treat the learned object as instantaneous, which becomes problematic under few-step execution. They extend MeanFlow's window-level learning to the control space, where the natural window-level object is the minimum-energy control coefficient rather than an averaged velocity field. The paper also relates to traditional control theory for linear systems, particularly the use of controllability Gramians to solve finite-horizon steering problems, but applies this in a learning context for large-scale swarm control.

## Limitations
The paper primarily focuses on linear time-invariant systems, limiting its direct applicability to more complex nonlinear or time-varying dynamics common in real-world swarm systems. While the experimental results demonstrate effectiveness for 2D and 3D navigation tasks, the paper doesn't evaluate scalability beyond 10,000 agents (though it claims "large-scale" capability). The authors acknowledge that the method assumes controllability (W(t, r) nonsingular), which may not hold for all practical systems. The paper also doesn't address how to handle noisy or partially observed swarm states, a critical consideration for real-world deployments.

## Appendix: Worked Example
Let's walk through a concrete example of the method's operation with specific values. Consider a small swarm of 5 agents moving in 2D space (d=2) with A=0 (drift-free) and B=I. The initial state at t=0 is z0 = [1, 2] (a single agent position), and the target state at t=1 is z1 = [4, 5].

1. **Bridge Construction**: The paper uses a minimum-energy bridge between z0 and z1:
   zτ = Φ(τ, 0)z0 + W(0, τ)Φ(1, τ)⊤W(0, 1)−1(z1 − Φ(1, 0)z0)
   Since A=0, Φ(τ, 0)=I, and W(0, τ)=τI. This simplifies to:
   zτ = (1-τ)z0 + τz1 = [1+3τ, 2+3τ]

2. **Interval Selection**: Choose a time window [t=0.2, r=0.7]. The bridge states are:
   zt = z0.2 = [1.6, 2.6]
   zr = z0.7 = [3.1, 3.6]

3. **Coefficient Calculation**: For A=0, B=I, the coefficient is simply:
   c(zt, t, r) = (zr - zt)/(r-t) = ([1.5, 1.0]/0.5) = [3, 2]

4. **Control Signal**: The control signal over [0.2, 0.7] is:
   u(τ) = B⊤Φ(0.7, τ)⊤c = I * I * [3, 2] = [3, 2] (constant over interval)

5. **State Propagation**: The state update is:
   zr = Φ(0.7, 0.2)zt + W(0.2, 0.7)c = I*[1.6, 2.6] + (0.5)I*[3, 2] = [3.1, 3.6] (matches target)

This demonstrates how the coefficient directly determines the control signal and state progression, with no need for iterative calculations during deployment. The method's power comes from being able to learn this coefficient through the differential identity rather than directly regressing onto it, enabling efficient training.

## References

- Anqi Dong, Yongxin Chen, Karl H. Johansson, Johan Karlsson, "MeanFlow Meets Control: Scaling Sampled-Data Control for Swarms", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.20189

Tags: #multi-agent #swarm-control #control-theory #meanflow #linear-dynamics
