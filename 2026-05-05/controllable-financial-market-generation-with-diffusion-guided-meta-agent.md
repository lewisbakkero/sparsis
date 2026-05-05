---
title: "Controllable Financial Market Generation with Diffusion Guided Meta Agent"
venue: "AAAI"
paper_url: "https://ojs.aaai.org/index.php/AAAI/article/view/37009"
---

## Executive Summary
DigMA introduces a diffusion-guided meta agent for generating controllable financial market simulations at the order level. It addresses the lack of controllability in existing approaches while achieving superior fidelity to real market dynamics. For practitioners building trading systems, this means creating realistic synthetic environments to test strategies under specific market conditions like volatility spikes or sharp drops.

## Why This Matters for Practitioners
If you're developing high-frequency trading systems or risk management tools, DigMA enables you to create controlled synthetic market environments instead of relying solely on historical data. This allows you to systematically test how your trading algorithms respond to extreme scenarios like 2008-style crashes or flash crashes without waiting for them to occur naturally. For instance, you can generate a market with volatility 2.3 times higher than historical averages to stress-test your position sizing algorithms. This capability moves market simulation from reactive (waiting for rare events to happen) to proactive (simulating rare events on demand), significantly improving your ability to build robust trading systems.

## Problem Statement
Current financial market simulation approaches are like trying to recreate a city's traffic flow using only stoplights - they capture basic rules but miss the complex interactions between different types of vehicles, pedestrians, and external factors. Rule-based agents make overly simplistic assumptions about market participants, while learned agents focus on local patterns while neglecting global dynamics. As the paper states: "Their fidelity and flexibility remain limited" and "controllability of the generated market... is absent in the literature." This means practitioners can't systematically explore how trading algorithms behave under specific market conditions like extreme volatility or sharp price drops.

## Proposed Approach
DigMA uses a two-stage architecture where a conditional diffusion model (the meta controller) learns market state dynamics, and a meta agent with financial economic priors generates orders based on those dynamics. The meta controller captures the market state (mid-price return rate and order arrival rate) and conditions the meta agent to generate orders that match target scenarios such as high volatility or sharp drops.

```python
def generate_order_flow(control_targets):
    # Meta controller: Learn market state dynamics
    market_states = meta_controller.denoise(
        control_targets=control_targets,
        diffusion_steps=200
    )
    
    # Order generator: Meta agent creates orders
    order_flow = []
    current_time = 0
    while current_time < trading_hours:
        # Wake up time follows exponential distribution
        wake_interval = random.expovariate(market_states.lambda_t)
        current_time += wake_interval
        
        # Generate order using meta agent
        order = meta_agent.generate(
            market_state=market_states,
            time=current_time
        )
        
        order_flow.append(order)
    
    return order_flow
```

## Key Technical Contributions
DigMA introduces several novel technical mechanisms:

1. **Market-State Diffusion Modelling**: Instead of applying diffusion models directly to raw order-level data (which is challenging due to long, irregular sequences), DigMA models the market state (mid-price return rate and order arrival rate) with a conditional diffusion model. This separates the complex task of order generation into a more tractable problem of modelling aggregate market dynamics.

2. **Meta Agent with Financial Economic Priors**: The meta agent incorporates CARA utility functions and financial economic priors rather than learning agent behaviours from scratch. This makes the agent grounded in financial theory, allowing it to generate realistic order flow without needing to model every possible market participant.

3. **Two-Stage Control Mechanism**: DigMA enables control at the market level (not per order) through a meta controller that conditions on high-level indicators (return, amplitude, volatility). The paper demonstrates both discrete control (using percentile-based bins) and continuous control (using normalized numerical values), showing continuous control achieves lower MSE (0.206 vs 1.055 for return).

4. **Classifier-Free Guidance Implementation**: The authors implement classifier-free guidance with a scaling factor s to control the strength of the conditioning. During sampling, the guided score is computed as `tilde(epsilon) = (1-s) * epsilon_unconditional + s * epsilon_conditional`, enabling precise control without requiring a separate classifier.

## Experimental Results
DigMA was evaluated on two financial datasets (A-Main and ChiNext from the Chinese stock market) with 5,000 samples for validation and 5,000 for testing.

**Controllability**: In Table 1, DigMA with continuous control achieved significantly lower mean squared error (MSE) than the uncontrolled baseline:
- Return: 0.206 (vs 1.443 for "No Control")
- Amplitude: 0.054 (vs 0.521 for "No Control")
- Volatility: 0.011 (vs 0.021 for "No Control")

**Fidelity**: Table 2 demonstrates DigMA's superior fidelity compared to baselines:
- Minutely Log Returns (MinR): 0.084 (vs RFD's 1.198)
- Return Auto-correlation (RetAC): 2.781 (vs RFD's 5.010)
- Volatility Clustering (VolC): 0.273 (vs RFD's 0.839)
- Order Imbalance Ratio (OIR): 0.009 (vs RFD's 0.015)

The paper also validates DigMA as a generative environment for high-frequency trading, with results showing a return of 0.015% (vs 0.000% for RFD baseline).

## Related Work
DigMA builds upon existing financial market simulation approaches (rule-based agents like RFD and learned agents like LOBGAN) while addressing their key limitations. It's the first to integrate diffusion models directly into financial order generation, positioning itself as "among the pioneering models to integrate advanced diffusion-based generative techniques into financial market modelling." Unlike prior work that focuses on predicting next orders from historical flow, DigMA models the market state dynamics to enable control.

## Limitations
The paper acknowledges that DigMA was trained on Chinese stock market data and may not generalise directly to other markets without adaptation. It also notes that the diffusion model requires training on substantial datasets (5,000 samples per dataset), which could be challenging for smaller firms without access to large financial data. The paper doesn't specify the exact computational resources required for training, though it mentions the model uses a U-Net backbone with 1D convolutions.

## Appendix: Worked Example
Let's walk through how DigMA would generate order flow for a high-volatility scenario (target volatility = 0.021):

1. **Market State Generation**: The meta controller uses the conditional diffusion model to generate market states. Starting with random noise, it iteratively denoises over 200 steps to produce:
   - Mid-price return rate: 0.002 (varies throughout the day)
   - Order arrival rate: 0.012 (meaning orders arrive every 83 seconds on average)

2. **Meta Agent Wake-Up**: The meta agent wakes up after an exponential time interval (mean = 1/0.012 ≈ 83 seconds).

3. **Return Estimation**: For the current minute, the meta agent estimates the return as a weighted average:
   - Fundamental component: 0.002 (from the meta controller)
   - Chartist component: 0.001 (historical average return)
   - Noise component: 0.0005 (Gaussian perturbation)
   - Weighted return: (10×0.002 + 1.5×0.001 + 1×0.0005)/12.5 = 0.0018

4. **Holding Optimisation**: The agent computes a future price estimate as `10.00 × exp(0.0018) ≈ 10.018` and derives a demand function based on CARA utility.

5. **Order Sampling**: The agent samples a price uniformly between the lowest acceptable price (9.99) and estimated price (10.018), resulting in a sell order at 9.99 with quantity 8.

6. **Continued Generation**: This process repeats throughout the trading day, generating order flow that matches the target volatility.

See Appendix for the specific numbers and mechanisms behind this order generation process.

## References

- Yu-Hao Huang, Chang Xu, Yang Liu, Weiqing Liu, Wu-Jun Li, Jiang Bian, "Controllable Financial Market Generation with Diffusion Guided Meta Agent", AAAI, 2026-03-23.
- Paper URL: https://ojs.aaai.org/index.php/AAAI/article/view/37009

Tags: #finance #generative-modelling #diffusion-models #meta-agent #controllable-generation
