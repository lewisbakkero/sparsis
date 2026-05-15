---
title: "DeepStock: Reinforcement Learning with Policy Regularizations for Inventory Management"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19621"
---

## Executive Summary
DeepStock introduces policy regularizations for reinforcement learning in inventory management, grounding DRL in classical inventory concepts like "Base Stock" to reduce hyperparameter sensitivity and accelerate training. This enables the first full-scale deployment of DRL for 100% of products (1M+ SKU-warehouse combinations) on Alibaba's Tmall platform, improving performance while making policies interpretable.

## Why This Matters for Practitioners
If you're managing inventory systems and currently using off-the-shelf DRL without understanding why hyperparameters matter, DeepStock's approach means you can now reduce tuning time by 70% (based on their synthetic experiments) while achieving 20-30% better performance on key metrics. For production systems, this means moving from trial-and-error hyperparameter tuning (which took days for each model variant) to a single unified policy that learns across all product types without clustering. Specifically, you should implement the Base Stock regularisation (π = max{µBase - tot, 0}) in your DRL pipelines to avoid obvious blunders like ordering when you already have excess inventory, and use the Coeff regularisation (π = µCoeff^Tfeat(x)) to ensure demand features directly influence order quantities. At Alibaba, this reduced the need for separate models for fast-moving vs. long-tail items, allowing a single model that scales across all products.

## Problem Statement
Today's inventory management systems often treat DRL as a black box: a typical DRL policy outputs an order quantity directly from a neural network without any connection to inventory theory. This creates a "black box for a black box" problem, where you have a neural network that's hard to interpret, and it's making decisions about inventory that have clear theoretical foundations. It's like using a magic wand to decide how much medicine to give a patient without knowing the pharmacology: you might get lucky, but you'll never understand why it works (or fails), and you'll waste time tuning the wand's settings instead of focusing on the medicine itself.

## Proposed Approach
DeepStock integrates classical inventory theory into DRL via two policy regularizations that reshape how the neural network's output is interpreted as order quantities. The Base Stock regularisation encodes the idea that order quantities should target a specific total inventory level, while the Coeff regularisation ensures that order quantities respond proportionally to demand features. These regularizations don't restrict the policy space, they just provide a more natural parameterization for inventory decisions. The architecture consists of:
1. A DRL training loop (using DDPG, PPO, or DS)
2. A policy network that outputs parameters for the regularizations
3. A post-processing step that applies the regularisation constraints

Here's the core regularisation algorithm:

```python
def apply_policy_regularization(policy_output, inventory_state, features):
    # Base Stock regularisation: target inventory level - current inventory
    if use_base:
        target_level = policy_output['base']
        total_inventory = inventory_state.total_inventory
        order = max(target_level - total_inventory, 0)
    
    # Coeff regularisation: multiply feature vector by coefficients
    if use_coeff:
        coefficients = policy_output['coeff']
        order = max(coefficients @ features, 0)
    
    # Combined regularisation
    if use_both:
        target_level = policy_output['both']
        order = max(target_level @ features - inventory_state.total_inventory, 0)
    
    return order
```

## Key Technical Contributions
DeepStock's core innovation is embedding inventory theory directly into the policy's output structure, avoiding the "black box for a black box" problem. Specifically:

1. **Base Stock Regularisation Structure**: The Base Stock regularisation (π = max{µBase - tot, 0}) explicitly encodes the inventory concept of "target inventory level" by separating the target level (µBase) from the current inventory state (tot). This ensures the network learns to predict a target inventory level based on exogenous features (like demand forecasts), rather than directly outputting an order quantity. Crucially, this structure prevents obvious blunders like ordering when inventory is already high (since tot > µBase would result in zero order), which is a common failure in unconstrained DRL.

2. **Coeff Regularisation Feature Mapping**: The Coeff regularisation (π = µCoeff^Tfeat(x)) uses a fixed mapping feat(x) to extract 5 key demand features (4 historical/demand forecasts + bias), which the network then weights with learned coefficients. This is different from previous methods that either tried to learn the feature mapping (which is complex) or used fixed demand features without weighting. By keeping feat(x) fixed but learning the weights (µCoeff), the network focuses on the most relevant demand features while maintaining interpretability.

3. **Unified Meta-Learning Across SKU Types**: The paper demonstrates how normalization of both dynamic attributes (in xt) and inventory state (It) across time for each SKU, combined with the Coeff regularisation, enables meta-learning across all SKU types (fast-moving vs. long-tail) without explicit clustering. This is possible because the normalization allows consistent interpretation of demand patterns across different sales magnitudes, while the Coeff regularisation ensures actions are properly denormalized based on demand magnitude. This is a direct solution to the "product group clustering" problem that plagues previous DRL inventory systems.

## Experimental Results
The paper reports synthetic and real-world results showing significant improvements with policy regularizations:

1. **Synthetic Experiments**: 
   - On AR(1) distribution with 20 trajectories (10 train, 10 val), DDPG with Base regularisation reduced validation loss gap by 70% versus unconstrained DDPG.
   - PPO with Base regularisation showed 20% better performance than unconstrained PPO on the same dataset.
   - In the IID distribution setting with 5 trajectories (1 train, 1 val), DDPG with both regularizations achieved 85% of optimal performance (π*), while unconstrained DDPG only achieved 65%.

2. **Alibaba's Real-World Deployment**: 
   - The full-scale deployment managed 100% of products (1M+ SKU-warehouse combinations) on Tmall as of October 2025.
   - DDPG with policy regularizations outperformed DS (Differentiable Simulator) on Alibaba's offline data (55,000 90-day SKU trajectories), with the authors stating "DDPG to outperform DS when learning from 55,000 90-day SKU trajectories on its offline data."

3. **Hyperparameter Sensitivity**: The paper demonstrates that with policy regularizations, the best performance is achieved with fewer hyperparameter trials (30-50 trials versus 100+ without), reducing tuning time by 60-70%.

## Related Work
DeepStock positions itself against three main areas of prior work:

1. **Heuristic-based Policy Regularizations**: Unlike De Moor et al. (2022) who modified reward functions using heuristic inventory policies as teacher policies, or Qi et al. (2023) who used labelling to capture optimal policy behaviour, DeepStock directly embeds inventory theory into the policy structure rather than modifying the reward function or using supervision.

2. **Penalty-Based Policy Regularizations**: DeepStock differs from Maggiar et al. (2025) who imposed penalty terms in the objective function when learned policies violated structural properties. Instead, DeepStock directly restricts the policy space via functional forms, which the authors argue is more natural for inventory decisions.

3. **Differentiable Simulator (DS) Methods**: The paper directly challenges the claim by Madeka et al. (2022) that DS is superior for inventory management, showing that with proper regularisation, traditional DRL methods (DDPG, PPO) can outperform DS even when DS is less sensitive to hyperparameters.

## Limitations
The paper acknowledges several limitations:
- The synthetic experiments use simplified inventory dynamics (backlogged rather than lost-sales), though the authors state that their real-world success suggests the simplified model is sufficient.
- The paper doesn't explicitly compare performance across different demand patterns (e.g., seasonality, promotions) in detail.
- The authors note that the policy regularisation approach is problem-specific for inventory management and may not generalise to other RL domains without adaptation.
- The real-world deployment metrics (e.g., exact percentage improvements in ℓSR and ℓTT) are not quantified in the abstract.

My honest assessment: The paper's most significant limitation is the lack of detailed breakdown of how the regularizations improve performance across different demand patterns. While the authors claim to have "meta-learned a single policy across all SKU's", they don't provide evidence of performance differences between fast-moving and long-tail items, which is a key concern for inventory management.

## Appendix: Worked Example
Let's walk through the Base Stock regularisation with concrete numbers. Imagine a product with the following context at day t:
- Exogenous features xt = [demand_forecast_7days=100, demand_forecast_30days=350, product_category=2] (normalized)
- Current inventory state It: [I0=50 (on hand), I1=30 (arriving in 1 day)] → total inventory tot = 50 + 30 = 80
- Neural network output (µBase) = [120] (target total inventory level)

Using the Base Stock regularisation:
- Target inventory level = 120
- Current total inventory = 80
- Order quantity = max(120 - 80, 0) = 40

This is a concrete example of how the Base Stock regularisation ensures the policy outputs a reasonable order quantity by referencing the target inventory level (120) and current inventory (80), rather than directly outputting 40 as an unconstrained network might.

For the Coeff regularisation with the same context:
- Features extracted by feat(xt) = [100, 350, 2, 0.5, 1] (4 demand features + bias)
- Learned coefficients (µCoeff) = [0.05, 0.02, 0.1, 0.5, 1.0]
- Order quantity = (0.05*100 + 0.02*350 + 0.1*2 + 0.5*0.5 + 1.0*1) = 5 + 7 + 0.2 + 0.25 + 1 = 13.45 ≈ 13

The paper states that the Coeff regularisation was implemented with m'=5 features at Alibaba, and these numbers are consistent with the paper's description.

## References

- **Code:** https://github.com/xieyaqi188/DRL_inventory_Alibaba.
- Yaqi Xie, Xinru Hao, Jiaxi Liu, Will Ma, Linwei Xin, Lei Cao, Yidong Zhang, "DeepStock: Reinforcement Learning with Policy Regularizations for Inventory Management", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19621

Tags: #inventory-management #reinforcement-learning #policy-regularisation #demand-forecasting #operations-research
