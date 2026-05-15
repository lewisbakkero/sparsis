---
title: "Physics-Informed Neural Network with Adaptive Clustering Learning Mechanism for Information Popularity Prediction"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19599"
---

## Executive Summary
PIACN (Physics-Informed Adaptive Clustering Network) predicts information popularity by modelling macroscopic physical patterns using the Richards function and handling information heterogeneity through adaptive clustering. It outperforms state-of-the-art approaches by over 12% on real-world social media datasets, offering engineers a more accurate method for forecasting content virality.

## Why This Matters for Practitioners
If you're building recommendation systems for social media platforms or content monitoring tools, this paper shows that incorporating physics-informed constraints and content-specific clustering can significantly improve prediction accuracy. Instead of relying solely on deep learning models that capture micro-level features, you should consider adding a physics-informed loss based on the Richards function (which models the characteristic S-shaped spread of information) and implement adaptive clustering to differentiate between content types. For example, when designing your next recommendation system, integrate an adaptive clustering layer to distinguish between political, entertainment, and scientific content, and embed the Richards function into your loss function to ensure predictions align with observed macroscopic patterns of information spread.

## Problem Statement
Current popularity prediction models resemble a myopic bot that studies individual leaves but misses the shape of the entire tree. They focus exclusively on micro-level features (like individual user interactions) while neglecting the macroscopic patterns of information spread, which follow a characteristic S-shaped curve where popularity grows rapidly, then plateaus. This is especially problematic because different content types (political, entertainment, scientific) spread at different rates and reach different maximum popularity levels. For instance, political news may spread rapidly but plateau early, while scientific content may have a slower but more sustained growth pattern.

## Proposed Approach
PIACN combines five modules: cascade embedding (learns micro-level dynamics), temporal learning (captures macroscopic evolution), adaptive clustering (handles content heterogeneity), prediction network (makes final output), and physical modelling (approximates Richards function parameters). The physical modelling network guides the training by embedding physical constraints into the loss function.

Here's a simplified overview of PIACN's core workflow:
1. Input: Information cascade features and popularity time series
2. Cascade embedding: Converts micro-level features into latent representations
3. Temporal learning: Processes popularity time series to capture long-term patterns
4. Adaptive clustering: Groups content by type without labels
5. Physical modelling: Adjusts Richards function parameters to match observed patterns
6. Output: Predicted popularity increment

```python
def PIACN(cascade_features, popularity_series):
    # Cascade embedding network
    micro_representations = cascade_embedding_network(cascade_features)
    
    # Temporal learning network
    macro_representations = temporal_learning_network(popularity_series, micro_representations)
    
    # Adaptive clustering network
    cluster_centers = adaptive_clustering_network(macro_representations)
    
    # Physical modelling network
    richards_params = physical_modeling_network(micro_representations, cluster_centers)
    
    # Prediction network
    prediction = prediction_network(micro_representations, macro_representations, cluster_centers)
    
    return prediction
```

## Key Technical Contributions
PIACN introduces two key innovations that address fundamental gaps in current approaches:

1. **Physics-informed constraint using the Richards function**: Rather than treating information spread as a black box, PIACN explicitly models the macroscopic physical pattern using the Richards function, which mathematically captures the S-shaped curve of information spread. The model learns the parameters (α, β, γ, δ) of the Richards function through a dedicated neural network that approximates these parameters. During training, it embeds the Richards function into the loss function as a constraint, ensuring predictions align with the observed macroscopic pattern. This isn't just a statistical fit, it's a physical constraint that guides the model towards learning patterns consistent with how information naturally spreads.

2. **Adaptive clustering for information heterogeneity**: Unlike previous models that treat all content as homogeneous, PIACN implements an unsupervised adaptive clustering mechanism that identifies natural groupings of content based on propagation patterns. It uses the Student's t-distribution as a kernel to measure similarity between data representations and cluster centres, with the cluster centres being learned during backpropagation. This approach surpasses traditional clustering algorithms like K-means because it can be integrated into an end-to-end deep learning framework. The clustering centres are then used as distinctive features to differentiate between content types, allowing the model to make predictions that account for the specific propagation patterns of different content categories.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
PIACN was evaluated on three real-world datasets (Weibo, Twitter, and APS), with results showing significant improvements over state-of-the-art baselines. The authors report more than 12% improvement in prediction accuracy compared to existing methods. On all three datasets, the Richards function fit achieved R-squared values of 0.99, confirming that information spread follows the predicted macroscopic pattern. While the paper mentions using metrics like MAE (Mean Absolute Error) and RMSE (Root Mean Square Error), specific numerical values aren't provided in the excerpt. The improvements are described as "significant" but without specific statistical tests, which is a limitation noted in the paper's experimental section.

## Related Work
PIACN positions itself at the intersection of physics-informed neural networks (PINNs) and information popularity prediction. While previous work in popularity prediction has focused on mathematical modelling, feature engineering, and deep learning approaches, PIACN differs by incorporating physics-informed constraints (a concept previously used in fields like epidemiology and fluid mechanics) and adaptive clustering for content heterogeneity. The paper builds on recent deep learning models like DeepCas, CasCN, and TEDDY, which use graph convolutional networks and temporal modelling, but extends these by adding physics-informed constraints and clustering mechanisms. This represents a significant step beyond the "micro-focused" approaches of prior work.

## Limitations
The paper acknowledges two key limitations: 
1. The adaptive clustering mechanism is unsupervised, meaning it may not perfectly capture all nuances of content types, especially if there are subtle variations within content categories.
2. The paper doesn't explicitly discuss scalability for extremely large-scale platforms or real-time prediction requirements, though it mentions the model's efficiency in the temporal learning network (using multi-layer dilated convolutions).

Additionally, while the paper shows that different content types have different propagation patterns, it doesn't specify how many content categories are typically encountered in practical applications. The adaptive clustering mechanism might require retraining as new content categories emerge, which could impact real-world deployment.

## Appendix: Worked Example
Let's walk through how PIACN processes a political news article on Weibo over 24 hours with popularity recorded at 3-hour intervals (8 data points total).

Step 1: Cascade embedding network processes micro-level features (user features and propagation speed) using self-attention. For this political news article, it outputs a 128-dimensional latent representation [0.2, -0.1, 0.4, ...] capturing how quickly it's spreading (rapid early retweets) and which users are sharing it (influential political accounts).

Step 2: The temporal learning network processes the popularity time series using multi-layer dilated temporal convolution with increasing dilation factors (1, 2, 4, 8). It captures both short-term (3-hour) and long-term (24-hour) patterns, producing a 64-dimensional macro representation [0.3, -0.2, 0.5, ...].

Step 3: The adaptive clustering network identifies the article as political content (cluster 3), with the cluster centre vector [0.2, -0.1, 0.4, ...] (128 dimensions) representing the typical propagation pattern for political content.

Step 4: The physical modelling network adjusts the Richards function parameters based on the micro and cluster representations. For political content, it learns α=0.85 (upper popularity limit), β=0.3 (growth rate), γ=1.2 (inflection point), and δ=0.7 (steepness).

Step 5: The prediction network combines the micro representation, macro representation, and cluster centre to predict the popularity increment over the next 24 hours. The model predicts a 35.2% increase in popularity, compared to the current popularity level.

This example demonstrates how PIACN integrates macroscopic physical patterns (via Richards function) and content heterogeneity (via adaptive clustering) to make more accurate predictions of information popularity.

## References

- Guangyin Jin, Xiaohan Ni, Yanjie Song, Kun Wei, Jie Zhao, Leiming Jia, Witold Pedrycz, "Physics-Informed Neural Network with Adaptive Clustering Learning Mechanism for Information Popularity Prediction", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19599

Tags: #information-retrieval #social-networks #physics-informed #adaptive-clustering #richards-function
