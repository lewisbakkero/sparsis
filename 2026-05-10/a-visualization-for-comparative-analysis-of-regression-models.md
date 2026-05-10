---
title: "A Visualization for Comparative Analysis of Regression Models"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19291"
---

## Executive Summary
The paper introduces a visualization methodology that reveals critical error patterns in regression models that standard metrics like MAE and RMSE obscure. By using a two-step approach, first with 1D visualizations to identify underperforming models, then a 2D Error Space for direct comparison, it helps engineers make better-informed decisions about model selection, especially in domains where error directionality and distribution matter more than aggregate scores.

## Why This Matters for Practitioners
If you're deploying regression models in safety-critical applications like predictive maintenance for industrial machinery, standard metrics alone can lead to dangerous deployment choices. In the AI4I 2020 case study, Model E1 had slightly better MAE (20.49 vs. 25.58) and RMSE (32.85 vs. 33.55), but the 2D Error Space revealed E1 systematically underestimates to avoid overestimating, critical because overestimating RUL risks unexpected failures. As an engineer, when comparing models with similar MAE/RMSE, always use this visualization to check error distribution patterns rather than relying on aggregate scores alone. For your next model selection, implement this two-step visualization approach to identify whether a model's errors are consistently biased (e.g., underestimating risks) or clustered in specific error ranges.

## Problem Statement
Imagine you're a medical AI engineer selecting between two cancer diagnosis models. Both have similar accuracy scores (85% vs. 84%), but one model systematically overestimates tumor size in 30% of cases while the other underestimates in 25%. Standard metrics can't reveal this critical distribution difference, like looking at a weather report that says "average temperature is 22°C" without showing whether temperatures were consistently mild or had extreme swings. The paper shows this problem is pervasive in regression model evaluation: aggregate metrics conceal whether errors are clustered, directional, or contain dangerous outliers.

## Proposed Approach
The authors propose a two-step visualization methodology:
1. Use 1D visualizations (boxplots) to quickly identify underperforming models based on error distribution
2. For promising models, project their errors into a 2D Error Space to compare directly

This creates a "comparison zone" where you can see which model performs better on specific data points, with regions colored to indicate where one model is better than the other.

```python
def visualize_2d_error_space(model1_errors, model2_errors):
    """
    Visualize the error distribution of two models in 2D space.
    
    Args:
        model1_errors: List of errors for model 1 (predicted - actual)
        model2_errors: List of errors for model 2 (predicted - actual)
    
    Returns:
        Plot with:
        - x-axis: model1_errors
        - y-axis: model2_errors
        - Diagonals representing equal absolute error
        - Colour map showing proximity to median
        - Mahalanobis distance for robust outlier detection
    """
    # Create 2D error space
    x = model1_errors
    y = model2_errors
    
    # Calculate median for colour mapping
    median_x = np.median(x)
    median_y = np.median(y)
    
    # Calculate Mahalanobis distance (simplified for illustration)
    cov_matrix = np.cov(x, y)
    dist = mahalanobis_distance(x, y, cov_matrix)
    
    # Create colormap based on Mahalanobis distance
    colours = plt.cm.viridis(dist)
    
    # Plot with colour mapping
    plt.scatter(x, y, c=colours, alpha=0.6)
    
    # Add comparison zones
    plt.plot([-100, 100], [-100, 100], 'k--', label='Equal error')
    plt.plot([-100, 100], [100, -100], 'k--', label='Symmetric error')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Mahalanobis distance from median')
    
    plt.xlabel(f"{model1} errors")
    plt.ylabel(f"{model2} errors")
    plt.title("2D Error Space Visualization")
    plt.show()
```

## Key Technical Contributions
The paper introduces three novel visualization components that transform how we compare regression models:

1. **The 2D Error Space with comparison zones**: Rather than plotting predictions against ground truth, this plots model1 error against model2 error. The diagonal y = x represents equal error, while the regions above and below represent where each model performs better. The authors use a colour map based on Mahalanobis distance to show proximity to the median, revealing whether errors are clustered or scattered.

2. **Mahalanobis distance for error comparison**: Unlike Euclidean distance, Mahalanobis accounts for correlations between error axes. In the case study, this revealed that Model E1's errors were consistently biased in one direction (underestimating), while Model E2's errors were more scattered but with a higher risk of overestimation. The paper demonstrates this with a visual comparison showing how Euclidean distance creates a circular pattern while Mahalanobis reveals an elongated ellipse that matches the actual error distribution.

3. **Percentile-based error distribution visualization**: The visualization uses a colormap to show the percentile-based distribution of errors, making it easy to spot dense regions (where most errors cluster) and outliers (where errors deviate from the norm). This helps engineers quickly identify where models fail most often, rather than relying on aggregate metrics that mask these patterns.

## Experimental Results
The paper demonstrates the visualization method on the AI4I 2020 Predictive Maintenance dataset (10,000 synthetic observations of industrial machinery), comparing two neural networks with identical architecture but different loss function configurations:

- Model E1 (a=0.2, low asymmetry, penalizes overestimation more) had MAE=20.49, RMSE=32.85, R²=0.14
- Model E2 (a=0.8, higher asymmetry, more balanced penalty) had MAE=25.58, RMSE=33.55, R²=0.10

While standard metrics suggested Model E1 performed better, the 2D Error Space visualization revealed Model E1 systematically underestimates RUL (to avoid dangerous overestimations), while Model E2's errors were more scattered with a higher risk of overestimation. The visualization clearly showed the error distribution along the diagonal, confirming the training setup's intended effect.

## Related Work
The paper positions itself within the broader context of regression model evaluation techniques, acknowledging that while standard metrics like MAE and RMSE remain prevalent in the literature [3], visualization techniques have evolved to enhance understanding of error distributions. It notes that traditional techniques like scatter plots and residual plots become difficult to interpret with larger datasets due to point overlap, while hexbin plots and density plots fail to capture how points are structured relative to a central reference. The paper builds on these approaches but introduces a novel comparison framework focused on directly comparing two models' error distributions.

## Limitations
The paper acknowledges that the visualization method has not been tested on extremely large datasets (beyond 10,000 observations), which could introduce performance bottlenecks in visualization rendering. Additionally, the method is currently designed for pairwise comparisons between models, and extending it to compare three or more models would require additional visualization techniques. The authors also note that the method doesn't directly address how to select hyperparameters for the visualization itself (e.g., the optimal colour mapping intensity).

## Appendix: Worked Example
Let's walk through a concrete example of the 2D Error Space visualization using the AI4I dataset:

Start with two neural networks (E1 and E2) trained to predict Remaining Useful Life (RUL) for industrial machinery components. Both models share the same architecture (two hidden layers with 128 and 64 neurons, ReLU activations, dropout p=0.2) and preprocessing steps, but differ in their loss function's asymmetry parameter (E1: a=0.2, E2: a=0.8).

For a sample of 1000 data points from the dataset:
- Model E1 errors (predicted RUL - actual RUL) range from -15 to +5 (median: -3)
- Model E2 errors range from -10 to +20 (median: +2)

The 2D Error Space visualization plots each data point as (E1 error, E2 error). For example:
- Data point 1: E1 error = -5, E2 error = -1 → (x=-5, y=-1)
- Data point 2: E1 error = +2, E2 error = +15 → (x=2, y=15)

The visualization reveals that most points lie above the line y = x (indicating E2 has larger errors than E1 on those points), and the distribution forms an elongated cloud along the diagonal, showing strong correlation in where the models make errors. The Mahalanobis distance-based colour map shows that the majority of points cluster near the median (-3, +2), with cooler colours (blue) for points further from this median, indicating outliers.

This visualization confirms the training setup: Model E1 systematically underestimates (E1 errors are consistently negative) to avoid dangerous overestimations (which would risk unexpected failures), while Model E2's errors are more scattered, with a higher risk of overestimation.

## References

- Nassime Mountasir, Baptiste Lafabregue, Bruno Albert, Nicolas Lachiche, "A Visualization for Comparative Analysis of Regression Models", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19291

Tags: #regression-analysis #model-evaluation #error-visualization #machine-learning #predictive-maintenance
