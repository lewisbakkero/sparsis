---
title: "Joint Return and Risk Modeling with Deep Neural Networks for Portfolio Construction"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19288"
---

## Executive Summary
This paper presents a joint return and risk modelling framework using deep neural networks for portfolio construction, replacing the traditional two-step approach of separately estimating expected returns and covariance matrices. The model learns dynamic return and risk structures end-to-end from sequential financial data, achieving superior risk-adjusted performance (Sharpe ratio 0.91) compared to equal-weight and historical mean-variance benchmarks during 2020-2024.

## Why This Matters for Practitioners
If you're building or maintaining investment systems that rely on historical covariance matrices for risk estimation, this paper demonstrates a practical alternative that adapts to changing market regimes without requiring manual re-engineering of risk models. Specifically, you should consider replacing static covariance estimation with a learned risk representation that captures volatility clustering and regime shifts, particularly during periods of market stress. For production systems, this means implementing a multivariate LSTM-based pipeline that generates both return forecasts and risk estimates from the same temporal representation, reducing the need for separate risk modules. When operationalising this approach, focus first on incorporating dynamic risk estimation into your existing portfolio optimisation layer rather than overhauling the entire system.

## Problem Statement
Traditional portfolio construction is like using a map from the 18th century in a modern city: it's based on assumptions that no longer reflect the current landscape. The mean-variance framework requires separately estimating expected returns and covariance matrices from historical statistics, creating a fundamental mismatch when market conditions change rapidly, a bit like trying to navigate London's traffic with a 19th-century street map. This separation leads to unstable allocations during volatile periods, as the return predictions and risk estimates become inconsistent with each other.

## Proposed Approach
The framework integrates multivariate time-series forecasting, dynamic risk estimation, and portfolio optimisation into a single pipeline. A multivariate LSTM network processes historical returns to generate latent representations that simultaneously drive return forecasting and risk estimation. These representations are used to derive dynamic volatility and correlation estimates, which inform the Sharpe ratio-based optimisation module.

```python
def neural_portfolio_strategy(window_length, historical_returns):
    # Extract latent representation from historical returns
    latent = lstm_model(historical_returns)
    
    # Predict expected returns
    expected_returns = linear_projection(latent)
    
    # Derive dynamic risk measures
    volatility = rolling_stddev(expected_returns)
    covariance = cov(expected_returns)
    
    # Solve Sharpe ratio optimisation
    portfolio_weights = sharp_optimisation(expected_returns, covariance)
    
    return portfolio_weights
```

## Key Technical Contributions
The paper's core innovation lies in how it integrates return and risk modelling through a unified representation. Specifically:

1. **Shared Latent Representation Architecture**: The model employs a multivariate LSTM that extracts a single latent representation from historical returns, which is then used both for return prediction and risk estimation. Unlike prior approaches that use separate models for returns and risk, this shared representation ensures consistency between the two components. The authors demonstrate that this consistency is critical for improved risk-adjusted performance during market regime shifts.

2. **Dynamic Risk Estimation from Predicted Returns**: Instead of relying on historical covariance matrices, the framework computes the covariance of predicted returns (Cov(μ̂ₜ)) to derive dynamic risk estimates. This approach ensures that risk measures align with the return predictions rather than being based on potentially outdated historical data. The authors show this captures volatility clustering better than static historical estimators.

3. **Sharpe Ratio Optimisation with Learned Risk**: The optimisation module explicitly uses the learned risk structure to maximise the Sharpe ratio. This differs from traditional approaches that use historical risk measures, allowing the portfolio to adapt to changing market conditions without requiring manual reconfiguration of risk parameters.

## Experimental Results
The framework was evaluated on daily returns from ten large-cap US equities (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, JPM, V, UNH) spanning 2010-2024, with 2020-2024 as the test period. The deep forecasting model achieved an RMSE of 0.0264 and directional accuracy of 51.9%, indicating moderate but economically meaningful predictive power. For portfolio performance (Table 2), the Neural Portfolio strategy achieved an annual return of 36.4% and Sharpe ratio of 0.91, significantly outperforming equal-weight (annual return 23.32%, Sharpe 0.7756) and historical mean-variance (annual return 20.62%, Sharpe 0.7474) benchmarks. The authors note that the improvement in Sharpe ratio indicates superior risk-adjusted performance rather than mere return amplification.

## Related Work
The paper positions itself as extending the shift towards end-to-end portfolio construction frameworks while addressing a gap in joint return and risk modelling. Unlike prior work that focuses primarily on return prediction (e.g., Fischer & Krauss, 2018) or proposes end-to-end allocation strategies (e.g., Zhang et al., 2020), this approach explicitly models the dynamic relationship between return and risk. The authors acknowledge that while some studies (e.g., Uysal et al., 2021) explore risk-aware optimisation, their framework is unique in deriving risk estimates directly from the same learned representation used for return prediction.

## Limitations
The paper acknowledges limitations including: (1) the use of a single rolling window (length L, unspecified in the text) for feature extraction; (2) evaluation on a limited set of ten large-cap US equities; (3) no consideration of transaction costs in the optimisation process. The authors note that future work should incorporate transaction costs, regime-aware training, and multi-horizon forecasting. From an engineering perspective, the lack of ablation studies on the optimal window length for different market regimes represents a significant gap for production deployment.

## Appendix: Worked Example
Consider a 20-day rolling window (L=20) processing daily log returns for five assets (AAPL, MSFT, GOOGL, JPM, META) from 2023-01-01 to 2023-01-20. The LSTM network (with 64 hidden units) processes these returns to generate a 64-dimensional latent representation hₜ. From this representation, the expected return forecasts are computed as μ̂ₜ = Wμhₜ + bμ, yielding predicted daily returns for each asset.

For risk estimation, the volatility for each asset is calculated as σ̂ₜ,i = √[(1/20)∑ᵢ₌ₜ₋₂₀ᵗ⁻¹(rₖ,i - r̄ᵢ)²]. The cross-asset covariance matrix Σ̂ₜ is derived from the predicted returns: Σ̂ₜ = Cov(μ̂ₜ). If the predicted returns for the five assets are [0.0021, 0.0018, 0.0032, 0.0027, 0.0015], the covariance matrix would be calculated from these values rather than historical data. This covariance matrix is then used in the Sharpe ratio optimisation to determine the portfolio weights that maximise return per unit of risk.

See Appendix for detailed numerical example of how this process transforms raw returns into portfolio weights.

## References

- Keonvin Park, "Joint Return and Risk Modeling with Deep Neural Networks for Portfolio Construction", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19288

Tags: #finance #portfolio-optimisation #deep-learning #neural-networks #risk-management
