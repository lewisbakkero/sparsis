---
title: "Generalized Stock Price Prediction for Multiple Stocks Combined with News Fusion"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2603.19286"
---

## Executive Summary
This paper introduces a generalised stock price prediction model that trains a single model across multiple stocks using news fusion. Unlike prior work that required separate models per stock, their approach dynamically filters news relevant to each stock using stock name embeddings within attention mechanisms, achieving a 7.11% reduction in Mean Absolute Error (MAE) compared to baselines.

## Why This Matters for Practitioners
If you're maintaining separate stock prediction models for each asset class in production, this paper demonstrates a path to reduce operational complexity while improving accuracy. Instead of training and deploying 100+ individual models for different stocks, you could implement a single generalised model that processes news dynamically for each target stock. The key implementation insight is using stock name embeddings within attention mechanisms to filter irrelevant news without requiring pre-filtered datasets or separate training for each stock. This approach could save significant engineering effort in model maintenance while delivering measurable accuracy improvements.

## Problem Statement
Current stock prediction systems face a fundamental trade-off: filtering news using keyword searches yields limited data (like short social media posts), while aggregating all news introduces noise that obscures relevant information. Imagine managing a news feed for a large company where you need to identify only the 5% of articles that mention your specific product, but the system can't distinguish between "TechCorp" and "TechInc" without explicit keyword mapping. The result is either missing critical information or drowning in irrelevant noise.

## Proposed Approach
The system processes daily financial news using a pre-trained LLM, filters news relevant to each stock using stock name embeddings within attention mechanisms, and fuses this with historical price data. The core architecture consists of four main modules: News Encoding, Attentive Pooling, News-Price Fusion, and Patch Reprogramming.

```python
def predict_stock_price(week_news, stock_name, historical_prices):
    # News Encoding
    news_embeddings = encode_news_with_plm(week_news)
    
    # Attentive Pooling
    news_representation = attentive_pooling(
        news_embeddings, 
        stock_name_embedding=embed_stock_name(stock_name)
    )
    
    # News-Price Fusion
    fused_features = news_price_fusion(
        news_representation, 
        normalized_prices=normalize(historical_prices)
    )
    
    # Patch Reprogramming
    llm_input = patch_reprogramming(fused_features)
    
    # Predict using Time-LLM
    return time_llm_predict(llm_input)
```

## Key Technical Contributions
The paper makes several significant technical contributions in how it handles news integration and model generalisation.

The core innovation is the dynamic news filtering mechanism that addresses the dual challenges of noise reduction and stock-specific relevance. Unlike previous approaches that either filtered news before the prediction model (using keyword-based queries) or aggregated all news (introducing noise), this method integrates stock name embeddings directly into the attention mechanism for filtering.

1. **Stock-Conditioned Attention Mechanisms**: The paper introduces three variants that incorporate stock name embeddings into the pooling process:
   - **Cross-attentive Pooling (CAP)**: Uses the stock name embedding as the query to modulate attention weights over news articles (eq. 2: $ct = Softmax(eW_cB_t^T)B_t$)
   - **Self-attentive Pooling (SAP)**: Appends the stock name embedding to the beginning of the news sequence and applies self-attention (eq. 3: $ct = Softmax(w_s\tilde{B}_t^T)\tilde{B}_t$)
   - **Position-aware Self-attentive Pooling (PA-SAP)**: Incorporates positional embeddings to account for temporal order within news sequences while adding the stock name embedding (eq. 4: $ct = Softmax(w_p\bar{B}_t^T)\bar{B}_t$)

2. **Integrated News-Price Fusion**: The paper moves beyond simple concatenation by using bidirectional cross-attention and GCN to model relationships between news and price data. The News-Price Fusion module combines price-to-news and news-to-price cross-attention with GCN to model both within-day relations and temporal dynamics (eq. 7: $H = GCNConv(N,P), S_{GCN} = CausalCNN(H)$).

3. **Generalised Model Training**: Unlike previous approaches that trained separate models for each stock or market, they train a single model on aggregated data from multiple stocks (6 Taiwan stocks, 42 U.S. stocks), enabling cross-market applicability. This requires the news filtering mechanism to operate dynamically for each stock during inference.

See Appendix for a step-by-step worked example with concrete numbers.

## Experimental Results
On the Taiwan Stock Exchange dataset (6 stocks: TSMC, MediaTek, Foxconn, Realtek, Novatek, Delta), the proposed method achieved a 7.11% reduction in MAE compared to the TimeLLM baseline (without news). Across all six stocks, the best-performing method (+SAP) achieved the lowest MAE (0.1200 for MediaTek, 0.1398 for TSMC) compared to the baseline TimeLLM (0.1496 for TSMC, 0.1365 for MediaTek).

For MediaTek, the results show:
- LSTM MAE: 0.1403
- FPT MAE: 0.1365
- TimeLLM MAE: 0.1365
- +News MAE: 0.1347
- +SAP MAE: 0.1294
- +CAP MAE: 0.1276
- +PA-SAP MAE: 0.1200

The paper also validated cross-market applicability on the BigData23 U.S. dataset (42 stocks from multiple sectors), though specific MAE improvements aren't detailed in the provided text.

## Related Work
The paper positions itself within the growing field of text-enhanced time series forecasting. It builds on Time-LLM for its foundation and improves upon the BELT framework [10] and BigData22 baselines [8] by introducing dynamic news filtering through stock name embeddings. Unlike prior work that relied on extracting news for individual stocks through information retrieval (which limits data quantity) or aggregating all news (introducing noise), this approach dynamically filters news relevance during the prediction process. The paper specifically differentiates itself from approaches that filter news before feeding it into the prediction model (like [4]) by integrating the filtering process directly into the attention mechanism.

## Limitations
The paper acknowledges limitations in its approach: the model's performance might vary across different market conditions and sectors. They did not investigate how the model performs during extreme market events like the 2020 pandemic or major geopolitical events. The paper also doesn't specify how the model would handle news articles in multiple languages, despite mentioning Chinese articles for Taiwan data and English for BigData23. The authors note that their approach depends on the quality of the underlying pre-trained language model, but they don't explore how different PLMs might affect performance.

## Appendix: Worked Example
Let's walk through a single day's processing for TSMC stock with news articles and closing prices.

1. **News Collection**: On a given day, the system collects 219 news articles (average size, per Table 1), each containing approximately 509.96 characters.

2. **News Encoding**: Using BERT, each article is encoded into a 768-dimensional embedding (standard for BERT-base). For the sake of example, we'll consider 3 articles:
   - Article 1: "TSMC announces new chip technology" → [vector A: 768 dimensions]
   - Article 2: "TechCorp reports earnings" → [vector B: 768 dimensions]
   - Article 3: "TSMC partnership with Apple" → [vector C: 768 dimensions]

3. **Stock Name Embedding**: The stock name "TSMC" is encoded using BERT (character-level embedding), resulting in a 768-dimensional vector [vector T].

4. **Attentive Pooling (CAP)**: For Cross-attentive Pooling, the stock name embedding (vector T) serves as the query. The attention weights are calculated based on the similarity between vector T and each article embedding:
   - Weight for Article 1: 0.7 (high similarity)
   - Weight for Article 2: 0.1 (low similarity)
   - Weight for Article 3: 0.8 (high similarity)
   
   The pooled representation = 0.7*vector A + 0.1*vector B + 0.8*vector C = [vector P: 768 dimensions]

5. **Price Data**: The historical price data for the previous 20 trading days (normalized using Standard Scaling) is processed into a 20x768-dimensional matrix.

6. **News-Price Fusion**: The pooled news embedding (vector P) is fused with the price sequence using bidirectional cross-attention:
   - Price-to-news attention: Price sequence serves as query, news embedding as key/value → [matrix X]
   - News-to-price attention: News embedding serves as query, price sequence as key/value → [matrix Y]
   - GCN layer combines these with the original price and news embeddings → [fused features]

7. **Patch Reprogramming**: The fused features are mapped into the LLM embedding space using the patch reprogramming technique, resulting in a sequence compatible with the frozen LLaMA3-8b backbone.

8. **Prediction**: The LLM predicts the next day's closing price based on this fused representation.

## References

- **Code:** https://github.com/thuml/Time-Series-Library
- Pei-Jun Liao, Hung-Shin Lee, Yao-Fei Cheng, Li-Wei Chen, Hung-yi Lee, Hsin-Min Wang, "Generalized Stock Price Prediction for Multiple Stocks Combined with News Fusion", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19286

Tags: #finance #time-series-forecasting #news-fusion #attention-mechanisms #multi-stock-modelling
