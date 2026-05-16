---
title: "Auditing Google's AI Overviews and Featured Snippets: A Case Study on Baby Care and Pregnancy"
venue: "arXiv cs.AI"
paper_url: "https://arxiv.org/abs/2511.12920"
---

## Executive Summary
The authors conducted a systematic audit of Google's AI Overviews (AIO) and Featured Snippets (FS) for baby care and pregnancy queries, analysing 1,508 real user queries. They identified critical quality issues including inconsistent information between AIO and FS (33% of cases), a critically low presence of medical safeguards (11% in AIO, 7% in FS), and a dominance of commercial sources for FS. For production engineers, this research demonstrates that AI-generated search features in health domains require mandatory medical safety checks and consistency validation before deployment.

## Why This Matters for Practitioners
If you're building health-related search features, you must implement mandatory medical safeguard checks before rendering any AI-generated content (as Google's system lacks this). Your system should also track and resolve inconsistencies between different response formats, like how AIO and FS presented inconsistent information in 33% of cases. Additionally, you should explicitly audit source credibility, as FS more often linked to commercial sources (7% business links in FS vs 12% in AIO), potentially biasing user trust. For production systems, these findings mean: 1) Implement medical safety checks verifying content against clinical guidelines before rendering, 2) Create consistency validation between different response formats (e.g., AIO vs. FS), and 3) Diversify source categories to avoid over-reliance on commercial domains.

## Problem Statement
Imagine you're a new parent searching for "can I take paracetamol during pregnancy?" and Google displays two conflicting responses: one AI Overview stating "paracetamol is safe for pain relief in pregnancy" and a Featured Snippet claiming "paracetamol use in pregnancy is associated with increased risk of developmental issues." This isn't hypothetical, it happens in 33% of baby care and pregnancy searches. The current landscape resembles a medical clinic where different staff members give opposing advice for the same question, with no way to verify accuracy, potentially putting parents and babies at real risk.

## Proposed Approach
The authors built a comprehensive framework to audit AI-powered search features in high-stakes domains. Their approach involved:
1. Collecting 1,508 well-formed queries related to baby care and pregnancy
2. Categorising queries by question type (binary, wh*, how-to, etc.) and sentiment
3. Systematically crawling Google SERPs from a fixed location
4. Developing a multi-dimensional quality evaluation framework assessing consistency, relevance, medical safeguards, source categories, and sentiment alignment

Here's the pseudocode for their query classification and audit pipeline:

```python
def audit_google_search_queries():
    # Step 1: Query collection and filtering
    raw_queries = fetch_queries_from_public_dataset(keyword=["baby", "pregnancy"])
    filtered_queries = remove_duplicates_and_short_queries(raw_queries)
    human_related_queries = llm_filter(filtered_queries, model="gpt-4o-mini")
    
    # Step 2: Query categorisation
    categorized_queries = {
        "binary": [],
        "wh*": [],
        "when": [],
        "how_to": [],
        "how_adj_adv": [],
        "why": []
    }
    for query in human_related_queries:
        category = detect_question_type(query)
        categorized_queries[category].append(query)
    
    # Step 3: Sentiment variation for binary queries
    sentiment_queries = []
    for binary_query in categorized_queries["binary"]:
        for sentiment in ["neutral", "positive", "negative"]:
            sentiment_query = modify_query_for_sentiment(binary_query, sentiment)
            sentiment_queries.append(sentiment_query)
    
    # Step 4: Audit pipeline
    for query in sentiment_queries[:1508]:  # Limit to 1,508
        serp = crawl_google_search(query, location="new_york")
        aios = extract_ai_overviews(serp)
        fs = extract_featured_snippets(serp)
        analyze_consistency(aios, fs)
        analyze_safeguards(aios, fs)
        analyze_source_categories(aios, fs)
```

## Key Technical Contributions
The paper makes several key technical contributions to AI search system auditing:

1. They established a multi-dimensional quality framework for evaluating AI search features in high-stakes domains, moving beyond simple relevance metrics to include medical safeguards, consistency between components, and source credibility. This framework includes five core metrics: consistency (between AIO and FS), relevance, medical safeguards, source categories, and sentiment valence, each tailored to health information quality.

2. The authors developed a robust method for handling "suppressed" AIOs that exist in the HTML but aren't visible to users, which is a critical technical detail for accurate audit results. They identified these by checking for presence of primary identifier elements and container elements without styling attributes that would make them visible.

3. They created a taxonomy of contradictions for the health domain, moving beyond strict logical definitions to a more human-intuitive approach. This taxonomy includes binary contradictions (direct opposites like "safe" vs "unsafe") and numeric mismatches (different but specific ranges or measurements), which proved crucial for identifying inconsistencies in health information.

4. The study implemented a systematic method for evaluating query sentiment alignment with response sentiment, controlling for topic and question type, to detect confirmation bias in AI-generated responses. This required creating sentiment variations of binary queries and conducting manual annotation of response sentiment values.

## Experimental Results
The study analysed 1,508 real baby care and pregnancy-related queries across six question types and three sentiment variations:

- AIO appeared more frequently than FS (84% vs 33% overall), with both co-occurring in 22% of cases
- Inconsistent information between AIO and FS was found in 33% of co-occurring cases (41% for highlighted content)
- Relevance was high: AIO was relevant 97% of the time, FS was relevant 89% of the time
- Medical safeguards were critically low: AIO included safeguards in 11% of responses, FS in 7%
- Source categories: Both AIO and FS cited more health-related sources than regular results, but FS more often linked to commercial sources (12% business links in AIO vs 7% in FS)
- No evidence of confirmation bias was found when controlling for topic and question type

The authors report that these results were significant (p < 0.05 for consistency with question type in relevant analysis), though they don't specify the exact statistical tests used.

## Related Work
This work builds on prior research in featured snippet auditing but extends it to AI-generated content (AIO), which has been under-audited. Previous work on featured snippets (Strzelecki and Rutecka 2019, 2020; Scull 2020) focused on their appearance and source domains, but this paper is the first to systematically compare AIO and FS for consistency and safety. They also extend prior work on generative search engine auditing (Venkit et al. 2024; Li and Sinnamon 2024) by focusing specifically on high-stakes health information where quality directly impacts user well-being, and by controlling for topic and question type when investigating confirmation bias.

## Limitations
The study focused on only one domain (baby care and pregnancy), which might not generalise to all health-related queries. They also didn't evaluate the impact of these inconsistencies on user behaviour or health outcomes. The manual annotation process, while rigorous, might have introduced some subjectivity in evaluating medical safeguards. The authors note that their methodology could be extended to other high-stakes domains like legal or political information, but they didn't explore this. The paper also doesn't detail how Google's algorithm changes over time might affect these results.

## Appendix: Worked Example
Let's walk through the consistency analysis for a specific query to illustrate how the authors identified inconsistencies in health information:

Consider the query: "Is it safe for pregnant women to eat sushi?" (a binary question).

- AIO response: "Moderate sushi consumption (up to 2-3 servings per week) is generally safe during pregnancy according to the American College of Obstetricians and Gynecologists."
- FS response: "Pregnant women should avoid all sushi due to mercury risks and potential bacterial contamination."

These two responses represent a binary contradiction (moderate consumption safe vs. avoid entirely), making them inconsistent. The authors classified this as a "binary contradiction" in their taxonomy.

The analysis process for this query would involve:
1. Identifying both responses in the SERP (AIO above organic results, FS in the featured snippet section)
2. Parsing the text of both responses (AIO: 2-3 servings per week safe; FS: avoid all sushi)
3. Checking for countervailing evidence (yes - one says safe with limits, one says avoid entirely)
4. Classifying the contradiction as binary (direct opposition between "safe" and "avoid")
5. Recording this as an inconsistent case

This specific example illustrates the 33% inconsistency rate found across all queries. Note that for this query, the AIO provided a specific medical guideline (2-3 servings per week) while the FS provided a blanket statement, highlighting the lack of medical safeguards in the FS response.

## References

- Desheng Hu, Joachim Baumann, Aleksandra Urman, Elsa Lichtenegger, Robin Forsberg, Aniko Hannak, Christo Wilson, "Auditing Google's AI Overviews and Featured Snippets: A Case Study on Baby Care and Pregnancy", arXiv cs.AI, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2511.12920

Tags: #health-information-systems #search-engine-audit #medical-safeguards #content-consistency #source-categorisation
