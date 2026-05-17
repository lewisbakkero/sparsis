---
title: "Continual Learning for Food Category Classification Dataset: Enhancing Model Adaptability and Performance"
venue: "arXiv cs.LG"
paper_url: "https://arxiv.org/abs/2603.19624"
---

## Executive Summary
The authors developed a text-guided continual learning framework for food category classification that incrementally updates without catastrophic forgetting. Unlike traditional models requiring full retraining when new foods are introduced, their approach processes dish names and ingredients through TF-IDF vectors to maintain high accuracy (98.38%) while adapting to new categories like "dosa" or "kimchi" with minimal computational overhead. This is critical for production systems in dietary monitoring and personalised nutrition where food varieties constantly evolve.

## Why This Matters for Practitioners
If you're building a food recognition system for dietary tracking platforms, this work shows you can avoid full retraining cycles when adding new food categories (e.g., regional dishes), reducing operational overhead by up to 90% compared to standard retraining approaches. Specifically, implement a TF-IDF-based text feature pipeline that processes dish names and ingredients, then use the incremental update protocol described in the paper (1-10 epochs with batch size 32). For production deployment, this means you can add new food categories with minimal human intervention while maintaining 98.38% accuracy. The L2 regularisation (0.01) and early stopping (patience=5) in their architecture prevent catastrophic forgetting, critical for maintaining existing classification capabilities when adding new categories.

## Problem Statement
Traditional food recognition systems are like a library with a fixed collection: if a new book (food item) appears that wasn't in the original catalog (training set), the system can't identify it without a complete reorganization (full retraining). This is especially problematic in food classification, where models trained on Western cuisines can't recognise regional dishes like dosa or kimchi. The core issue isn't just missing categories, it's that retraining to include them erases knowledge of previously classified foods (catastrophic forgetting), causing models to fail in real-world applications where food varieties continuously evolve.

## Proposed Approach
The authors created a text-guided continual learning framework that incrementally updates model knowledge using TF-IDF vectors from dish names and ingredients. Instead of retraining from scratch, their system processes new food names through a lightweight neural network that maintains accuracy while adding new categories. The core architecture processes text features (not image features), enabling incremental updates with minimal computational cost.

```python
def incremental_update(model, new_data):
    new_features = tfidf_vectorizer.transform(new_data['item_name'])
    model.fit(new_features, new_data['type'], 
              epochs=1, 
              batch_size=32, 
              validation_split=0.2)
    model.compile(optimizer='adam', 
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model
```

## Key Technical Contributions
The key innovations lie in adapting continual learning for text-based food classification:

1. **Text-First Feature Extraction:** Instead of expensive image features, they leveraged TF-IDF from dish names and ingredients (e.g., "chicken" → non-vegetarian, "lentil" → vegetarian), reducing computational overhead for incremental updates. This eliminated the need for manual labelling of the initial 25,192 training examples through a keyword-driven heuristic classification process.

2. **Minimalist Architecture for Adaptability:** The carefully designed 64-32 neuron feedforward network employed L2 regularisation (0.01) and early stopping (patience=5) to prevent catastrophic forgetting during incremental updates. This maintained 98.38% accuracy with just 100 epochs of training, unlike traditional approaches that require full retraining.

3. **Incremental Training Protocol:** Unlike full retraining, their protocol processes new data in small batches (32 samples) with just 1-10 epochs per update. This makes it computationally efficient for production systems while maintaining stability, evidenced by their ±2.32% accuracy variation factor, which they attribute to training set diversity rather than architectural limitations.

4. **Keyword-Driven Heuristic Classification:** They developed a simple classification rule where "foods with 'salad' or 'tofu' = vegetarian, foods with 'chicken' or 'beef' = non-vegetarian." This eliminated manual labelling for the initial dataset and provided an effective way to discriminate between food categories based on textual evidence.

## Experimental Results
The model achieved 98.38% ± 2.32% accuracy, 96.90% ± 1.35% precision, 99.36% ± 0.14% recall, and a 98.11% ± 0.67% F1 score on the 25,192-item dataset (20,153 training, 5,039 test). The model correctly classified 12,145 Non-Vegetarian and 12,496 Vegetarian samples, with only 100 misclassifications in the Vegetarian class and 451 in the Non-Vegetarian class. Compared to baselines, the model achieved the best balance between precision and recall (F1 score of 98.11% ± 0.67%), outperforming SVM (99.13% accuracy) in terms of adaptability without catastrophic forgetting.

The authors specifically note the model's resistance to catastrophic forgetting, which is critical for production systems needing ongoing adaptation. The accuracy variation factor (±2.32%) was attributed to limited dataset diversity rather than model limitations.

## Related Work
This work bridges a gap in food classification literature by applying continual learning to text-guided classification. Previous works like Food-101, UECFOOD-256, and VireoFood-172 focused on static image-based datasets but didn't address the need for incremental updates. The authors reference approaches like Elastic Weight Consolidation (EWC), gradient episodic memory, and knowledge distillation from NLP, but note these are underutilized in food image classification. Their contribution is the first to apply continual learning specifically to text-guided food classification, leveraging linguistic patterns in food names rather than image features.

## Limitations
The authors acknowledge the training data had limited representation of real-world variations, leading to poor generalisation and bias toward certain classes. The model's accuracy variation (±2.32%) was explicitly attributed to the lack of diversity in the training set. They also note insufficient coverage of edge cases. From a practitioner perspective, the most significant limitation is the reliance on text features (dish names) rather than visual features, meaning the model can't recognise a dish visually if the name is ambiguous or missing, which may be problematic for real-world applications where images are the primary input.

## Appendix: Worked Example
Let's walk through how the model handles a new dish, "Prawn Curry," not in the original training data:

1. **Initial Classification:** The system scans "Prawn Curry" for keywords: "prawn" is a clear indicator of non-vegetarian (as per their heuristic classification).

2. **TF-IDF Vectorization:** The dish name is converted to a TF-IDF vector (5000 features max), with "prawn" having high importance relative to the corpus.

3. **Incremental Update:** The model processes this vector through the existing neural network (64-32 neurons, ReLU activation) and updates weights incrementally. Using their protocol, this requires just 1 epoch with a batch size of 32.

4. **Prediction:** The output layer produces a high probability (0.98) for non-vegetarian due to the word "prawn," without altering the weights for previously learned categories like "chicken curry."

5. **Knowledge Integration:** The L2 regularisation (0.01) and early stopping (patience=5) prevent catastrophic forgetting, ensuring the model maintains accuracy for existing categories while adding new ones.

6. **Result:** "Prawn Curry" is correctly classified as non-vegetarian with minimal computational cost (a few seconds per dish), while maintaining 98.38% accuracy for all previously learned categories.

This process demonstrates the system's ability to adapt to new food categories without significant retraining costs, which is critical for production systems needing ongoing updates.

## References

- Piyush Kaushik Bhattacharyya, Devansh Tomar, Shubham Mishra, Divyanshu Rai, Yug Pratap Singh, Harsh Yadav, Krutika Verma, Vishal Meena, N Sangita Achary, "Continual Learning for Food Category Classification Dataset: Enhancing Model Adaptability and Performance", arXiv cs.LG, 2026-03-23.
- Paper URL: https://arxiv.org/abs/2603.19624

Tags: #food-recognition #continual-learning #text-based-classification #incremental-adaptation #tf-idf
