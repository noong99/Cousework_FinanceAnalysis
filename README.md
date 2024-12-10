# stats507
This is coursework repo for University of Michigan STATS 507 Data Science and Analytics using Python.

# Project
Financial News Sentiment Analysis with FinBERT and LSTM

### Overview

This project focuses on sentiment analysis of financial news articles to support stock market trend predictions. By leveraging domain-specific models like FinBERT and integrating them with deep learning architectures like LSTM, the project demonstrates the potential to enhance sentiment classification accuracy. The analysis incorporates sentiment scores as an additional feature to enrich the model's understanding of financial sentiment.

### Key Features
- FinBERT for Financial Text: Utilizes FinBERT, a pre-trained model fine-tuned for the financial domain, for sentiment analysis.
- Hybrid Model: Combines FinBERT with LSTM networks to capture sequential dependencies and improve classification performance.
- SMOTE Implementation: Addresses class imbalance in the dataset using Synthetic Minority Over-sampling Technique (SMOTE).
- Comprehensive Evaluation: Models are evaluated using metrics like accuracy, precision, recall, and F1-score on benchmark datasets.

### Dataset
- Source: Stock-related news articles.
- Features Used: Text, sentimentScore, and sentiment labels (Positive, Neutral, Negative).
- Preprocessing:
  - Reduced dataset to 4,000 rows for computational efficiency (similar to FinBERT benchmarks).
  - Stratified 6:2:2 split into training, validation, and testing sets.
  - Applied SMOTE to balance sentiment classes.

