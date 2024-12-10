
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

### Methodology

1. FinBERT Sentiment Analysis:
    - Pre-trained FinBERT model applied to classify sentiment in financial text.
2. LSTM Model Training:
    - Sequential dependencies in text captured through LSTM networks.
3. Hybrid Approach:
    - FinBERT-generated embeddings combined with LSTM for enhanced performance.

### Results
The results indicate that FinBERT achieves the best overall accuracy, while the hybrid model demonstrates the potential to capture sequential patterns and sentiment for better overall understanding.

### How to Run
To run this project:

1. Download the dataset file: [Dataset](/data/news_data_sampled.csv)
2. Open the relevant .ipynb file in the Project directory, depending on the model you want to execute:
  - [Sentiment Analysis - FinBERT](/sentiment_finbert.ipynb)
  - [Sentiment Analysis - LSTM](/sentiment_lstm_bert.ipynb
  - [Sentiment Analysis - FinBERT + LSTM](/sentiment_finbert_lstm.ipynb)
3. Run the selected notebook to perform sentiment analysis!

### Project Directory Structure

```plaintext
Project  
│  
├── data_analysis  
│   └── data_analysis.ipynb  
│  
├── models  
│   ├── model_finbert.py  
│   ├── model_lstm_bert.py  
│   └── model_finbert_lstm.py  
│  
├── docs  
│   ├── ProjectProposal.pdf  
│   └── FinalProject.pdf  
│  
├── data  
│   └── news_data_sampled.csv  
│  
├── sentiment_finbert.ipynb  
├── sentiment_lstm_bert.ipynb  
└── sentiment_finbert_lstm.ipynb  
