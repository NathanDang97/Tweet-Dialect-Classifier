# Tweet Dialect Classifier

## 🧠 Overview
### Project Goal 
This project aims to develop a 3-way dialect classifier that detects whether a tweet is written in Standard English (White), (Standard) African American English (AAE), or African American Vernacular English (AAVE). The classifier is then used to perform sentiment bias analysis across dialects using multiple popular sentiment models (e.g. BERTweet, RoBERTa, RoBERTa-Latest). The broader goal is to investigate the algorithmic bias in social media sentiment analysis pipelines.

### Project Outline
- **Dataset**:
  - The dialect classifier is trained on a large-scale Twitter corpus labeled via linguistic heuristics and AAVE corpora (e.g., jazmiahenry/aave_corpora).
  - The sentiment analysis evaluation uses the TWEETEval benchmark, which includes over 60K tweets labeled for sentiment (positive, negative, neutral).
    
- **Model Training**:
  - A transformer-based classifier (e.g., bert-base-uncased) was trained to distinguish among the three dialect groups.
  - Additional linguistic features were incorporated to enhance dialect distinction.
  - Training and evaluation were conducted on Google Colab Pro+ (A100 GPU), and performance was measured via accuracy and F1-score.
    
- **Sentiment Analysis:**
  - The trained dialect classifier was used to group tweets by dialect.
  - Sentiment predictions from 3 popular models (BERTweet, RoBERTa, and RoBERTa-Latest) were analyzed across dialect groups.
  - Bias and fairness were evaluated using balanced and full datasets, and results were statistically tested using Kruskal-Wallis Test

## 📦 Dataset Setup
1. For the dialect classifier: download or preprocess the dialect corpus using the provided script in the _scripts_ folder. Use the following command to install the requirements and download the dataset
```bash
pip install -r requirements.txt
python download_tweetteraae.py
```
The script tokenizes and saves the data using Hugging Face's tokenizer with max_length truncation and padding.

2. For sentiment analysis: download the TWEETEval dataset using the following command
```bash
python download_tweeteval.py
```
The script not only downloads the raw data, but also shows how you can tokenize it for independent interest. In other words, this project only requires the raw TWEETEval data, tokenizing it is a simple extra feature.

## 📚 Notebooks
There are 3 notebooks (.ipynb) for this project. All of them can be found in the _notebooks_ folder with the following breakdown
1. Dialect Classifier: 

2. Sentiment Analysis

3. Statistical Analysis and Additional Evaluation

## 🔍 Results
### Dialect Classifier Performance

### Sentiment Analysis

### Additional Evaluation

## Acknowledgments
- TWEETEval benchmark: https://huggingface.co/datasets/tweet_eval
- AAVE corpora: https://github.com/jazmiahenry/aave_corpora
- Hugging Face Transformers for modeling and tokenizer infrastructure.
