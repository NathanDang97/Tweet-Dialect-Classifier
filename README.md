# Tweet Dialect Classifier

## 🧠 Overview
### Project Goal 
This project aims to develop a 3-way dialect classifier that detects whether a tweet is written in Standard English, (Standard) African American English, or African American Vernacular English. The classifier is then used to perform sentiment bias analysis across dialects using multiple popular sentiment models (e.g. BERTweet, RoBERTa, RoBERTa-Latest). The broader goal is to investigate the algorithmic bias in social media sentiment analysis pipelines.

### Project Outline
- **Dataset**:
  - The dialect classifier is trained on a large-scale Twitter corpus labeled via linguistic heuristics obtained [here](https://slanglab.cs.umass.edu/TwitterAAE/) and the implementation for their project can be found [here](https://github.com/slanglab/twitteraae)
  - In addition, we used the AAVE corpora (e.g., jazmiahenry/aave_corpora) to engineer some linguistical features.
  - The sentiment analysis evaluation uses the TWEETEval benchmark, which includes over 60K tweets labeled for sentiment (positive, negative, neutral).
    
- **Model Training**:
  - A transformer-based classifier (e.g., bertweet-base) was trained to distinguish among the three dialect groups.
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
python download_tweetteraae_data.py
```
The script tokenizes and saves the data using Hugging Face's tokenizer with max_length truncation and padding.

2. For sentiment analysis: download the TWEETEval dataset using the following command
```bash
python download_tweeteval_data.py
```
The script not only downloads the raw data, but also shows how you can tokenize it for independent interest. In other words, this project only requires the raw TWEETEval data, tokenizing it is a simple extra feature.

## 📚 Notebooks
There are 3 notebooks (.ipynb) for this project. All of them can be found in the _notebooks_ folder with the following breakdown
1. Dialect Classifier (tweets-classifier.ipynb): contains a pipeline for training a 3-way classifier on the Tweeter AAE Dataset. For simplification, we adapted the naming conventions from the [TweeterAAE project](https://slanglab.cs.umass.edu/TwitterAAE/) (mentioned above) for the 3 groups as: White (for Standard English), AAE-no-AAVE (for Standard Aftican American English), and AAVE (for African American Vernacular English).

2. Sentiment Analysis (sentiment-analysis.ipynb): contains a pipeline for evaluating the performance of the pre-trained BERTweet, RoBERTa, and RoBERTa-Latest for sentiment analysis on the full TWEETEval Dataset. Accuracy and F1-scores, as well as confusion matrices per model are provided. 

3. Statistical Analysis and Additional Evaluation (statistical_analysis.ipynb): contains a pipeline for statistical analysis on the results obtained from the sentiment analysis using Kruskal-Wallis test. In addition, this notebook explores model performance disparities across the 3 dialect groups including: A more in-depth metrics per class, Top confusions pairs per model/group, Cross-model disagreement, and Top most "difficult" tweets (where models got wrong predictions).

## 🔍 Results
### Dialect Classifier Performance
The transformer-based dialect classifier, fine-tuned using the BERTweet model with additional linguistic features (POS tags, n-grams), achieved strong overall performance distinguishing among Standard English (White), Standard African American English (AAE-no-AAVE), and African American Vernacular English (AAVE). Notably, the classifier achieved exceptionally high accuracy (~99%) in identifying AAVE tweets. However, distinguishing between White and AAE-no-AAVE dialects proved more challenging, with accuracy scores ~80%, reflecting the linguistic similarities and nuanced differences between these two dialects.

### Sentiment Analysis
Three sentiment models (BERTweet, RoBERTa, RoBERTa-Latest) were evaluated on the TWEETEval dataset, grouped by dialect:
- BERTweet consistently performed the best across all dialects, achieving accuracy and F1-scores around 82%, slightly outperforming the RoBERTa-based models.
- RoBERTa demonstrated strong performance as well (~80% accuracy and F1-score), slightly behind BERTweet but significantly ahead of RoBERTa-Latest.
- RoBERTa-Latest underperformed relative to the other models, showing noticeably lower accuracy and F1-scores (~75%) across dialect groups.

### Additional Evaluation
Statistical analyses using the Kruskal-Wallis test revealed no statistically significant differences in sentiment analysis performance across dialect groups (AAE-no-AAVE, AAVE, and White). However, further qualitative analysis highlighted subtle differences and areas of model uncertainty:
- Model Disagreement: Approximately 19% overall disagreement among models was found, indicating frequent divergence in sentiment predictions. Disagreement was notably higher for AAE-no-AAVE and AAVE groups (~24%) compared to White tweets in balanced conditions (~1%), suggesting linguistic complexities in African American dialects pose more challenges to existing sentiment models.
- Error Analysis: Qualitative inspection of tweets where models consistently struggled revealed linguistic nuances such as idiomatic expressions, slang usage, zero-copula constructions, and culturally-specific references contributing to misclassification.
- Top Confusion Pairs: Most confusion was found between neutral and negative sentiments across all models, indicating sentiment subtlety as a common challenge, particularly for dialect groups with nuanced emotional expressions.

These insights highlight the need for dialect-specific training and evaluation to improve fairness and reduce bias in sentiment analysis pipelines.

## Acknowledgments
- TwitterAAE (Research on African-American English on Twitter): https://slanglab.cs.umass.edu/TwitterAAE/
- TWEETEval benchmark: https://huggingface.co/datasets/tweet_eval
- AAVE corpora: https://github.com/jazmiahenry/aave_corpora
- Hugging Face Transformers for modeling and tokenizer infrastructure.
