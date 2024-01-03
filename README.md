# Sentiment-Analysis
This app uses Natural Language Processing (NLP) libraries like NLTK and  Logistic Regression (machine learning model) to predict the sentiment of a tweet. It has been deployed using Streamlit.
# Sentiment Analysis using TF-IDF, NLTK, and Logistic Regression

This repository contains a Sentiment Analysis project that utilizes TF-IDF (Term Frequency-Inverse Document Frequency), NLTK (Natural Language Toolkit), and Logistic Regression to analyze sentiment in textual data.

## Overview

Sentiment Analysis is the process of determining the sentiment or opinion expressed in text data. This project focuses on using machine learning techniques, specifically the TF-IDF vectorization method, NLTK library for natural language processing, and Logistic Regression as a classification algorithm, to perform sentiment analysis on textual data.

## Features

- **TF-IDF Vectorization:** Utilizes TF-IDF to convert text data into numerical vectors, representing the importance of words in a document relative to a collection of documents.
- **NLTK (Natural Language Toolkit):** Employs NLTK for various natural language processing tasks such as tokenization, stemming, and stop-word removal.
- **Logistic Regression:** Implements Logistic Regression, a popular classification algorithm, to predict sentiment based on the features extracted using TF-IDF.

## Setup and Usage

1. **Installation:**
   - Clone the repository:
     ```bash
     git clone https://github.com/vishal91-hub/Sentiment-Analysis.git
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Training and Evaluation:**
   - Train the model using provided dataset:
     ```bash
     python train.py
     ```
   - Evaluate the model:
     ```bash
     python evaluate.py
     ```

3. **Usage:**
   - Use the trained model for sentiment analysis on new text data:
     ```python
     # Example code snippet
     from sentiment_analyzer import SentimentAnalyzer

     sa = SentimentAnalyzer(model_path='path/to/saved/model')
     result = sa.analyze_sentiment("Your text here")
     print(result)
     ```

## Dependencies

- Python 3.x
- NLTK
- scikit-learn
- Other necessary libraries (specified in `requirements.txt`)

## Contributors

- [Your Name](https://github.com/your_username)
- [Contributor 1](https://github.com/contributor_1)
- [Contributor 2](https://github.com/contributor_2)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
