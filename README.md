# Sentiment Analysis with LSTM in PyTorch

This repository contains code for performing sentiment analysis on Twitter data using an LSTM (Long Short-Term Memory) neural network. The model is trained to classify tweets into four categories: Irrelevant, Negative, Neutral, and Positive.

## Dataset

The dataset used for this project can be found on Kaggle: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). The `twitter_training.csv` file includes the following columns:
- `id`: Unique identifier for each tweet
- `entity`: The entity the tweet is about
- `sentiment`: The sentiment label for the tweet (Irrelevant, Negative, Neutral, Positive)
- `content`: The text content of the tweet

## Preprocessing Steps

The preprocessing steps involved in preparing the data for the LSTM model include:

1. **Loading and cleaning data**: Read the data from the CSV file and handle any missing or malformed entries.
2. **Normalizing text**: Perform text normalization, which includes:
   - Lowercasing the text
   - Expanding contractions (e.g., converting "can't" to "cannot")
   - Removing URLs, punctuation, and special characters
   - Removing stopwords (common words that do not contribute much to the sentiment)
3. **Tokenizing and padding sequences**: Convert text into sequences of tokens and pad them to ensure uniform input length for the model.

## Model Architecture

The LSTM model is built using PyTorch and includes the following layers:

1. **Embedding layer**: Converts input tokens into dense vectors of fixed size.
2. **LSTM layer**: Processes the sequences of token embeddings, capturing temporal dependencies.
3. **Fully connected layer**: Maps the output of the LSTM to the desired number of sentiment classes (Irrelevant, Negative, Neutral, Positive).

## Usage

### Installation

Ensure you have Python and the necessary libraries installed. You can install the required packages using:

```bash
pip install -r requirements.txt
