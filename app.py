import streamlit as st
import pickle
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
 # Replace '/path/to/nltk_data' with your NLTK data directory

# Load the Pickle model
with open('sentiment_model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

def label_mapping(label):
    return 'negative' if label == 1 else 'positive'
# Function to preprocess user input
def clean_input(text):
    # Apply the same preprocessing steps used before training the model
    # Example steps: lowercase, remove links, special characters, stopwords, etc.
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"(http\S+|www\S+)", "", text)  # Remove links
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in cleaned_tokens])
    return cleaned_text

# Streamlit app interface
st.title('Sentiment Analysis App')

tweet_input = st.text_input('Enter your tweet:')
if st.button('Predict'):
    if tweet_input:
        # Clean the user input
        cleaned_input = clean_input(tweet_input)
        
        # Make prediction using the model
        prediction = pipeline.predict([cleaned_input])
        predicted_sentiment = label_mapping(prediction[0])
        st.write(f'Predicted Sentiment: {predicted_sentiment}')
    else:
        st.warning('Please enter a tweet for prediction.')
