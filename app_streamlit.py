import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model('SimpleRNN/simple_rnn_imdb.h5')

# Load the trained model and vectorizer
# with open("sentiment_model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# with open("vectorizer.pkl", "rb") as vec_file:
#     vectorizer = pickle.load(vec_file)

# App layout


def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, -1) + 3 for word in words if word_index.get(word, -1) < 10000]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review



def predict_sentiment(review):

    processed_review = preprocess_text(review)

    prediction = model.predict(processed_review)

    sentiment = 'Positive 😊' if prediction[0][0] > 0.48 else 'Negative 😞'

    return sentiment, prediction[0][0]




st.set_page_config(page_title="Movie Review Sentiment Classifier", page_icon="🎬", layout="centered")
st.title("🎬 Movie Review Sentiment Classifier")

st.markdown(
    """
    <div style='text-align: center;'>
        <h3>Type your movie review below:</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Input field
review = st.text_area("Your Review", height=200, placeholder="Enter your review here...")

# Prediction logic
if st.button("Classify Review"):
    if review.strip() == "":
        st.warning("Please enter a review to classify.")
    else:
        # review_vector = vectorizer.transform([review])

        # prediction = model.predict(review_vector)
        # probability = model.predict_proba(review_vector)
        
        # sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
        # confidence = np.max(probability) * 100
        sentiment,score=predict_sentiment(review)

        score = 1-score if sentiment=='Negative 😞' else score
        
        # confi = f"{(score*100):.2f}%"

        st.markdown(
            f"<div style='text-align: center; font-size: 24px; padding: 20px;'>\n"
            f"<strong>Sentiment:</strong> {sentiment}<br>"
            f"<strong>Confidence:</strong> {(score*100):.2f}%"
            f"</div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown(
    """
    <hr>
    <div style='text-align: center;'>
        <small>Developed with  using Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)
