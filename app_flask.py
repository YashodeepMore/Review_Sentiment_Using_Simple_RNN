from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and word index
model = load_model('simple_rnn_imdb.h5')
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Preprocessing
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, -1) + 3 for word in words if word_index.get(word, -1) < 10000]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction
def predict_sentiment(review):
    processed = preprocess_text(review)
    prediction = model.predict(processed)
    sentiment = "positive" if prediction[0][0] > 0.48 else "negative"
    confidence = prediction[0][0] if sentiment == "positive" else 1 - prediction[0][0]
    return sentiment, round(confidence * 100, 2)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    review = data.get("review", "").strip()
    if not review:
        return jsonify({"error": "Empty review received."})

    try:
        sentiment, confidence = predict_sentiment(review)
        return jsonify({
            "sentiment": sentiment,
            "confidence": float(confidence)  # ✅ convert to native float
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
