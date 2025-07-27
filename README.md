# Movie Review Sentiment Analysis

This project is a deep learning-based sentiment analysis web application that classifies IMDB movie reviews as **positive** or **negative**. It uses a **Simple RNN** with **Leaky ReLU**, and is deployed using **Streamlit** for an interactive web interface.

---

##  Demo

Try out the live demo here (if deployed): **[Add your link]**

---

##  Model Architecture

- **Model Type**: Sequential RNN
- **Embedding Layer**: Converts word indices to dense vectors
- **RNN Layer**: SimpleRNN with one hidden layer
- **Activation**: `LeakyReLU` in hidden layer to prevent vanishing gradients
- **Output Layer**: Dense layer with `sigmoid` activation for binary classification

---

##  Model Summary

| Parameter       | Value                     |
|-----------------|---------------------------|
| Dataset         | IMDB Movie Review Dataset |
| Accuracy        | ~87% on test data         |
| Optimizer       | Adam                      |
| Loss Function   | Binary Crossentropy       |
| Metric          | Accuracy                  |
| Frameworks Used | TensorFlow, Keras         |


##  Streamlit Web App Features

- Clean and interactive UI
- User can input any custom movie review
- Model predicts sentiment (Positive or Negative)
- Shows real-time result using trained model
- Shows Confidence of prediction



