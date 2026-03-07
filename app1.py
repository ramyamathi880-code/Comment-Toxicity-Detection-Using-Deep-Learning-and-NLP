import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load Model and Tokenizer
# -----------------------------

model = load_model("toxicity_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# -----------------------------
# Text Cleaning Function
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text


# -----------------------------
# Page Title
# -----------------------------

st.title("Deep Learning Comment Toxicity Detector")
st.write("This app detects whether a comment is toxic or non-toxic.")

# -----------------------------
# Single Comment Prediction
# -----------------------------

st.subheader("Check Single Comment")
comment = st.text_area("Enter a comment")
if st.button("Predict"):

    cleaned = clean_text(comment)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200)
    prediction = model.predict(padded)
    if prediction[0][0] > 0.5:
          st.error("Toxic Comment")
    else:
          st.success("Non Toxic Comment")

