import streamlit as st
import pickle
import re

# Load saved model and vectorizer
model = pickle.load(open("toxic_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

st.set_page_config(page_title="Toxic Comment Detector")

st.title("💬 Toxic Comment Detection System")
st.write("Enter a comment to check whether it is toxic or not.")

user_input = st.text_area("Enter your comment:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error("⚠️ Toxic Comment Detected")
        else:
            st.success("✅ This is a Non-Toxic Comment")
