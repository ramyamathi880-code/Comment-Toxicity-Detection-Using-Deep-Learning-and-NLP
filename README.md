## Comment-Toxicity-Detection-Using-Deep-Learning-and-NLP
# Toxic Comment Detection System using Deep Learning & Streamlit

## Project Title
Deep Learning for Comment Toxicity Detection with Streamlit

---

## Project Description

Online communities and social media platforms have become an important part of modern communication. However, toxic comments such as harassment, hate speech, and offensive language negatively impact healthy online discussions.

This project aims to build an automated system that can detect toxic comments in real-time using Natural Language Processing (NLP) and Machine Learning techniques. The trained model is deployed as a web application using Streamlit, allowing users to check whether a given comment is toxic or non-toxic.

---

## Objectives

- To preprocess and clean raw text data  
- To train a machine learning model for toxicity detection  
- To deploy the model using Streamlit for real-time predictions  
- To help moderators identify harmful comments automatically  

---

## Technologies Used

- Python  
- Pandas, NumPy  
- NLTK (for text preprocessing)  
- Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression / Linear SVM  
- Streamlit  

---

## Dataset

The dataset contains online comments with labels indicating whether the comment is toxic.

Main columns used:
- comment_text → Input text  
- toxic → Output label (0 = Non-Toxic, 1 = Toxic)

---

## Project Workflow

### 1. Data Exploration and Preparation

- Loaded the dataset using Pandas  
- Checked for missing values and data types  
- Cleaned the text (lowercasing, removing special characters)  
- Performed tokenization, stopword removal, and lemmatization  
- Converted text into numerical form using TF-IDF vectorization  

### 2. Model Development

- Split data into training and validation sets  
- Trained a baseline model using Logistic Regression / Linear SVM  
- Evaluated the model using accuracy and classification metrics  

### 3. Streamlit Application Development

- Saved the trained model and vectorizer  
- Built a Streamlit web application  
- User enters a comment and gets real-time prediction  
- Option to upload CSV file for bulk predictions  

---

## How to Run the Project

### Step 1: Install required libraries

pip install pandas numpy nltk scikit-learn streamlit

### Step 2: Run Streamlit app

streamlit run app.py

---

## Sample Inputs

Toxic Comments:
- You are stupid  
- I hate you  
- Go and die  

Non-Toxic Comments:
- Have a nice day  
- Thank you for your help  
- This is a good project  

---

## Future Improvements

- Use deep learning models like LSTM or BERT for better accuracy  
- Add multi-class classification for different toxicity types  
- Deploy the application on cloud platforms like Heroku or AWS  

---

## Conclusion

This project demonstrates how NLP and machine learning can be used to detect toxic content automatically. The Streamlit-based interface makes the system easy to use for non-technical users and can help maintain healthy online communication.

---

## Author

Ramya  
