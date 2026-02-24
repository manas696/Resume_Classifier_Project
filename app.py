import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("resumes.csv")

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['resume_text'])
y = df['category']

model = LogisticRegression()
model.fit(X, y)
 
# Streamlit UI
st.title("AI Resume Classifier")
st.write("Enter resume text to classify:")

resume_text = st.text_area("Resume Text")

if st.button("Classify"):
    if resume_text.strip() != "":
        vect_text = vectorizer.transform([resume_text])
        prediction = model.predict(vect_text)[0]
        st.success(f"Predicted Category: {prediction}")
    else:
        st.warning("Please enter some resume text.")