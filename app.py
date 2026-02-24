import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("resumes.csv")

# Ensure correct column names (fix KeyError)
df.rename(columns={'category': 'Category', 'resume_text': 'Resume'}, inplace=True)

# -----------------------
# Train model
# -----------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Resume'])
y = df['Category']

model = LogisticRegression()
model.fit(X, y)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AI Resume Classifier", layout="wide")

st.title("💼 AI Resume Classifier")
st.markdown(
    """
    This app classifies resumes into categories like Data Science, Web Development, Backend Development, Android Development, Programming, etc.
    - Enter a single resume or upload multiple resumes as a CSV.
    """
)

# Tabs for single vs multiple resumes
tab1, tab2 = st.tabs(["Single Resume", "Multiple Resumes"])

# -----------------------
# Single Resume Classification
# -----------------------
with tab1:
    st.subheader("Single Resume Classification")
    resume_text = st.text_area("Enter Resume Text Here")
    if st.button("Classify Single Resume"):
        if resume_text.strip() != "":
            vect_text = vectorizer.transform([resume_text])
            prediction = model.predict(vect_text)[0]
            st.success(f"Predicted Category: **{prediction}**")
        else:
            st.warning("Please enter some resume text.")

# -----------------------
# Multiple Resume Classification
# -----------------------
with tab2:
    st.subheader("Multiple Resume Classification (CSV Upload)")
    uploaded_file = st.file_uploader("Upload CSV with column 'resume_text'", type="csv")
    if uploaded_file is not None:
        try:
            df_multi = pd.read_csv(uploaded_file)
            if 'resume_text' not in df_multi.columns:
                st.error("CSV must contain column: 'resume_text'")
            else:
                vect_text = vectorizer.transform(df_multi['resume_text'])
                predictions = model.predict(vect_text)
                df_multi['Predicted Category'] = predictions
                st.success("Classification Completed!")
                st.dataframe(df_multi)
                
                # Optional: Download results
                csv = df_multi.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Classified CSV",
                    data=csv,
                    file_name='classified_resumes.csv',
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"Error reading CSV: {e}")