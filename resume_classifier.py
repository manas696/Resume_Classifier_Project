import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("resumes.csv")

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["resume_text"])
y = df["category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# User input to classify
new_resume = input("Enter resume text to classify: ")
new_vector = vectorizer.transform([new_resume])
prediction = model.predict(new_vector)
print("Predicted Category:", prediction[0])