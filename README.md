AI Resume Classifier

A machine learning-based resume classification system that categorizes resumes into different job domains using TF-IDF vectorization and Logistic Regression. Perfect for recruiters, HRs, or personal projects.
 
# Features:

Classifies resumes into 5 categories:

- Data Science
- Backend Development
- Web Development
- Android Development
- Programming.


Additional features:

* Supports single resume classification or multiple resumes at once.
* Built with Python, Pandas, scikit-learn, and Streamlit.
* Interactive web interface for quick testing and demo.
* Open-source and easy to modify for learning and personal projects.







💻 Installation:

1.Install Python 3: python.org 
2.Install required libraries:


```pip install pandas scikit-learn streamlit```




 
 
 
 🖥️ Run project ( CLI version ):

```python resume_classifier.py```
Enter any resume text when prompted.
Example:
Input:


I have experience in Django, REST APIs, and backend services.
Output:


Predicted Category: Backend Development






💻🖥️ Streamlit Web Version:


```streamlit run app.py```

* Open browser → enter resume text → click Classify.
* Multiple resume files can also be tested using the bulk upload feature.










📁 Dataset:

resumes.csv contains sample resume texts and their categories for training.

* Column:

resume_text: resume content
category: corresponding job category








⚙️ How It Works:

- TF-IDF Vectorizer: Converts text into numerical features.
- Logistic Regression: Trains on sample resumes to classify new inputs.
- Streamlit UI: Provides an interactive interface for testing and deployment.








⚙️ Technology Stack:

- 🐍 Python 3.10+
- 📊 Pandas
- 🤖 scikit-learn
- 🖥️ Streamlit









📒 Usage:

- Recruiters: Quickly categorize incoming resumes.
- HR Automation: Integrate as a backend service for resume screening.
- Students: Learn ML & NLP concepts practically.









📄 License:

This project is open-source and free to use for learning and personal projects.
