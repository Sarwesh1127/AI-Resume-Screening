
import streamlit as st
import joblib
import re
import string
import nltk
import docx2txt
import fitz  # PyMuPDF for PDFs
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load models and vectorizer
model_role = joblib.load("resume_role_classifier.pkl")
model_decision = joblib.load("hire_reject_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Stop words & Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

# File extraction
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

# UI
st.title("📄 AI-based Resume Screening System")
st.write("Upload a resume and get Job Role + Hire/Reject prediction.")

uploaded_file = st.file_uploader("Upload Resume (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    resume_text = ""
    
    if file_ext == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_ext == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    elif file_ext == "txt":
        resume_text = uploaded_file.read().decode("utf-8")
    
    if resume_text.strip():

        st.subheader("📜 Extracted Resume Text Preview:")
        st.text(resume_text[:200] + ("..." if len(resume_text) > 2000 else ""))
        
        processed = preprocess_text(resume_text)
        vectorized = vectorizer.transform([processed])
        processed = preprocess_text(resume_text)
        vectorized = vectorizer.transform([processed])
        
        # Predictions
        role_pred = model_role.predict(vectorized)[0]
        role_conf = model_role.predict_proba(vectorized).max() * 100
        decision_pred = model_decision.predict(vectorized)[0]
        decision_conf = model_decision.predict_proba(vectorized).max() * 100
        
        st.subheader("🎯 Predicted Job Role:")
        st.success(f"{role_pred} ({role_conf:.2f}% confidence)")
        
        st.subheader("✅ Hiring Suggestion:")
        st.info(f"{decision_pred} ({decision_conf:.2f}% confidence)")
    else:
        st.error("Could not read text from file. Please check format.")
