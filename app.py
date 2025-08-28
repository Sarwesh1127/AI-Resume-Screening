import streamlit as st
import joblib
import re
import string
import nltk
import docx2txt
import fitz  # PyMuPDF for PDFs
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PIL import Image
import io
import tempfile
import os
from docx2pdf import convert  # for DOCX → PDF


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

model_role = joblib.load("resume_role_classifier.pkl")
model_decision = joblib.load("hire_reject_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def pdf_to_images(file):
    images = []
    pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High resolution
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images

def docx_to_images(file):
    with tempfile.TemporaryDirectory() as tmpdir:
        docx_path = os.path.join(tmpdir, "resume.docx")
        pdf_path = os.path.join(tmpdir, "resume.pdf")

        with open(docx_path, "wb") as f:
            f.write(file.read())

        convert(docx_path, pdf_path)
        images = []
        pdf_doc = fitz.open(pdf_path)
        try:
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
        finally:
            pdf_doc.close()
        text = docx2txt.process(docx_path)
    return images, text


# Streamlit UI
st.set_page_config(page_title="AI Resume Screening System", page_icon="📄", layout="wide")

st.title("📄 AI-based Resume Screening System")
st.markdown("#### Upload a candidate’s resume and let AI predict the **Job Role** and **Hiring Suggestion** 🎯")

uploaded_file = st.file_uploader("📂 Upload Resume (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    resume_text = ""

    st.subheader("📜 Uploaded Resume (Preview)")

    if file_ext == "pdf":
        images = pdf_to_images(uploaded_file)
        for img in images:
            st.image(img,  use_container_width=True)
        uploaded_file.seek(0)
        resume_text = extract_text_from_pdf(uploaded_file)

    elif file_ext == "docx":
        images, resume_text = docx_to_images(uploaded_file)
        for img in images:
            st.image(img,  use_container_width=True)

    elif file_ext == "txt":
        resume_text = uploaded_file.read().decode("utf-8")
        st.text_area("TXT Resume Preview", resume_text, height=400)


    # Predictions
    if resume_text.strip():
        processed = preprocess_text(resume_text)
        vectorized = vectorizer.transform([processed])

        role_pred = model_role.predict(vectorized)[0]
        role_conf = model_role.predict_proba(vectorized).max() * 100
        decision_pred = model_decision.predict(vectorized)[0]
        decision_conf = model_decision.predict_proba(vectorized).max() * 100

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🎯 Predicted Job Role")
            st.success(f"**{role_pred}** ({role_conf:.2f}% confidence)")

        with col2:
            st.markdown("### ✅ Hiring Suggestion")
            if decision_pred.lower() == "hire":
                st.success(f"**{decision_pred}** ({decision_conf:.2f}% confidence)")
            else:
                st.error(f"**{decision_pred}** ({decision_conf:.2f}% confidence)")
    else:
        st.error("❌ Could not extract resume text. Please check file format.")






# import streamlit as st
# import joblib
# import re
# import string
# import nltk
# import docx2txt
# import fitz  # PyMuPDF for PDFs
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from PIL import Image
# import io

# # =========================
# # 🔹 NLTK Setup
# # =========================
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# # =========================
# # 🔹 Load Pre-trained Models
# # =========================
# model_role = joblib.load("resume_role_classifier.pkl")
# model_decision = joblib.load("hire_reject_classifier.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")

# # =========================
# # 🔹 Preprocessing Utilities
# # =========================
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def preprocess_text(text):
#     text = clean_text(text)
#     return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

# # =========================
# # 🔹 File Extraction Helpers
# # =========================
# def extract_text_from_pdf(file):
#     with fitz.open(stream=file.read(), filetype="pdf") as doc:
#         text = ""
#         for page in doc:
#             text += page.get_text()
#     return text

# def extract_text_from_docx(file):
#     return docx2txt.process(file)

# # Convert PDF to Images
# def pdf_to_images(file):
#     images = []
#     pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
#     for page_num in range(len(pdf_doc)):
#         page = pdf_doc[page_num]
#         pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High resolution
#         img = Image.open(io.BytesIO(pix.tobytes("png")))
#         images.append(img)
#     return images


# # Streamlit UI
# st.set_page_config(page_title="AI Resume Screening System", page_icon="📄", layout="wide")

# st.title("📄 AI-based Resume Screening System")
# st.markdown("#### Upload a candidate’s resume and let AI predict the **Job Role** and **Hiring Suggestion** 🎯")

# uploaded_file = st.file_uploader("📂 Upload Resume (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])

# if uploaded_file is not None:
#     file_ext = uploaded_file.name.split(".")[-1].lower()
#     resume_text = ""

#     st.subheader("📜 Uploaded Resume (Preview)")
    
#     if file_ext == "pdf":
#         images = pdf_to_images(uploaded_file)
#         for img in images:
#             st.image(img, use_column_width=True)
#         uploaded_file.seek(0)  
#         resume_text = extract_text_from_pdf(uploaded_file)

#     elif file_ext == "docx":
#         resume_text = extract_text_from_docx(uploaded_file)
#         st.text_area("DOCX Content Preview", resume_text, height=300)

#     elif file_ext == "txt":
#         resume_text = uploaded_file.read().decode("utf-8")
#         st.text_area("TXT Content Preview", resume_text, height=300)


#     # AI Predictions
#     if resume_text.strip():
#         processed = preprocess_text(resume_text)
#         vectorized = vectorizer.transform([processed])

#         role_pred = model_role.predict(vectorized)[0]
#         role_conf = model_role.predict_proba(vectorized).max() * 100
#         decision_pred = model_decision.predict(vectorized)[0]
#         decision_conf = model_decision.predict_proba(vectorized).max() * 100

#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown("### 🎯 Predicted Job Role")
#             st.success(f"**{role_pred}** ({role_conf:.2f}% confidence)")

#         with col2:
#             st.markdown("### ✅ Hiring Suggestion")
#             if decision_pred.lower() == "hire":
#                 st.success(f"**{decision_pred}** ({decision_conf:.2f}% confidence)")
#             else:
#                 st.error(f"**{decision_pred}** ({decision_conf:.2f}% confidence)")
#     else:
#         st.error("❌ Could not extract resume text. Please check file format.")






# import streamlit as st
# import joblib
# import re
# import string
# import nltk
# import docx2txt
# import fitz  # PyMuPDF for PDFs
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# model_role = joblib.load("resume_role_classifier.pkl")
# model_decision = joblib.load("hire_reject_classifier.pkl")
# vectorizer = joblib.load("tfidf_vectorizer.pkl")

# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'\[.*?\]', '', text)
#     text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
#     text = re.sub(r'\w*\d\w*', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def preprocess_text(text):
#     text = clean_text(text)
#     return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

# def extract_text_from_pdf(file):
#     with fitz.open(stream=file.read(), filetype="pdf") as doc:
#         text = ""
#         for page in doc:
#             text += page.get_text()
#     return text

# def extract_text_from_docx(file):
#     return docx2txt.process(file)

# st.set_page_config(page_title="AI Resume Screening System", page_icon="📄", layout="wide")

# # CSS for look
# st.markdown("""
#     <style>
#         .main {
#             background-color: #f8f9fa;
#         }
#         .stButton>button {
#             background-color: #007bff;
#             color: white;
#             border-radius: 10px;
#             padding: 0.5em 1.5em;
#             font-size: 16px;
#         }
#         .stButton>button:hover {
#             background-color: #0056b3;
#         }
#         .stTextInput>div>div>input {
#             border-radius: 8px;
#         }
#         .reportview-container .markdown-text-container {
#             font-family: "Helvetica Neue", sans-serif;
#         }
#     </style>
# """, unsafe_allow_html=True)


# st.title("📄 AI-based Resume Screening System")
# st.markdown("#### Upload a candidate’s resume and let AI predict the **Job Role** and **Hiring Suggestion** 🎯")

# uploaded_file = st.file_uploader("📂 Upload Resume (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"])

# if uploaded_file is not None:
#     file_ext = uploaded_file.name.split(".")[-1].lower()
#     resume_text = ""
    
#     if file_ext == "pdf":
#         resume_text = extract_text_from_pdf(uploaded_file)
#     elif file_ext == "docx":
#         resume_text = extract_text_from_docx(uploaded_file)
#     elif file_ext == "txt":
#         resume_text = uploaded_file.read().decode("utf-8")
    
#     if resume_text.strip():
#         st.subheader("📜 Extracted Resume Text (Preview)")
#         st.write(resume_text[:500] + ("..." if len(resume_text) > 500 else ""))

#         processed = preprocess_text(resume_text)
#         vectorized = vectorizer.transform([processed])

#         role_pred = model_role.predict(vectorized)[0]
#         role_conf = model_role.predict_proba(vectorized).max() * 100
#         decision_pred = model_decision.predict(vectorized)[0]
#         decision_conf = model_decision.predict_proba(vectorized).max() * 100
#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown("### 🎯 Predicted Job Role")
#             st.success(f"**{role_pred}** \n({role_conf:.2f}% confidence)")

#         with col2:
#             st.markdown("### ✅ Hiring Suggestion")
#             if decision_pred.lower() == "hire":
#                 st.success(f"**{decision_pred}** \n({decision_conf:.2f}% confidence)")
#             else:
#                 st.error(f"**{decision_pred}** \n({decision_conf:.2f}% confidence)")
#     else:
#         st.error("❌ Could not read text from file. Please check the format.")
# else:
#     st.info("👆 Upload a resume file above to get started.")



