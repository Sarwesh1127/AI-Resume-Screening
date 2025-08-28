
import pandas as pd
import numpy as np
import re
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.read_csv("data\AI_Resume_Screening.csv")

if 'Decision' not in df.columns:
    np.random.seed(42)
    df['Decision'] = np.random.choice(['Hire', 'Reject'], size=len(df))

def combine_text(row):
    return f"{row['Skills']} {row['Education']} {str(row['Certifications'])}"

df['resume_text'] = df.apply(combine_text, axis=1)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['cleaned_text'] = df['resume_text'].apply(preprocess_text)

# Features & labels
X = df['cleaned_text']
y_role = df['Job Role']
y_decision = df['Decision']

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Train/Test split
X_train, X_test, y_train_role, y_test_role, y_train_dec, y_test_dec = train_test_split(
    X_tfidf, y_role, y_decision, test_size=0.2, random_state=42
)

# Model 1: Role classifier
model_role = LogisticRegression(max_iter=200)
model_role.fit(X_train, y_train_role)
print("Role Accuracy:", metrics.accuracy_score(y_test_role, model_role.predict(X_test)))

# Model 2: Hire/Reject classifier
model_decision = LogisticRegression(max_iter=200)
model_decision.fit(X_train, y_train_dec)
print("Decision Accuracy:", metrics.accuracy_score(y_test_dec, model_decision.predict(X_test)))

# Save models & vectorizer
joblib.dump(model_role, "resume_role_classifier.pkl")
joblib.dump(model_decision, "hire_reject_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Training complete. Models saved.")
