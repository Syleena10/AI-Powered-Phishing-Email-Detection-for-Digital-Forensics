# AI-Powered-Phishing-Email-Detection-for-Digital-Forensics
AI-powered phishing email detection system designed for digital forensics investigations. This project uses TF-IDF and machine learning to analyze email text as digital evidence, helping identify suspicious communications and support evidence triage during incident response.

## Overview
This project uses Machine Learning (AI) to detect phishing emails as part of a digital forensics investigation workflow. Emails are treated as digital evidence, and the model helps identify suspicious communications for further analysis.

## Objectives
Classify emails as Safe (0) or Phishing (1)
Use TF-IDF to extract meaningful text features
Train a machine learning model to detect phishing patterns
Support forensic evidence triage

## Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib


## Dataset
CSV file containing:
Email Text
Email Type (Safe Email / Phishing Email)



# Step-by-Step Recreation Guide

## Step 1: Install Dependencies
pip install pandas numpy scikit-learn matplotlib


## Step 2: Import Libraries
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


## Step 3: Load Dataset
df = pd.read_csv("Phishing_Email.csv")
print(df.head())


## Step 4: Clean & Prepare Data
df = df[['Email Type', 'Email Text']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({
    'Safe Email': 0,
    'Phishing Email': 1
})

df.dropna(inplace=True)


## Step 5: Text Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df['text'] = df['text'].apply(clean_text)


## Step 6: Train/Test Split
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


## Step 7: TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    token_pattern=r'\b[a-zA-Z]{3,}\b'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


## Step 8: Train Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)


## Step 9: Make Predictions
y_pred = model.predict(X_test_vec)


## Step 10: Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


## Step 11: Visualize Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test_vec, y_test)
plt.show()


## Step 12: Extract Important Words
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

word_importance = list(zip(feature_names, coefficients))

top_phishing = sorted(word_importance, key=lambda x: x[1], reverse=True)[:10]
top_safe = sorted(word_importance, key=lambda x: x[1])[:10]

print("Top Phishing Words:", top_phishing)
print("Top Safe Words:", top_safe)


## Step 13: Visualize Word Importance
words = [w[0] for w in top_phishing]
values = [w[1] for w in top_phishing]

plt.figure()
plt.barh(words, values)
plt.title("Top Phishing Words")
plt.gca().invert_yaxis()
plt.show()
