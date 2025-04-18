# HealthAI-Disease-Prediction
A machine learning project that predicts diseases based on symptoms using NLP and classification models.
HealthAI – Symptom-Based Disease Prediction using Machine Learning

📌 Project Overview

HealthAI is an AI-powered health assistant that predicts diseases based on user-entered symptoms. Designed using NLP and supervised learning models, it aims to improve accessibility and early diagnosis for people who may not have immediate access to medical care.

🧠 Objective

Develop a scalable and intelligent disease prediction tool based on natural language symptoms.

Compare various machine learning models for optimal predictive performance.

Enhance early-stage medical guidance using AI.

📊 Dataset

Source: Kaggle - Symptom2Disease

Total Records: 4,920

Features: 41 symptoms mapped to 131 diseases

Used for training and evaluating model predictions.

🔄 Preprocessing

Text normalization (lowercasing, punctuation removal)

Tokenization and stopword removal with NLTK

TF-IDF vectorization to convert text to numerical format

Output used to train classification algorithms

🧪 Models Implemented

Logistic Regression

Naïve Bayes

Support Vector Machine (SVM)

Random Forest (Best Accuracy: ~97.5%)

Gradient Boosting (High-performance alternative)

📈 Evaluation Metrics

Accuracy: Model correctness

Precision: Relevance of predictions

Recall: Sensitivity to actual conditions

F1 Score: Harmonic mean of precision and recall

Confusion Matrix: Visual representation of model performance

▶️ How to Run This Project

Clone or download this repository

Install dependencies:

pip install pandas numpy scikit-learn nltk matplotlib seaborn

Place Symptom2Disease.csv in the root folder

Open and run the notebook: HealthAI_Final_Project_Notebook.ipynb
