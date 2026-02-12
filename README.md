# FUTURE_ML_02
Machine learning system that automatically classifies customer support tickets and predicts priority using NLP. Built with Python, TF-IDF, and Logistic Regression to help businesses organize, prioritize, and respond to customer issues faster while improving support efficiency and decision-making.
# 📌 Support Ticket Classification & Priority Prediction (ML Project)

## 🧠 Project Overview

This project builds a Machine Learning system that automatically classifies customer support tickets and predicts their priority using Natural Language Processing (NLP). It helps businesses organize support requests, identify urgent issues, and respond faster to customers.

This project demonstrates a real-world application of Machine Learning used in SaaS companies, customer support platforms, and IT service management systems.

## 🎯 Objectives

* Automatically classify support tickets into categories
* Predict ticket priority (Low, Medium, High, Critical)
* Reduce manual work for support teams
* Improve response time and efficiency
* Demonstrate practical implementation of NLP and ML

## 🛠️ Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* NLTK (Natural Language Processing)
* TF-IDF Vectorization
* Logistic Regression

## 📂 Project Structure

support_ticket_ml/
│
├── tickets.csv
├── main.py
├── preprocess.py
├── train_model.py
├── predict.py
├── requirements.txt
└── README.md

## 📊 Dataset

Customer Support Ticket dataset containing:

* Ticket Description (input text)
* Ticket Type (category label)
* Ticket Priority (urgency level)

The dataset is used to train models for automatic classification and prioritization of support tickets.

## ⚙️ Features Implemented

* Text cleaning and preprocessing
* Stopword removal using NLTK
* Feature extraction using TF-IDF
* Category classification model
* Priority prediction model
* Model evaluation using accuracy, precision, recall
* Real-time prediction system

## 🤖 How It Works

1. Load dataset and clean text
2. Convert ticket text into numerical features using TF-IDF
3. Train machine learning models
4. Predict:

   * Ticket category
   * Ticket priority
5. Display predictions for new tickets entered by user

## ▶️ How to Run the Project

Install dependencies:
pip install -r requirements.txt

Run the system:
python main.py

Example input:
My payment failed but money got deducted

Example output:
Predicted Category: Billing Inquiry
Predicted Priority: High

## 💼 Business Use Case

This system can help companies:

* Automatically organize customer tickets
* Identify urgent issues quickly
* Improve response time
* Reduce manual effort
* Enhance customer satisfaction

## 🏁 Conclusion

This project demonstrates how Machine Learning and Natural Language Processing can be applied to automate customer support operations. It provides a practical solution for ticket classification and prioritization, helping organizations improve efficiency and service quality.
