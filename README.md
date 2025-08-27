# Customer-Churn-Prediction

This project predicts customer churn (whether a customer will leave or stay) using machine learning models on the Telco Customer Churn dataset.
It also includes an interactive dashboard where users can upload their own CSV files and instantly get churn predictions

#Project Overview
Customer churn is a major challenge in subscription-based businesses. The goal of this project is to build models that can classify whether a customer is likely to churn based on their demographics, services, and payment methods.

Key highlights:

-Data preprocessing (handling categorical variables, missing values, and encoding).

-Trained Logistic Regression and Random Forest classifiers.

-Compared models using metrics such as Accuracy, Precision, Recall, and F1-score.

-Built a Python-based Dashboard for real-time predictions on user-provided datasets.

#📈 Results
Logistic Regression
Metric	Churn = 0	Churn = 1
Precision	0.85	0.65
Recall	0.89	0.56
F1-Score	0.87	0.60

Accuracy: 80.4%

Random Forest
Metric	Churn = 0	Churn = 1
Precision	0.83	0.63
Recall	0.89	0.49
F1-Score	0.86	0.55

Accuracy: 78.7%

#🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

Streamlit / Tkinter / Dash (for dashboard)

