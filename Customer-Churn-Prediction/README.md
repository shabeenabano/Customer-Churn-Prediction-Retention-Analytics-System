# Customer Churn Prediction & Retention Analytics System

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. This project uses customer data and machine learning to predict the probability of customer churn and identify customers who may be at higher risk of leaving.

The project combines **Exploratory Data Analysis (EDA), Feature Engineering, Machine Learning, Risk Scoring, and Retention Recommendations** into an end-to-end Data Science solution.

## 🎯 Objectives

* Analyze customer behavior and churn patterns
* Identify important factors associated with customer churn
* Build machine learning models for churn prediction
* Compare Logistic Regression, Decision Tree, and Random Forest
* Tune the Logistic Regression model using GridSearchCV
* Generate customer-level churn risk scores
* Categorize customers into Low, Medium, and High Risk
* Generate personalized retention recommendations
* Provide an interactive Streamlit dashboard

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## 📊 Machine Learning Models

The project evaluates the following models:

* Logistic Regression
* Decision Tree
* Random Forest

The models are compared using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Hyperparameter tuning is performed using **GridSearchCV** to improve the Logistic Regression model.

## 🚀 Interactive Dashboard

A Streamlit dashboard is included to allow users to enter customer information and receive:

* Churn probability
* Risk category
* Churn prediction
* Retention recommendation

The trained model and scaler are saved using **Joblib** and loaded directly by the dashboard.

## 📸 Dashboard Preview

The interactive Streamlit dashboard allows users to enter customer information and receive real-time churn predictions and retention recommendations.

![Customer Churn Prediction Dashboard](images/dashboard_prediction.png)


## 📈 Model Performance

Three machine learning models were trained and evaluated on the customer churn dataset.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.21% | 66.91% | 50.00% | 57.23% | 84.06% |
| Decision Tree | 78.36% | 64.78% | 40.05% | 49.50% | 82.17% |
| Random Forest | 78.93% | 64.73% | 44.89% | 53.02% | 83.74% |

### Hyperparameter Tuning

Logistic Regression was further optimized using GridSearchCV with 5-fold cross-validation.

**Best Parameters:**
- C = 10
- Solver = liblinear

**Best Cross-Validation ROC-AUC:** 84.74%

## 🔍 Key Insights

The exploratory data analysis and machine learning workflow helped identify several customer-level patterns associated with churn:

- Month-to-month contract customers showed higher churn levels compared with customers on longer-term contracts.
- Customers without Online Security showed higher churn levels in the analyzed dataset.
- Customers without Technical Support showed higher churn levels in the analyzed dataset.
- Electronic check payment users showed higher churn levels compared with some other payment methods.
- Customer tenure and monthly charges were important variables in understanding churn behavior.
- Customer churn risk was converted into Low, Medium, and High Risk categories for easier business interpretation.

> Note: These findings represent patterns and associations observed in the dataset and should not be interpreted as proof of causation.


## 🔄 Project Workflow

```text
Raw Customer Data
        ↓
Data Cleaning & Quality Checks
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Categorical Encoding
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Machine Learning Models
        ↓
Model Evaluation & Comparison
        ↓
Hyperparameter Tuning
        ↓
Churn Probability Scoring
        ↓
Customer Risk Categorization
        ↓
Retention Recommendations
        ↓
Streamlit Dashboard

## 💡 Retention Strategy

The predicted churn probability is converted into customer risk categories to support targeted retention actions.

| Risk Category | Churn Probability | Suggested Action |
|---|---:|---|
| Low Risk | < 40% | Maintain engagement and monitor customer behavior |
| Medium Risk | 40%–70% | Provide personalized offers and improve service engagement |
| High Risk | ≥ 70% | Prioritize retention campaigns and targeted incentives |

The system also generates recommendations based on customer characteristics, such as:

- Long-term contract incentives for high-risk month-to-month customers
- Online Security service offers
- Technical Support offers
- Automatic payment options
- Personalized retention incentives