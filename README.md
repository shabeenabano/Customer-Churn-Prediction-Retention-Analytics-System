# Customer-Churn-Prediction-Retention-Analytics-System
## 📌 Project Overview

This project is an end-to-end **Customer Churn Prediction and Retention Analytics System** developed using Python and Machine Learning.

The project analyzes customer information to identify patterns associated with churn, builds and compares multiple classification models, assigns customers to different churn-risk categories, and generates personalized retention strategies based on predicted risk.

The workflow covers **data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, churn probability prediction, risk scoring, and retention recommendations**.

## 🎯 Project Objective

The main objectives of this project are to:

- Analyze customer data and identify patterns associated with customer churn.
- Build and compare machine learning classification models for churn prediction.
- Optimize the selected model using hyperparameter tuning.
- Calculate customer churn probability and categorize customers into different risk levels.
- Generate retention strategies based on predicted customer churn risk.

  ## 🎯 Business Objective

The objective of this system is to help identify customers who are at risk of leaving the service and support proactive retention efforts.

The predicted churn probability and risk category can be used to prioritize customers for appropriate retention actions and improve customer relationship management.

## 🛠️ Technologies Used

- **Programming:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Models:** Logistic Regression, Decision Tree, Random Forest
- **Techniques:** Data Preprocessing, EDA, Feature Engineering, Classification, Hyperparameter Tuning
- **Risk Analysis:** Churn Probability, Risk Categorization, Retention Strategy
- **Environment:** Jupyter Notebook

  ## 🔍 Key Findings

- The dataset contains **7,043 customer records** and **21 features** used for churn analysis.
- Exploratory analysis was performed to understand customer characteristics and factors associated with churn.
- Multiple classification models were trained and compared, including **Logistic Regression, Decision Tree, and Random Forest**.
- Hyperparameter tuning was applied to improve the selected model.
- The system converts predicted churn probability into **Low, Medium, and High Risk** categories.
- Retention actions are generated based on the customer's predicted churn risk.

  ## 📈 Model Results
  

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.21% | 66.91% | 50.00% | 57.23% | 84.06% |
| Decision Tree | 78.36% | 64.78% | 40.05% | 49.50% | 82.17% |
| Random Forest | 78.93% | 64.73% | 44.89% | 53.02% | 83.74% |
| Tuned Logistic Regression | **80.28%** | **66.78%** | **50.81%** | **57.71%** | **84.05%** |


### Final Model

The final model is **Tuned Logistic Regression** with:

- **Best Parameters:** `C = 10`, `solver = liblinear`
- **Accuracy:** 80.28%
- **Precision:** 66.78%
- **Recall:** 50.81%
- **F1-Score:** 57.71%
- **ROC-AUC:** 84.05%
- **Best Cross-Validation ROC-AUC:** 84.74%

  ## 👩‍💻 Author

**Shabeena Bano**

GitHub: https://github.com/shabeenabano

LinkedIn: https://www.linkedin.com/in/shabeena-bano-49861542/
