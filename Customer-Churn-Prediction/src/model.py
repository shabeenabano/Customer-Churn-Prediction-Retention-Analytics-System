# Import pandas for data loading and data manipulation
import pandas as pd

# Import joblib for saving the trained model and scaler
import joblib

# Import train_test_split for splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler

# Import Logistic Regression for churn prediction
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

# Load the cleaned customer churn dataset
df = pd.read_csv("../data/telco_customer_churn_cleaned.csv")


# ---------------------------------------------------------
# PREPARE TARGET VARIABLE
# ---------------------------------------------------------

# Separate the target variable from the input features
X = df.drop(columns=["Churn"])

# Store the churn target
y = df["Churn"]


# ---------------------------------------------------------
# ENCODE CATEGORICAL FEATURES
# ---------------------------------------------------------

# Convert categorical columns into numerical dummy variables
X_encoded = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)


# ---------------------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------------------

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------------

# Create the StandardScaler
scaler = StandardScaler()

# Fit the scaler on training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# TRAIN LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------

# Create the Logistic Regression model
model = LogisticRegression(
    C=10,
    solver="liblinear",
    max_iter=1000,
    random_state=42
)

# Train the model using the scaled training data
model.fit(
    X_train_scaled,
    y_train
)


# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

# Save the trained churn prediction model
joblib.dump(
    model,
    "../models/churn_model.pkl"
)

# Save the scaler used during training
joblib.dump(
    scaler,
    "../models/scaler.pkl"
)


# ---------------------------------------------------------
# SAVE FEATURE NAMES
# ---------------------------------------------------------

# Save the feature names used by the model
with open("../models/feature_names.txt", "w") as file:

    for feature in X_train.columns:

        file.write(feature + "\n")


# ---------------------------------------------------------
# CONFIRMATION
# ---------------------------------------------------------

# Display a success message
print("Model training completed successfully!")

# Display the number of training features
print(f"Number of features: {X_train.shape[1]}")

# Display the number of training records
print(f"Training records: {X_train.shape[0]}")

# Display the number of test records
print(f"Testing records: {X_test.shape[0]}")