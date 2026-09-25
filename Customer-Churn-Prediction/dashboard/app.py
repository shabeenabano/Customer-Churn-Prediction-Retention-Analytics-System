# Import Streamlit for creating the interactive web dashboard
import streamlit as st

# Import pandas for handling customer input data
import pandas as pd

# Import numpy for numerical operations
import numpy as np

# Import joblib for loading the trained machine learning model and scaler
import joblib

# Import os for handling file paths
import os


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

# Configure the Streamlit page
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODEL FILES
# ---------------------------------------------------------

# Get the directory of the current app.py file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the trained churn model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "churn_model.pkl"
)

# Define the path to the saved scaler
SCALER_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "scaler.pkl"
)

# Define the path to the saved feature names
FEATURE_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "feature_names.txt"
)


# Load the trained Logistic Regression model
model = joblib.load(MODEL_PATH)

# Load the StandardScaler used during training
scaler = joblib.load(SCALER_PATH)

# Load the feature names used by the model
with open(FEATURE_PATH, "r") as file:
    feature_names = [line.strip() for line in file.readlines()]


# ---------------------------------------------------------
# DASHBOARD TITLE
# ---------------------------------------------------------

# Display the main dashboard title
st.title("📊 Customer Churn Prediction & Retention System")

# Display a short description
st.write(
    "Predict customer churn probability and generate "
    "a retention recommendation using a trained machine learning model."
)


# ---------------------------------------------------------
# CUSTOMER INPUT SECTION
# ---------------------------------------------------------

# Create a section heading
st.header("Enter Customer Information")


# Create two columns for better dashboard layout
col1, col2 = st.columns(2)


# ---------------------------------------------------------
# BASIC CUSTOMER INFORMATION
# ---------------------------------------------------------

with col1:

    # Select customer's gender
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    # Select whether customer is a senior citizen
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    # Select whether customer has a partner
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    # Select whether customer has dependents
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    # Enter customer tenure in months
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )


# ---------------------------------------------------------
# SERVICE INFORMATION
# ---------------------------------------------------------

with col2:

    # Select the customer's contract type
    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    # Select internet service
    internet_service = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    # Select online security service
    online_security = st.selectbox(
        "Online Security",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    # Select technical support service
    tech_support = st.selectbox(
        "Tech Support",
        [
            "No",
            "Yes",
            "No internet service"
        ]
    )

    # Select payment method
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# ---------------------------------------------------------
# BILLING INFORMATION
# ---------------------------------------------------------

# Create another section heading
st.subheader("Billing Information")


# Create two columns for billing inputs
bill_col1, bill_col2 = st.columns(2)


with bill_col1:

    # Enter monthly charges
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )


with bill_col2:

    # Calculate total charges using tenure and monthly charges
    total_charges = tenure * monthly_charges

    # Display calculated total charges
    st.metric(
        "Estimated Total Charges",
        f"${total_charges:,.2f}"
    )


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------

# Create the prediction button
predict_button = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)


# ---------------------------------------------------------
# PREDICTION LOGIC
# ---------------------------------------------------------

if predict_button:

    # Create a dictionary containing the customer's input data
    customer_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No phone service",
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": tech_support,
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "TenureGroup": (
            "0-12 Months" if tenure <= 12
            else "13-24 Months" if tenure <= 24
            else "25-48 Months" if tenure <= 48
            else "49-72 Months"
        ),
        "AverageMonthlyRevenue": (
            total_charges / max(tenure, 1)
        )
    }


    # Convert customer dictionary into a DataFrame
    customer_df = pd.DataFrame([customer_data])


    # -----------------------------------------------------
    # ONE-HOT ENCODING
    # -----------------------------------------------------

    # Convert categorical variables into dummy variables
    customer_encoded = pd.get_dummies(
        customer_df,
        drop_first=True,
        dtype=int
    )


    # -----------------------------------------------------
    # MATCH TRAINING FEATURES
    # -----------------------------------------------------

    # Add missing features that existed during training
    # but are not present in the current customer input
    for feature in feature_names:

        if feature not in customer_encoded.columns:

            customer_encoded[feature] = 0


    # Keep only the features used during model training
    customer_encoded = customer_encoded[
        feature_names
    ]


    # -----------------------------------------------------
    # SCALE INPUT DATA
    # -----------------------------------------------------

    # Apply the same scaler used during model training
    customer_scaled = scaler.transform(
        customer_encoded
    )


    # -----------------------------------------------------
    # CHURN PREDICTION
    # -----------------------------------------------------

    # Generate churn probability
    churn_probability = model.predict_proba(
        customer_scaled
    )[0][1]


    # Convert probability into percentage
    churn_percentage = churn_probability * 100


    # -----------------------------------------------------
    # RISK CATEGORY
    # -----------------------------------------------------

    # Assign a risk category based on churn probability
    if churn_probability >= 0.70:

        risk_category = "High Risk"

    elif churn_probability >= 0.40:

        risk_category = "Medium Risk"

    else:

        risk_category = "Low Risk"


    # -----------------------------------------------------
    # RETENTION RECOMMENDATION
    # -----------------------------------------------------

    # Generate a retention recommendation
    if (
        churn_probability >= 0.70
        and contract == "Month-to-month"
    ):

        recommendation = (
            "Offer a long-term contract incentive."
        )

    elif online_security == "No":

        recommendation = (
            "Offer Online Security service."
        )

    elif tech_support == "No":

        recommendation = (
            "Offer Technical Support service."
        )

    elif payment_method == "Electronic check":

        recommendation = (
            "Promote automatic payment options."
        )

    else:

        recommendation = (
            "Offer a personalized retention incentive."
        )


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    st.subheader("Prediction Result")


    # Create three columns for result metrics
    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        # Display churn probability
        st.metric(
            "Churn Probability",
            f"{churn_percentage:.2f}%"
        )


    with result_col2:

        # Display risk category
        st.metric(
            "Risk Category",
            risk_category
        )


    with result_col3:

        # Display model decision
        prediction = (
            "Likely to Churn"
            if churn_probability >= 0.50
            else "Likely to Stay"
        )

        st.metric(
            "Prediction",
            prediction
        )


    # -----------------------------------------------------
    # RETENTION RECOMMENDATION DISPLAY
    # -----------------------------------------------------

    st.subheader("💡 Recommended Retention Action")

    # Display the recommended action
    st.info(recommendation)


    # -----------------------------------------------------
    # PROBABILITY PROGRESS BAR
    # -----------------------------------------------------

    st.subheader("Churn Probability")

    # Display probability visually
    st.progress(
        min(churn_probability, 1.0)
    )