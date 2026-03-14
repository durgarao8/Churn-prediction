import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Predict whether a customer will churn using Machine Learning.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About This App")
st.sidebar.write("""
This application predicts **customer churn** using a trained **Logistic Regression model**.

Steps:
1. Enter customer details
2. Click Predict
3. View churn probability
""")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    tenure = st.number_input("Tenure (months)", 0, 72)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0)

with col3:
    contract = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])

    internet = st.selectbox("Internet Service",
                            ["DSL", "Fiber optic", "No"])

    payment = st.selectbox("Payment Method",
                           ["Electronic check",
                            "Mailed check",
                            "Bank transfer",
                            "Credit card"])

# -----------------------------
# Encode Inputs
# -----------------------------
gender = 1 if gender == "Male" else 0
senior = 1 if senior == "Yes" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

contract = contract_map[contract]
internet = internet_map[internet]
payment = payment_map[payment]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    features = np.array([[gender, senior, partner, dependents,
                          tenure, monthly_charges, total_charges,
                          contract, internet, payment]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.subheader("Churn Probability")

    prob_percent = int(probability * 100)

    st.metric("Churn Risk", f"{prob_percent}%")

    st.progress(prob_percent)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Machine Learning Churn Prediction App built with Streamlit")
