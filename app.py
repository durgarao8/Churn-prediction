import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Page Configuration
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
st.markdown("Predict whether a customer is likely to **churn or stay** using Machine Learning.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("ℹ️ About Project")

st.sidebar.write("""
This application predicts **customer churn** using a Machine Learning model.

Model Used:
- Logistic Regression

Dataset:
- Telco Customer Churn Dataset

Steps:
1. Enter customer details
2. Click Predict
3. View churn probability
""")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Enter Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

with col2:
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])

with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# -----------------------------
# Encoding Inputs
# -----------------------------
senior = 1 if senior == "Yes" else 0

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

contract = contract_map[contract]
internet = internet_map[internet]

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Churn"):

    with st.spinner("Analyzing customer data..."):

        features = np.array([[tenure,
                              monthly_charges,
                              total_charges,
                              senior,
                              contract,
                              internet]])

        # scale features
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    # -----------------------------
    # Probability Section
    # -----------------------------
    st.subheader("Churn Probability")

    prob_percent = int(probability * 100)

    st.metric(label="Churn Risk", value=f"{prob_percent}%")

    st.progress(prob_percent)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Machine Learning Churn Prediction App built with Streamlit"
)
