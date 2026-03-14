import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn.")

# Input fields
tenure = st.number_input("Tenure (months)")
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

# Predict button
if st.button("Predict"):

    # Convert inputs to array
    data = np.array([[tenure, monthly_charges, total_charges]])

    # Apply scaling
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.write("Customer is likely to churn")
    else:
        st.write("Customer is not likely to churn")
