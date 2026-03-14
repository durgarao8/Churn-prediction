import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Load model
model = pickle.load(open("logistic_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("Customer Churn Prediction")

st.write("Enter customer details")

col1,col2,col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender",["Male","Female"])
    senior = st.selectbox("Senior Citizen",["No","Yes"])
    partner = st.selectbox("Partner",["No","Yes"])
    dependents = st.selectbox("Dependents",["No","Yes"])
    phone = st.selectbox("Phone Service",["No","Yes"])
    multiline = st.selectbox("Multiple Lines",["No","Yes"])

with col2:
    internet = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
    security = st.selectbox("Online Security",["No","Yes"])
    backup = st.selectbox("Online Backup",["No","Yes"])
    device = st.selectbox("Device Protection",["No","Yes"])
    support = st.selectbox("Tech Support",["No","Yes"])
    tv = st.selectbox("Streaming TV",["No","Yes"])
    movies = st.selectbox("Streaming Movies",["No","Yes"])

with col3:
    contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
    paperless = st.selectbox("Paperless Billing",["No","Yes"])
    payment = st.selectbox("Payment Method",
        ["Electronic check","Mailed check","Bank transfer","Credit card"])
    tenure = st.number_input("Tenure",0,72)
    monthly = st.number_input("Monthly Charges")
    total = st.number_input("Total Charges")

# Binary encoding
def encode(x):
    return 1 if x=="Yes" else 0

gender = 1 if gender=="Male" else 0
senior = encode(senior)
partner = encode(partner)
dependents = encode(dependents)
phone = encode(phone)
multiline = encode(multiline)
security = encode(security)
backup = encode(backup)
device = encode(device)
support = encode(support)
tv = encode(tv)
movies = encode(movies)
paperless = encode(paperless)

contract_map = {"Month-to-month":0,"One year":1,"Two year":2}
internet_map = {"DSL":0,"Fiber optic":1,"No":2}
payment_map = {"Electronic check":0,"Mailed check":1,"Bank transfer":2,"Credit card":3}

contract = contract_map[contract]
internet = internet_map[internet]
payment = payment_map[payment]

if st.button("Predict"):

    features = np.array([[gender,senior,partner,dependents,
                          tenure,phone,multiline,internet,
                          security,backup,device,support,
                          tv,movies,contract,paperless,
                          payment,monthly,total]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")

    st.write("Churn Probability:", round(probability*100,2),"%")
