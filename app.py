import streamlit as st
import numpy as np
import pickle

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    color: white;
}

h1,h2,h3,label {
    color: #f0f0f0 !important;
}

input {
    background-color: #1f2c34 !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid #555 !important;
}

div[data-baseweb="select"] {
    background-color: #1f2c34 !important;
    border-radius: 8px !important;
}

.stButton>button {
    background-color: #00c9a7;
    color: black;
    border-radius: 10px;
    height: 45px;
    width: 200px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #00e6c3;
}

.result-box {
    padding:20px;
    border-radius:10px;
    font-size:20px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
model = pickle.load(open("logistic_model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ---------------------------
# Title
# ---------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Predict whether a telecom customer is likely to churn.")

# ---------------------------
# Input Layout
# ---------------------------
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

    tenure = float(st.text_input("Tenure (months)", "12"))
    monthly = float(st.text_input("Monthly Charges", "70"))
    total = float(st.text_input("Total Charges", "1500"))

# ---------------------------
# Encoding
# ---------------------------
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

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Churn"):

    features = np.array([[gender,senior,partner,dependents,
                          tenure,phone,multiline,internet,
                          security,backup,device,support,
                          tv,movies,contract,paperless,
                          payment,monthly,total]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.markdown(
        f"<div class='result-box' style='background:#ff4b4b'>⚠️ Customer likely to churn<br><br>Risk: {round(probability*100,2)}%</div>",
        unsafe_allow_html=True)

    else:
        st.markdown(
        f"<div class='result-box' style='background:#00c897'>✅ Customer unlikely to churn<br><br>Risk: {round(probability*100,2)}%</div>",
        unsafe_allow_html=True)
