"""
=============================================================
  Customer Churn Prediction Dashboard
  Built with Streamlit + Scikit-learn (Logistic Regression)
=============================================================
"""

import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  – clean, professional look
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: #f7f9fc;
}

/* ── Hide default Streamlit hamburger / footer ── */
#MainMenu, footer { visibility: hidden; }

/* ── Section card ── */
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-bottom: 1.25rem;
}

/* ── Dashboard title ── */
.dash-title {
    font-size: 2rem;
    font-weight: 600;
    color: #0f172a;
    margin: 0;
    letter-spacing: -0.5px;
}

.dash-subtitle {
    font-size: 0.95rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* ── Section header ── */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #1e293b;
    border-left: 4px solid #6366f1;
    padding-left: 0.65rem;
    margin-bottom: 1rem;
}

/* ── Churn result cards ── */
.result-churn {
    background: #fff1f2;
    border: 1.5px solid #fca5a5;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}

.result-safe {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}

.result-title {
    font-size: 1.35rem;
    font-weight: 600;
    margin: 0;
}

.result-subtitle {
    font-size: 0.88rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* ── Sidebar styling ── */
[data-testid="stSidebar"] {
    background: #1e293b !important;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
}

/* ── Predict button ── */
div.stButton > button {
    background: #6366f1;
    color: #ffffff;
    font-size: 1rem;
    font-weight: 500;
    padding: 0.65rem 2.2rem;
    border: none;
    border-radius: 8px;
    width: 100%;
    transition: background 0.2s;
}
div.stButton > button:hover {
    background: #4f46e5;
    color: #ffffff;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.82rem;
    color: #94a3b8;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #e2e8f0;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL & SCALER
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the saved model and scaler from disk (cached so it only runs once)."""
    model  = pickle.load(open("logistic_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Predictor")
    st.markdown("---")

    st.markdown("### 📋 About This App")
    st.markdown(
        "This dashboard uses a **Logistic Regression** model trained on the "
        "**Telco Customer Churn** dataset to predict whether a customer is "
        "likely to leave the service."
    )

    st.markdown("### 🧠 Model Details")
    st.markdown(
        "- **Algorithm**: Logistic Regression  \n"
        "- **Scaler**: StandardScaler  \n"
        "- **Dataset**: Telco Customer Churn  \n"
        "- **Target**: Churn (0 = No, 1 = Yes)"
    )

    st.markdown("### 📌 How to Use")
    st.markdown(
        "1. Fill in the customer details on the right  \n"
        "2. Click **Predict Churn**  \n"
        "3. View the prediction & probability  \n"
    )

    st.markdown("---")
    if model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model files not found. Place `logistic_model.pkl` and `scaler.pkl` in the project root.")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown(
    '<p class="dash-title">📡 Customer Churn Prediction Dashboard</p>'
    '<p class="dash-subtitle">Enter customer details below and click <strong>Predict Churn</strong> to get a real-time prediction.</p>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  INPUT SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-header">👤 Customer Information</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "📅 Tenure (months)",
        min_value=0, max_value=72, value=12, step=1,
        help="How many months the customer has been with the company."
    )
    senior_citizen = st.selectbox(
        "🧓 Senior Citizen",
        options=["No", "Yes"],
        help="Is the customer 65 years or older?"
    )
    partner = st.selectbox(
        "💑 Partner",
        options=["No", "Yes"],
        help="Does the customer have a partner?"
    )

with col2:
    monthly_charges = st.number_input(
        "💵 Monthly Charges ($)",
        min_value=0.0, max_value=200.0, value=65.0, step=0.5,
        help="The amount charged to the customer monthly."
    )
    dependents = st.selectbox(
        "👨‍👩‍👧 Dependents",
        options=["No", "Yes"],
        help="Does the customer have dependents?"
    )
    internet_service = st.selectbox(
        "🌐 Internet Service",
        options=["DSL", "Fiber optic", "No"],
        help="Type of internet service the customer uses."
    )

with col3:
    total_charges = st.number_input(
        "💳 Total Charges ($)",
        min_value=0.0, max_value=10000.0, value=780.0, step=1.0,
        help="Total amount charged to the customer over the tenure."
    )
    contract = st.selectbox(
        "📄 Contract Type",
        options=["Month-to-month", "One year", "Two year"],
        help="The contract term of the customer."
    )
    payment_method = st.selectbox(
        "💳 Payment Method",
        options=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        help="How the customer pays their bill."
    )

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ENCODE INPUTS  →  NumPy array
# ─────────────────────────────────────────────
def encode_inputs(
    tenure, monthly_charges, total_charges,
    contract, internet_service, payment_method,
    senior_citizen, partner, dependents
):
    """
    Convert raw UI inputs into the numeric feature vector
    expected by the trained model.

    Encoding mirrors what was done during training:
      - Binary Yes/No  →  1 / 0
      - Contract       →  Month-to-month=0, One year=1, Two year=2
      - Internet       →  DSL=0, Fiber optic=1, No=2
      - Payment        →  Electronic check=0, Mailed check=1,
                          Bank transfer=2, Credit card=3
    """
    senior_enc   = 1 if senior_citizen == "Yes" else 0
    partner_enc  = 1 if partner        == "Yes" else 0
    depend_enc   = 1 if dependents     == "Yes" else 0

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    payment_map  = {
        "Electronic check":          0,
        "Mailed check":              1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)":   3,
    }

    features = np.array([[
        tenure,
        monthly_charges,
        total_charges,
        contract_map[contract],
        internet_map[internet_service],
        payment_map[payment_method],
        senior_enc,
        partner_enc,
        depend_enc,
    ]])
    return features


# ─────────────────────────────────────────────
#  PREDICT BUTTON & RESULTS
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Churn", use_container_width=True)

if predict_clicked:
    if not model_loaded:
        st.error("❌ Cannot predict – model files are missing. See the sidebar for instructions.")
    else:
        # Show a spinner while "processing"
        with st.spinner("Analysing customer data…"):
            time.sleep(1.2)  # brief pause for UX feel

            # Build & scale the feature vector
            X_raw    = encode_inputs(
                tenure, monthly_charges, total_charges,
                contract, internet_service, payment_method,
                senior_citizen, partner, dependents
            )
            X_scaled = scaler.transform(X_raw)

            # Run inference
            prediction  = model.predict(X_scaled)[0]          # 0 or 1
            proba       = model.predict_proba(X_scaled)[0]     # [prob_0, prob_1]
            churn_proba = proba[1]                             # probability of churn

        # ── Results layout ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">📊 Prediction Results</p>', unsafe_allow_html=True)

        res_col, metric_col = st.columns([1, 1], gap="large")

        with res_col:
            if prediction == 1:
                st.markdown(
                    '<div class="result-churn">'
                    '<p class="result-title" style="color:#dc2626;">⚠️ Customer is likely to churn</p>'
                    '<p class="result-subtitle">Action recommended – consider a retention offer.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.warning("⚠️ High churn risk detected. Review the customer's contract and engagement.")
            else:
                st.markdown(
                    '<div class="result-safe">'
                    '<p class="result-title" style="color:#16a34a;">✅ Customer is not likely to churn</p>'
                    '<p class="result-subtitle">This customer appears satisfied and engaged.</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.success("✅ Low churn risk. No immediate action required.")

        with metric_col:
            # Metric cards for probabilities
            m1, m2 = st.columns(2)
            m1.metric(
                label="🔴 Churn Probability",
                value=f"{churn_proba * 100:.1f}%",
                delta=f"{(churn_proba - 0.5) * 100:+.1f}% vs baseline",
                delta_color="inverse",
            )
            m2.metric(
                label="🟢 Retention Probability",
                value=f"{(1 - churn_proba) * 100:.1f}%",
            )

            # Progress bar for churn probability
            st.markdown("**Churn Risk Level**")
            st.progress(float(churn_proba))

            # Colour-coded risk label
            if churn_proba >= 0.70:
                st.error(f"🔴 High Risk  ({churn_proba * 100:.1f}%)")
            elif churn_proba >= 0.40:
                st.warning(f"🟡 Medium Risk  ({churn_proba * 100:.1f}%)")
            else:
                st.success(f"🟢 Low Risk  ({churn_proba * 100:.1f}%)")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Feature summary table ──
        with st.expander("📋 View submitted customer profile"):
            profile = pd.DataFrame({
                "Feature":  [
                    "Tenure (months)", "Monthly Charges ($)", "Total Charges ($)",
                    "Contract Type", "Internet Service", "Payment Method",
                    "Senior Citizen", "Partner", "Dependents",
                ],
                "Value": [
                    tenure, monthly_charges, total_charges,
                    contract, internet_service, payment_method,
                    senior_citizen, partner, dependents,
                ],
            })
            st.dataframe(profile, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(
    '<div class="footer">Machine Learning Churn Prediction App built with Streamlit</div>',
    unsafe_allow_html=True,
)
