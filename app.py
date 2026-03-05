import streamlit as st
import pandas as pd 
import joblib 

# ─────────────────────────────────────────────

# PAGE CONFIG

# ─────────────────────────────────────────────

st.set_page_config(
page_title= "Fraud Detection System",
layout= "wide",
)
# ─────────────────────────────────────────────

# CUSTOM CSS

# ─────────────────────────────────────────────

st.markdown("""

<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.hero-title {
font-family: 'Syne', sans-serif;
font-size: 2.6rem; font-weight: 800;
color: #ffffff; letter-spacing: -1px;
}
.hero-sub { color: #8b8fa8; font-size: 1rem; margin-top: 4px; }
.section-label {
font-family: 'Syne', sans-serif;
font-size: 0.72rem; font-weight: 700;
letter-spacing: 2px; text-transform: uppercase;
color: #5b5fef; margin: 18px 0 8px 0;
}
.result-box { border-radius: 14px; padding: 28px 32px; margin-top: 20px; }
.result-fraud { background: linear-gradient(135deg,#3b0a0a,#1f0606); border: 1.5px solid #ff4444; }
.result-safe { background: linear-gradient(135deg,#052015,#030d0d); border: 1.5px solid #00c853; }
.result-title { font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:800; margin-bottom:6px; }
.fraud-color { color: #ff4444; }
.safe-color { color: #00c853; }
.result-sub { color: #8b8fa8; font-size: 0.9rem; }
.divider { border:none; border-top:1px solid #1e2030; margin: 22px 0; }
.stButton > button {
background: linear-gradient(135deg,#5b5fef,#7c3aed);
color:white; border:none; border-radius:10px;
font-family:'Syne',sans-serif; font-weight:700;
font-size:1rem; padding:14px 0; width:100%;
}
.stButton > button:hover { opacity: 0.88; }
.derived-note { font-size:0.8rem; color:#5b5fef; font-style:italic; margin-top:-10px; margin-bottom:10px; }
</style>

""", unsafe_allow_html= True)



# LOAD MODEL

def load_model():
    try:
        return joblib.load("models/random_forest_model_realistic.pkl")
    except FileNotFoundError:
        return None

model = load_model()

st.markdown("""
<style>
  .head-title{
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}
.hero-sub{
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# HEADER

st.title("FRAUD DETECTION SYSTEM")

st.markdown(
    '<div class="hero-sub">Enter transaction details - derived features are computed automatically.</div>',
    unsafe_allow_html=True
)
if model is None:
   st.warning("⚠️ Model not found. Make sure `models/random_forest_model_realistic.pkl` exists in your project folder.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# SECTION 1 — CORE TRANSACTION INPUTS (pre-transaction, raw values)

st.markdown('<div class="section-label"> Core Transaction Info</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    tx_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
    amount = st.number_input("Transaction Amount", min_value=0.0, value=5000.0, step=0.01, format="%.2f")

with col2:
    oldbalanceOrg = st.number_input("Old Balance — Origin", min_value=0.0, value=10000.0, step=0.01, format="%.2f")
    newbalanceOrig = st.number_input("New Balance — Origin", min_value=0.0, value=5000.0, step=0.01, format="%.2f")

with col3:
    oldbalanceDest = st.number_input("Old Balance — Destination", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    newbalanceDest = st.number_input("New Balance — Destination", min_value=0.0, value=5000.0, step=0.01, format="%.2f")



# SECTION 2 — TIME FEATURES


st.markdown('<div class="section-label"> Time of Transaction</div>', unsafe_allow_html=True)

col4, col5 = st.columns(2)
with col4:
    hour_of_day = st.slider("Hour of Day (0 = midnight)", min_value=0, max_value=23, value=14)
with col5:
    day_of_week = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])


# SECTION 3 — SENDER BEHAVIOURAL / AGGREGATION FEATURES

# (These come from historical data — enter known values or leave defaults)


st.markdown('<div class="section-label"> Sender Behavioural Profile</div>', unsafe_allow_html=True)
st.markdown('<div class="derived-note">💡 These are historical aggregates. If unknown, leave as defaults.</div>', unsafe_allow_html=True)

col6, col7, col8 = st.columns(3)
with col6:
    sender_risk_score = st.number_input("Sender Risk Score (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
    large_tx_ratio = st.number_input("Large Tx Ratio (0-1)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
    sender_low_balance_flag = st.selectbox("Sender Low Balance Flag", [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")

with col7:
    tx_count_by_sender = st.number_input("Tx Count by Sender", min_value=0, value=5, step=1)
    total_sent_by_sender = st.number_input("Total Sent by Sender", min_value=0.0, value=20000.0, step=0.01, format="%.2f")
    sender_avg_amount = st.number_input("Sender Avg Tx Amount", min_value=0.0, value=4000.0, step=0.01, format="%.2f")

with col8:
    sender_std_amount = st.number_input("Sender Std Dev Amount", min_value=0.0, value=1000.0, step=0.01, format="%.2f")
    time_since_last_tx = st.number_input("Time Since Last Tx (hours)", min_value=0.0, value=2.0, step=0.1, format="%.1f")
    rapid_fire_flag = st.selectbox("Rapid Fire Flag (multiple fast txns)", [0, 1], format_func=lambda x: "Yes (1)" if x else "No (0)")


# SECTION 4 — RECEIVER BEHAVIOURAL FEATURES

st.markdown('<div class="section-label"> Receiver Behavioural Profile</div>', unsafe_allow_html=True)

col9, col10 = st.columns(2)
with col9:
    recv_tx_count = st.number_input("Receiver Tx Count Received", min_value=0, value=3, step=1)
with col10:
    recv_total_received = st.number_input("Receiver Total Amount Received", min_value=0.0, value=8000.0, step=0.01, format="%.2f")


# AUTO-COMPUTE DERIVED FEATURES

def compute_features():
    # Encode transaction type
    type_map = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
    type_encoded = type_map[tx_type]

    # Derived behavioral flags
    dest_was_empty = 1 if oldbalanceDest == 0 else 0
    amount_ratio = amount / (oldbalanceOrg + 1) # avoid divide-by-zero
    is_large_amount = 1 if amount > 200_000 else 0
    is_round_amount = 1 if (amount % 1000 == 0 and amount > 0) else 0
    is_night = 1 if (hour_of_day >= 22 or hour_of_day <= 5) else 0
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0

    exact_round_amount_flag = 1 if (amount % 1000 == 0 and amount > 0) else 0
    mule_account_score = round( 
        (dest_was_empty * 0.4) +
        (rapid_fire_flag * 0.3) +
        (is_large_amount * 0.2) +
        (sender_low_balance_flag * 0.1), 4
    )
    amount_deviation = abs(amount - sender_avg_amount) / (sender_std_amount + 1)

    # Build DataFrame in the EXACT order the model was trained on
    data = {
        # Core
        "type_encoded": [type_encoded],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceOrg],
        "oldbalanceDest": [oldbalanceDest],
        # Standard derived
        "dest_was_empty": [dest_was_empty],
        "amount_ratio": [amount_ratio],
        "is_large_amount": [is_large_amount],
        "is_round_amount": [is_round_amount],
        "hour_of_day": [hour_of_day],
        "is_night": [is_night],
        "is_weekend": [is_weekend],
        # 5 unique features
        "sender_risk_score": [sender_risk_score],
        "large_tx_ratio": [large_tx_ratio],
        "sender_low_balance_flag":[sender_low_balance_flag],
        "exact_round_amount_flag":[exact_round_amount_flag],
        "mule_account_score": [mule_account_score],
        # Aggregation / behavioral profiling
        "tx_count_by_sender": [tx_count_by_sender],
        "total_sent_by_sender": [total_sent_by_sender],
        "sender_avg_amount": [sender_avg_amount],
        "sender_std_amount": [sender_std_amount],
        "amount_deviation": [amount_deviation],
        "recv_tx_count": [recv_tx_count],
        "recv_total_received": [recv_total_received],
        "time_since_last_tx": [time_since_last_tx],
        "rapid_fire_flag": [rapid_fire_flag],
    }
    return pd.DataFrame(data)


# Show auto-computed values in an expander

with st.expander("Preview Auto-Computed Derived Features"):
    df_preview = compute_features()
    derived_cols = ["dest_was_empty","amount_ratio","is_large_amount","is_round_amount",
                     "is_night","is_weekend","exact_round_amount_flag","mule_account_score","amount_deviation"]
    st.dataframe(df_preview[derived_cols], use_container_width=True)


# PREDICT BUTTON

st.markdown('<hr class="divider">', unsafe_allow_html=True)

if st.button("Analyse Transaction for Fraud"):

    if model is None:
        st.error("Cannot run — model file not loaded.")
    else:
        input_df = compute_features()

        with st.spinner("Running model..."):
            prediction = model.predict(input_df)[0]

            if hasattr(model, "predict_proba"):
               proba = model.predict_proba(input_df)[0]
               fraud_prob = proba[1]
               safe_prob = proba[0]
            else:
               score = float(model.predict(input_df)[0])
               fraud_prob = float(np.clip(score, 0, 1))
               safe_prob = 1 - fraud_prob

        # Result card
        if prediction == 1:
           st.markdown(f"""
           <div class="result-box result-fraud">
                <div class="result-title fraud-color"> Fraud Detected</div>
                <div class="result-sub">This transaction has been flagged as <strong>potentially fraudulent</strong>.
                Consider blocking or escalating for manual review.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-safe">
                <div class="result-title safe-color">Transaction Looks Safe</div>
                <div class="result-sub">No fraud indicators detected. This transaction appears <strong>legitimate</strong>.</div>
            </div>""", unsafe_allow_html=True)

        # Probability bars
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Confidence Breakdown</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Legitimate probability")
            st.progress(float(safe_prob))
            st.write(f"**{safe_prob * 100:.1f}%**")
        with c2:
            st.caption("Fraud probability")
            st.progress(float(fraud_prob))
            st.write(f"**{fraud_prob * 100:.1f}%**")

        # Full feature table
        with st.expander("📋 Full feature vector sent to model"):
            st.dataframe(input_df, use_container_width=True)

# FOOTER


st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.caption("Group 4 Fraud Detection System · Random Forest · Built with Streamlit")