import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    text-align: center;
    margin-bottom: 4px;
}
.hero-sub {
    text-align: center;
    color: #8b8fa8;
    font-size: 1rem;
    margin-bottom: 24px;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5b5fef;
    margin: 20px 0 8px 0;
}
.result-box {
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 20px;
}
.result-fraud {
    background: linear-gradient(135deg, #3b0a0a, #1f0606);
    border: 1.5px solid #ff4444;
}
.result-suspicious {
    background: linear-gradient(135deg, #2b1a00, #1a1000);
    border: 1.5px solid #ff9800;
}
.result-safe {
    background: linear-gradient(135deg, #052015, #030d0d);
    border: 1.5px solid #00c853;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 6px;
}
.fraud-color     { color: #ff4444; }
.suspicious-color { color: #ff9800; }
.safe-color      { color: #00c853; }
.result-sub      { color: #8b8fa8; font-size: 0.9rem; }
.divider {
    border: none;
    border-top: 1px solid #1e2030;
    margin: 24px 0;
}
.info-box {
    background: #0e1117;
    border: 1px solid #1e2030;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #8b8fa8;
    margin-bottom: 16px;
}
.stButton > button {
    background: linear-gradient(135deg, #5b5fef, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    padding: 14px 0;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        model    = joblib.load("models/gradient_boosting_fraud_model.pkl")
        features = joblib.load("models/feature_list.pkl")
        return model, features
    except FileNotFoundError:
        return None, None

model, feature_list = load_model()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown('<div class="hero-title">🔒 FRAUD DETECTION SYSTEM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Group 4 · Gradient Boosting · Digital Banking Fraud Detection</div>',
    unsafe_allow_html=True
)

if model is None:
    st.error("⚠️ Model not found. Make sure `models/gradient_boosting_fraud_model.pkl` and `models/feature_list.pkl` exist.")
    st.stop()

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1 — CORE TRANSACTION INFO
# Only pre-transaction columns (no newbalance columns)
# ─────────────────────────────────────────────

st.markdown('<div class="section-label">Core Transaction Info</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-box">💡 Only pre-transaction values are used. '
    'Post-transaction balances (newbalanceOrig, newbalanceDest) are excluded '
    'to reflect a real-time fraud decision.</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    tx_type = st.selectbox(
        "Transaction Type",
        ["CASH_OUT", "TRANSFER"],
        help="Only CASH_OUT and TRANSFER are relevant — fraud only occurs in these types"
    )

with col2:
    amount = st.number_input(
        "Transaction Amount (₦)",
        min_value=0.0,
        value=50000.0,
        step=100.0,
        format="%.2f"
    )

with col3:
    step = st.number_input(
        "Step (Simulated Hour)",
        min_value=0,
        value=150,
        step=1,
        help="1 step = 1 simulated hour in the dataset"
    )

col4, col5 = st.columns(2)

with col4:
    oldbalanceOrg = st.number_input(
        "Sender Balance (Before Transaction)",
        min_value=0.0,
        value=100000.0,
        step=100.0,
        format="%.2f",
        help="How much the sender had BEFORE this transaction"
    )

with col5:
    oldbalanceDest = st.number_input(
        "Receiver Balance (Before Transaction)",
        min_value=0.0,
        value=0.0,
        step=100.0,
        format="%.2f",
        help="How much the receiver had BEFORE this transaction"
    )


# ─────────────────────────────────────────────
# SECTION 2 — SENDER HISTORY
# Used to compute aggregation features
# ─────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Sender Historical Profile</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-box">💡 These values come from the sender\'s transaction history. '
    'If unknown, leave as defaults — the model will still work.</div>',
    unsafe_allow_html=True
)

col6, col7 = st.columns(2)

with col6:
    sender_avg_amount = st.number_input(
        "Sender's Average Transaction Amount (₦)",
        min_value=0.0,
        value=40000.0,
        step=100.0,
        format="%.2f",
        help="Average amount this sender typically sends"
    )

with col7:
    sender_total_sent = st.number_input(
        "Sender's Total Amount Ever Sent (₦)",
        min_value=0.0,
        value=200000.0,
        step=100.0,
        format="%.2f",
        help="Cumulative total sent by this sender historically"
    )


# ─────────────────────────────────────────────
# AUTO-COMPUTE ALL 10 FEATURES
# Matches exactly what the model was trained on
# ─────────────────────────────────────────────

def compute_features():
    # --- Group 1: Core ---
    type_encoded = 0 if tx_type == "CASH_OUT" else 1

    # --- Group 2: Derived (pre-transaction only) ---
    # Cap amount_ratio at 10 to match training preprocessing
    amount_ratio   = min(amount / (oldbalanceOrg + 1), 10)
    dest_was_empty = 1 if oldbalanceDest == 0 else 0
    hour_of_day    = step % 24

    # --- Group 3: Aggregation ---
    # sender_avg_amount comes directly from user input above

    # --- Group 4: Unique Features ---
    # 🌟 Drain Score — how much of sender's balance was drained
    drain_score = amount / (oldbalanceOrg + amount + 1)

    # 🌟 Both Accounts Suspicious — sender nearly empty + receiver was empty
    both_accounts_suspicious = 1 if (
        oldbalanceOrg <= amount and oldbalanceDest == 0
    ) else 0

    # Build DataFrame in EXACT order model was trained on
    data = {
        # Core (3)
        "type_encoded":             [type_encoded],
        "amount":                   [amount],
        "oldbalanceOrg":            [oldbalanceOrg],

        # Derived (3)
        "amount_ratio":             [amount_ratio],
        "dest_was_empty":           [dest_was_empty],
        "hour_of_day":              [hour_of_day],

        # Aggregation (1)
        "sender_avg_amount":        [sender_avg_amount],

        # Core balance (1)
        "oldbalanceDest":           [oldbalanceDest],

        # Unique (2)
        "drain_score":              [drain_score],
        "both_accounts_suspicious": [both_accounts_suspicious],
    }

    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# FEATURE PREVIEW
# ─────────────────────────────────────────────

with st.expander("🔍 Preview Auto-Computed Features"):
    df_preview = compute_features()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Derived Features**")
        # ✅ FIX: Build list safely — no duplicates possible
        derived_cols = ["amount_ratio", "dest_was_empty", "hour_of_day"]
        st.dataframe(
            df_preview[derived_cols],
            width='stretch'
        )
    with col_b:
        st.markdown("**Unique Features**")
        st.dataframe(
            df_preview[["drain_score", "both_accounts_suspicious"]],
            width='stretch'
        )

    st.markdown("**Full Feature Vector (sent to model)**")
    st.dataframe(df_preview[feature_list], width='stretch')


# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)

if st.button("🔍 Analyse Transaction for Fraud"):

    input_df = compute_features()

    # Ensure columns are in correct order
    input_df = input_df[feature_list]

    with st.spinner("Running model..."):
        prediction  = model.predict(input_df)[0]
        proba       = model.predict_proba(input_df)[0]
        fraud_prob  = proba[1]
        safe_prob   = proba[0]

    # ── 3-Level Risk Classification ──
    if fraud_prob >= 0.55:
        risk_level = "FRAUD"
    elif fraud_prob >= 0.15:
        risk_level = "SUSPICIOUS"
    else:
        risk_level = "LEGIT"

    # ── Result Card ──
    if risk_level == "FRAUD":
        st.markdown(f"""
        <div class="result-box result-fraud">
            <div class="result-title fraud-color">🚨 Fraud Alert</div>
            <div class="result-sub">
                This transaction has been flagged as <strong>fraudulent</strong>.
                Recommend <strong>blocking</strong> and escalating for manual review.
                Fraud probability: <strong>{fraud_prob*100:.1f}%</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    elif risk_level == "SUSPICIOUS":
        st.markdown(f"""
        <div class="result-box result-suspicious">
            <div class="result-title suspicious-color">⚠️ Suspicious Transaction</div>
            <div class="result-sub">
                This transaction shows <strong>suspicious patterns</strong>.
                Recommend flagging for <strong>analyst review</strong> before processing.
                Fraud probability: <strong>{fraud_prob*100:.1f}%</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-box result-safe">
            <div class="result-title safe-color">✅ Transaction Looks Safe</div>
            <div class="result-sub">
                No fraud indicators detected. This transaction appears <strong>legitimate</strong>.
                Fraud probability: <strong>{fraud_prob*100:.1f}%</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Confidence Breakdown ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Confidence Breakdown</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.caption("✅ Legitimate probability")
        st.progress(float(safe_prob))
        st.write(f"**{safe_prob * 100:.1f}%**")
    with c2:
        st.caption("🚨 Fraud probability")
        st.progress(float(fraud_prob))
        st.write(f"**{fraud_prob * 100:.1f}%**")

    # ── What Triggered The Flag ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">What The Model Saw</div>', unsafe_allow_html=True)

    drain_score_val            = input_df['drain_score'].values[0]
    both_suspicious_val        = input_df['both_accounts_suspicious'].values[0]
    dest_empty_val             = input_df['dest_was_empty'].values[0]
    amount_ratio_val           = input_df['amount_ratio'].values[0]

    flags = []
    if drain_score_val > 0.8:
        flags.append(f"🔴 **Drain Score {drain_score_val:.2f}** — sender's account nearly emptied in one transaction")
    if both_suspicious_val == 1:
        flags.append("🔴 **Both Accounts Suspicious** — sender had little left AND receiver was empty")
    if dest_empty_val == 1:
        flags.append("🟡 **Receiver Was Empty** — destination account had no prior balance")
    if amount_ratio_val >= 5:
        flags.append(f"🟡 **High Amount Ratio {amount_ratio_val:.1f}** — transaction is large relative to sender balance")

    if flags:
        for flag in flags:
            st.markdown(f"- {flag}")
    else:
        st.markdown("- 🟢 No major fraud indicators triggered")


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.caption("Group 4 · Digital Banking Fraud Detection System · Gradient Boosting · Built with Streamlit")
