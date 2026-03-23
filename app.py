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
.fraud-color      { color: #ff4444; }
.suspicious-color { color: #ff9800; }
.safe-color       { color: #00c853; }
.result-sub       { color: #8b8fa8; font-size: 0.9rem; }
.score-box {
    background: #0e1117;
    border: 1px solid #1e2030;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
}
.score-label {
    font-size: 0.75rem;
    color: #8b8fa8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.score-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
}
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
# LOAD MODELS
# ─────────────────────────────────────────────

@st.cache_resource
def load_models():
    try:
        gb_model   = joblib.load("models/gradient_boosting_fraud_model.pkl")
        iso_model  = joblib.load("models/isolation_forest_model.pkl")
        iso_scaler = joblib.load("models/iso_scaler.pkl")
        features   = joblib.load("models/feature_list.pkl")
        weights    = joblib.load("models/model_weights.pkl")
        return gb_model, iso_model, iso_scaler, features, weights
    except FileNotFoundError as e:
        return None, None, None, None, None

gb_model, iso_model, iso_scaler, feature_list, weights = load_models()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown('<div class="hero-title">🔒 FRAUD DETECTION SYSTEM</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Group 4 · Gradient Boosting + Isolation Forest · Digital Banking Fraud Detection</div>',
    unsafe_allow_html=True
)

if gb_model is None:
    st.error(
        "⚠️ Models not found. Make sure all 5 model files exist in the `models/` folder:\n\n"
        "- `gradient_boosting_fraud_model.pkl`\n"
        "- `isolation_forest_model.pkl`\n"
        "- `iso_scaler.pkl`\n"
        "- `feature_list.pkl`\n"
        "- `model_weights.pkl`"
    )
    st.stop()

# Show model weights in use
GB_WEIGHT = weights['gb']
IF_WEIGHT = weights['if']

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1 — CORE TRANSACTION INFO
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
        min_value=0.0, value=50000.0, step=100.0, format="%.2f"
    )
with col3:
    step = st.number_input(
        "Step (Simulated Hour)",
        min_value=0, value=150, step=1,
        help="1 step = 1 simulated hour in the dataset"
    )

col4, col5 = st.columns(2)

with col4:
    oldbalanceOrg = st.number_input(
        "Sender Balance (Before Transaction)",
        min_value=0.0, value=100000.0, step=100.0, format="%.2f",
        help="How much the sender had BEFORE this transaction"
    )
with col5:
    oldbalanceDest = st.number_input(
        "Receiver Balance (Before Transaction)",
        min_value=0.0, value=0.0, step=100.0, format="%.2f",
        help="How much the receiver had BEFORE this transaction"
    )


# ─────────────────────────────────────────────
# SECTION 2 — SENDER HISTORY
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
        min_value=0.0, value=40000.0, step=100.0, format="%.2f",
        help="Average amount this sender typically sends"
    )
with col7:
    sender_total_sent = st.number_input(
        "Sender's Total Amount Ever Sent (₦)",
        min_value=0.0, value=200000.0, step=100.0, format="%.2f",
        help="Cumulative total sent by this sender historically"
    )


# ─────────────────────────────────────────────
# AUTO-COMPUTE ALL 10 FEATURES
# ─────────────────────────────────────────────

def compute_features():
    type_encoded             = 0 if tx_type == "CASH_OUT" else 1
    amount_ratio             = min(amount / (oldbalanceOrg + 1), 10)
    dest_was_empty           = 1 if oldbalanceDest == 0 else 0
    hour_of_day              = step % 24
    drain_score              = amount / (oldbalanceOrg + amount + 1)
    both_accounts_suspicious = 1 if (oldbalanceOrg <= amount and oldbalanceDest == 0) else 0

    data = {
        "type_encoded":              [type_encoded],
        "amount":                    [amount],
        "oldbalanceOrg":             [oldbalanceOrg],
        "amount_ratio":              [amount_ratio],
        "dest_was_empty":            [dest_was_empty],
        "hour_of_day":               [hour_of_day],
        "sender_avg_amount":         [sender_avg_amount],
        "oldbalanceDest":            [oldbalanceDest],
        "drain_score":               [drain_score],
        "both_accounts_suspicious":  [both_accounts_suspicious],
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
        st.dataframe(
            df_preview[["amount_ratio", "dest_was_empty", "hour_of_day"]],
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

    input_df = compute_features()[feature_list]

    with st.spinner("Running models..."):

        # ── Gradient Boosting score ──
        gb_proba   = gb_model.predict_proba(input_df)[0][1]

        # ── Isolation Forest score ──
        iso_raw    = iso_model.decision_function(input_df)
        iso_score  = float(iso_scaler.transform(-iso_raw.reshape(-1, 1)).flatten()[0])
        # Clip to 0-1 in case of out-of-range values at inference
        iso_score  = float(np.clip(iso_score, 0, 1))

        # ── Combined weighted score ──
        combined   = (GB_WEIGHT * gb_proba) + (IF_WEIGHT * iso_score)
        safe_prob  = 1 - combined

    # ── 3-Level Risk Classification ──
    if combined >= 0.55:
        risk_level = "FRAUD"
    elif combined >= 0.15:
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
                Combined fraud score: <strong>{combined*100:.1f}%</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    elif risk_level == "SUSPICIOUS":
        st.markdown(f"""
        <div class="result-box result-suspicious">
            <div class="result-title suspicious-color">⚠️ Suspicious Transaction</div>
            <div class="result-sub">
                This transaction shows <strong>suspicious patterns</strong>.
                Recommend flagging for <strong>analyst review</strong> before processing.
                Combined fraud score: <strong>{combined*100:.1f}%</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-box result-safe">
            <div class="result-title safe-color">✅ Transaction Looks Safe</div>
            <div class="result-sub">
                No fraud indicators detected. This transaction appears <strong>legitimate</strong>.
                Combined fraud score: <strong>{combined*100:.1f}%</strong>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Score Breakdown — all 3 scores ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Score Breakdown</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(f"""
        <div class="score-box">
            <div class="score-label">🤖 Gradient Boosting ({int(GB_WEIGHT*100)}%)</div>
            <div class="score-value">{gb_proba*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="score-box">
            <div class="score-label">🔍 Isolation Forest ({int(IF_WEIGHT*100)}%)</div>
            <div class="score-value">{iso_score*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="score-box">
            <div class="score-label">⚡ Combined Score</div>
            <div class="score-value">{combined*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Confidence Bar ──
    st.markdown('<div class="section-label">Confidence</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("✅ Legitimate probability")
        st.progress(float(np.clip(safe_prob, 0, 1)))
        st.write(f"**{np.clip(safe_prob, 0, 1)*100:.1f}%**")
    with c2:
        st.caption("🚨 Fraud probability")
        st.progress(float(np.clip(combined, 0, 1)))
        st.write(f"**{np.clip(combined, 0, 1)*100:.1f}%**")

    # ── What Triggered The Flag ──
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">What The Models Saw</div>', unsafe_allow_html=True)

    drain_score_val          = input_df['drain_score'].values[0]
    both_suspicious_val      = input_df['both_accounts_suspicious'].values[0]
    dest_empty_val           = input_df['dest_was_empty'].values[0]
    amount_ratio_val         = input_df['amount_ratio'].values[0]

    flags = []
    if drain_score_val > 0.8:
        flags.append(f"🔴 **Drain Score {drain_score_val:.2f}** — sender's account nearly emptied in one transaction")
    if both_suspicious_val == 1:
        flags.append("🔴 **Both Accounts Suspicious** — sender had little left AND receiver was empty")
    if iso_score > 0.6:
        flags.append(f"🔴 **Isolation Forest Score {iso_score:.2f}** — transaction is anomalous compared to normal patterns")
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
st.caption(
    f"Group 4 · Digital Banking Fraud Detection System · "
    f"GB ({int(GB_WEIGHT*100)}%) + IF ({int(IF_WEIGHT*100)}%) · Built with Streamlit"
)
