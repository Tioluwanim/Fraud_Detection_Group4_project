import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="FraudShield — Group 4",
    page_icon="🛡️",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── LIGHT THEME ── */
html, body { background-color: #f5f7fa !important; }
.stApp { background-color: #f5f7fa !important; }
.stApp > div { background-color: #f5f7fa !important; }
section[data-testid="stSidebar"] { background-color: #eef1f6 !important; }
.main .block-container { background-color: #f5f7fa !important; }
[data-testid="stAppViewContainer"] { background-color: #f5f7fa !important; }
[data-testid="stHeader"] { background-color: #f5f7fa !important; }
[data-testid="stToolbar"] { display: none; }

/* ── Typography base ── */
html, body, [class*="css"], .stApp { font-family: 'Outfit', sans-serif !important; color: #1a2035; }

/* ── Hide chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1.2rem 4rem 1.2rem !important; max-width: 760px !important; }

/* ── Input fields — light ── */
.stTextInput input,
.stNumberInput input,
input[type="number"],
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input {
    background-color: #ffffff !important;
    border: 1.5px solid #dde3ed !important;
    border-radius: 10px !important;
    color: #1a2035 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    caret-color: #0070f3 !important;
}
div[data-baseweb="input"],
div[data-baseweb="base-input"] {
    background-color: #ffffff !important;
    border: 1.5px solid #dde3ed !important;
    border-radius: 10px !important;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="base-input"]:focus-within {
    border-color: #0070f3 !important;
    box-shadow: 0 0 0 3px rgba(0,112,243,0.1) !important;
}

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1.5px solid #dde3ed !important;
    border-radius: 10px !important;
    color: #1a2035 !important;
}
div[data-baseweb="select"] span { color: #1a2035 !important; }
div[data-baseweb="popover"] { background-color: #ffffff !important; border: 1px solid #dde3ed !important; box-shadow: 0 8px 30px rgba(0,0,0,0.1) !important; }
li[role="option"] { background-color: #ffffff !important; color: #1a2035 !important; }
li[role="option"]:hover { background-color: #f0f4ff !important; }

/* ── Labels ── */
label, .stSelectbox label, .stNumberInput label, p {
    color: #4a5568 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
}

/* ── Number input buttons ── */
button[data-testid="stNumberInput-StepDown"],
button[data-testid="stNumberInput-StepUp"] {
    background-color: #f0f4ff !important;
    border: 1px solid #dde3ed !important;
    color: #4a5568 !important;
    border-radius: 6px !important;
}
button[data-testid="stNumberInput-StepDown"]:hover,
button[data-testid="stNumberInput-StepUp"]:hover {
    background-color: #0070f3 !important;
    color: #ffffff !important;
}

/* ── Expander ── */
details { background-color: #ffffff !important; border: 1px solid #dde3ed !important; border-radius: 10px !important; }
summary { color: #7a8499 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important; padding: 10px 16px !important; }
.streamlit-expanderHeader { background-color: #ffffff !important; color: #7a8499 !important; border-radius: 10px !important; }
.streamlit-expanderContent { background-color: #ffffff !important; border-top: 1px solid #dde3ed !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { background-color: #ffffff !important; border: 1px solid #dde3ed !important; border-radius: 8px !important; }

/* ── Progress bars ── */
.stProgress > div > div { background: linear-gradient(90deg,#0070f3,#00c9a7) !important; border-radius: 4px !important; }
.stProgress > div { background: #e8edf5 !important; border-radius: 4px !important; height: 6px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #0070f3 !important; }

/* ── Hero ── */
.hero-wrap { text-align: center; padding: 2.5rem 0 1.5rem 0; }
.hero-badge {
    display: inline-block;
    background: rgba(0,112,243,0.08); border: 1px solid rgba(0,112,243,0.25);
    color: #0070f3; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 2px; padding: 5px 16px;
    border-radius: 20px; margin-bottom: 18px;
}
.hero-title {
    font-size: clamp(2.2rem, 6vw, 3.2rem);
    font-weight: 900; line-height: 1.05;
    margin-bottom: 10px; letter-spacing: -2px;
    color: #0f172a;
}
.hero-title .gradient {
    background: linear-gradient(135deg, #0070f3, #00c9a7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    color: transparent;
}
.hero-sub { color: #7a8499; font-size: clamp(0.82rem, 2.5vw, 0.95rem); line-height: 1.6; }

/* ── Section headers ── */
.sec-header { display: flex; align-items: center; gap: 10px; margin: 2rem 0 0.8rem 0; }
.sec-dot { width: 6px; height: 6px; background: #0070f3; border-radius: 50%; flex-shrink: 0; }
.sec-title { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 600; letter-spacing: 2.5px; text-transform: uppercase; color: #0070f3; white-space: nowrap; }
.sec-line { flex: 1; height: 1px; background: linear-gradient(to right, rgba(0,112,243,0.25), transparent); }

/* ── Info pill ── */
.info-pill { background: rgba(0,112,243,0.05); border: 1px solid rgba(0,112,243,0.18); border-radius: 10px; padding: 10px 16px; font-size: 0.8rem; color: #0070f3; margin-bottom: 16px; line-height: 1.6; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0070f3, #00c9a7) !important;
    color: #ffffff !important; border: none !important; border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 14px 0 !important; width: 100% !important;
    letter-spacing: 0.5px !important; transition: opacity 0.2s, transform 0.1s !important;
    margin-top: 8px !important; box-shadow: 0 4px 20px rgba(0,112,243,0.25) !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result cards ── */
.result-card { border-radius: 16px; padding: 24px; margin: 16px 0; position: relative; overflow: hidden; }
.result-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; }
.result-fraud  { background: linear-gradient(145deg,#fff5f5,#fff0f0); border: 1px solid rgba(229,62,62,0.25); }
.result-fraud::before  { background: linear-gradient(90deg,#e53e3e,#f97316); }
.result-suspicious { background: linear-gradient(145deg,#fffbf0,#fff8e6); border: 1px solid rgba(214,158,46,0.3); }
.result-suspicious::before { background: linear-gradient(90deg,#d69e2e,#f6e05e); }
.result-safe   { background: linear-gradient(145deg,#f0fdf9,#edfcf6); border: 1px solid rgba(0,159,127,0.25); }
.result-safe::before   { background: linear-gradient(90deg,#00c9a7,#0070f3); }
.result-icon { font-size: 2.2rem; margin-bottom: 8px; display: block; }
.result-label { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; letter-spacing: 3px; text-transform: uppercase; margin-bottom: 5px; display: block; }
.result-fraud .result-label    { color: #e53e3e; }
.result-suspicious .result-label { color: #c07a00; }
.result-safe .result-label     { color: #00967a; }
.result-title { font-size: clamp(1.4rem,4vw,1.9rem); font-weight: 900; color: #0f172a; letter-spacing: -0.5px; margin-bottom: 8px; display: block; }
.result-desc { color: #64748b; font-size: 0.85rem; line-height: 1.7; margin-bottom: 14px; display: block; }
.result-score { display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; padding: 5px 14px; border-radius: 20px; }
.result-fraud .result-score      { background: rgba(229,62,62,0.08); color: #e53e3e; border: 1px solid rgba(229,62,62,0.2); }
.result-suspicious .result-score { background: rgba(214,158,46,0.1); color: #c07a00; border: 1px solid rgba(214,158,46,0.25); }
.result-safe .result-score       { background: rgba(0,201,167,0.1); color: #00967a; border: 1px solid rgba(0,201,167,0.25); }

/* ── Score pills ── */
.scores-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin: 14px 0; }
@media(max-width:520px){ .scores-row { grid-template-columns: 1fr; } }
.score-pill { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
.score-pill-label { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; letter-spacing: 1.5px; text-transform: uppercase; color: #94a3b8; margin-bottom: 6px; }
.score-pill-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
.score-pill-sub { font-size: 0.68rem; color: #cbd5e1; margin-top: 3px; }

/* ── Feature grid ── */
.feat-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 8px; margin: 12px 0; }
@media(max-width:520px){ .feat-grid { grid-template-columns: 1fr; } }
.feat-cell { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 13px 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.feat-cell-label { font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; letter-spacing: 1px; text-transform: uppercase; color: #94a3b8; margin-bottom: 5px; }
.feat-cell-value { font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 3px; }
.feat-cell-desc { font-size: 0.71rem; color: #94a3b8; line-height: 1.45; }
.feat-cell.high .feat-cell-value { color: #e53e3e; }
.feat-cell.med  .feat-cell-value { color: #c07a00; }
.feat-cell.low  .feat-cell-value { color: #00967a; }

/* ── Flags ── */
.flag-item { display: flex; align-items: flex-start; gap: 10px; padding: 11px 14px; border-radius: 10px; margin-bottom: 8px; font-size: 0.83rem; line-height: 1.55; }
.flag-red    { background: rgba(229,62,62,0.05);  border: 1px solid rgba(229,62,62,0.18);  color: #c53030; }
.flag-yellow { background: rgba(214,158,46,0.07); border: 1px solid rgba(214,158,46,0.2);  color: #b7791f; }
.flag-green  { background: rgba(0,201,167,0.06);  border: 1px solid rgba(0,201,167,0.2);  color: #00967a; }

/* ── Recommendation ── */
.rec-box { border-radius: 12px; padding: 18px 20px; margin-top: 14px; }
.rec-fraud      { background: rgba(229,62,62,0.04);  border: 1px solid rgba(229,62,62,0.2); }
.rec-suspicious { background: rgba(214,158,46,0.05); border: 1px solid rgba(214,158,46,0.2); }
.rec-safe       { background: rgba(0,201,167,0.05);  border: 1px solid rgba(0,201,167,0.2); }
.rec-title { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px; display: block; }
.rec-fraud .rec-title      { color: #e53e3e; }
.rec-suspicious .rec-title { color: #c07a00; }
.rec-safe .rec-title       { color: #00967a; }
.rec-step { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; font-size: 0.82rem; color: #4a5568; line-height: 1.55; }
.rec-num { background: #e8f0fe; color: #0070f3; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px; }

/* ── Footer ── */
.footer { text-align: center; padding: 2rem 0 1rem 0; font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; letter-spacing: 2px; color: #c1cad8; }

/* ── Mobile ── */
@media(max-width:640px){
    .block-container { padding: 1rem 0.6rem 3rem 0.6rem !important; }
    .result-card { padding: 18px 16px; }
    .hero-wrap { padding: 1.5rem 0 1rem 0; }
    .hero-title { letter-spacing: -1px; }
}
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODELS ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    try:
        gb_model   = joblib.load("models/gradient_boosting_fraud_model.pkl")
        iso_model  = joblib.load("models/isolation_forest_model.pkl")
        iso_scaler = joblib.load("models/iso_scaler.pkl")
        features   = joblib.load("models/feature_list.pkl")
        weights    = joblib.load("models/model_weights.pkl")
        return gb_model, iso_model, iso_scaler, features, weights
    except FileNotFoundError:
        return None, None, None, None, None

gb_model, iso_model, iso_scaler, feature_list, weights = load_models()

# ─── HERO ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">GROUP 4 · AI FRAUD DETECTION</div>
    <div class="hero-title">Fraud<span class="gradient">Shield</span></div>
    <div class="hero-sub">Real-time digital banking fraud analysis<br>powered by Gradient Boosting + Isolation Forest</div>
</div>
""", unsafe_allow_html=True)

# ── BUG FIX: Check all model components loaded, not just gb_model ──
if any(v is None for v in [gb_model, iso_model, iso_scaler, feature_list, weights]):
    st.error(
        "⚠️ Model files not found. Ensure all 5 files exist in `models/`:\n\n"
        "- `gradient_boosting_fraud_model.pkl`\n"
        "- `isolation_forest_model.pkl`\n"
        "- `iso_scaler.pkl`\n"
        "- `feature_list.pkl`\n"
        "- `model_weights.pkl`"
    )
    st.stop()

GB_WEIGHT = weights['gb']
IF_WEIGHT = weights['if']

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── SECTION 1: TRANSACTION DETAILS ────────────────────────────────────────────

st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Transaction Details</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="info-pill">ℹ️ Only pre-transaction values are used — post-transaction balances are excluded to simulate a real-time decision.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tx_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])
with col2:
    step = st.number_input("Time Step (Hour)", min_value=0, value=150, step=1)

amount = st.number_input("Transaction Amount (₦)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")

col3, col4 = st.columns(2)
with col3:
    oldbalanceOrg  = st.number_input("Sender Balance Before (₦)",   min_value=0.0, value=100000.0, step=1000.0, format="%.2f")
with col4:
    oldbalanceDest = st.number_input("Receiver Balance Before (₦)", min_value=0.0, value=0.0,      step=1000.0, format="%.2f")


# ─── SECTION 2: SENDER PROFILE ─────────────────────────────────────────────────

st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Sender Profile</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="info-pill">ℹ️ Sender historical data. Leave as defaults if unknown — the model handles missing history gracefully.</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    sender_avg_amount = st.number_input("Avg Transaction Amount (₦)", min_value=0.0, value=40000.0, step=1000.0, format="%.2f")
with col6:
    sender_total_sent = st.number_input("Total Ever Sent (₦)",        min_value=0.0, value=200000.0, step=1000.0, format="%.2f")


# ─── COMPUTE FEATURES ──────────────────────────────────────────────────────────

def compute_features():
    type_encoded             = 0 if tx_type == "CASH_OUT" else 1
    amount_ratio             = min(amount / (oldbalanceOrg + 1), 10)
    dest_was_empty           = 1 if oldbalanceDest < 0.01 else 0
    hour_of_day              = int(step) % 24
    drain_score              = amount / (oldbalanceOrg + amount + 1)
    both_accounts_suspicious = 1 if (oldbalanceOrg <= amount and oldbalanceDest < 0.01) else 0
    return pd.DataFrame({
        "type_encoded":             [type_encoded],
        "amount":                   [amount],
        "oldbalanceOrg":            [oldbalanceOrg],
        "amount_ratio":             [amount_ratio],
        "dest_was_empty":           [dest_was_empty],
        "hour_of_day":              [hour_of_day],
        "sender_avg_amount":        [sender_avg_amount],
        "oldbalanceDest":           [oldbalanceDest],
        "drain_score":              [drain_score],
        "both_accounts_suspicious": [both_accounts_suspicious],
    })

# ── BUG FIX: use_container_width=True replaces invalid width='stretch' ──
with st.expander("🔬 View computed feature vector"):
    df_preview = compute_features()
    ca, cb = st.columns(2)
    with ca:
        st.caption("Derived Features")
        st.dataframe(df_preview[["amount_ratio", "dest_was_empty", "hour_of_day"]], use_container_width=True)
    with cb:
        st.caption("Unique Features")
        st.dataframe(df_preview[["drain_score", "both_accounts_suspicious"]], use_container_width=True)
    st.caption("Full Feature Vector (sent to model)")
    st.dataframe(df_preview[feature_list], use_container_width=True)


# ─── ANALYSE BUTTON ─────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)

if st.button("🛡️  Analyse Transaction"):
    # ── BUG FIX: compute once and reuse throughout ──
    input_df = compute_features()[feature_list]

    with st.spinner("Analysing transaction..."):
        gb_proba = float(gb_model.predict_proba(input_df)[0][1])

        # ── BUG FIX: safer Isolation Forest score normalization ──
        # decision_function returns higher = more normal; lower = more anomalous
        # We negate so higher value = more anomalous, then min-max scale to [0,1]
        iso_raw   = float(iso_model.decision_function(input_df)[0])
        iso_neg   = -iso_raw  # higher = more anomalous
        # Use scaler fitted on negated scores (same direction as training)
        iso_scaled = float(iso_scaler.transform(np.array([[iso_neg]]))[0][0])
        iso_score  = float(np.clip(iso_scaled, 0.0, 1.0))

        combined  = float(np.clip((GB_WEIGHT * gb_proba) + (IF_WEIGHT * iso_score), 0.0, 1.0))
        safe_prob = float(np.clip(1.0 - combined, 0.0, 1.0))

    risk = "FRAUD" if combined >= 0.55 else "SUSPICIOUS" if combined >= 0.15 else "SAFE"

    # Pull feature values from the already-computed input_df
    drain_val        = float(input_df['drain_score'].values[0])
    both_val         = int(input_df['both_accounts_suspicious'].values[0])
    dest_empty_val   = int(input_df['dest_was_empty'].values[0])
    amount_ratio_val = float(input_df['amount_ratio'].values[0])
    hour_val         = int(input_df['hour_of_day'].values[0])
    type_val         = "CASH_OUT" if int(input_df['type_encoded'].values[0]) == 0 else "TRANSFER"

    # ── Result Card ──
    if risk == "FRAUD":
        st.markdown(f"""
        <div class="result-card result-fraud">
            <span class="result-icon">🚨</span>
            <span class="result-label">High Confidence Fraud</span>
            <span class="result-title">Block This Transaction</span>
            <span class="result-desc">Both the supervised model and anomaly detector flagged this transaction.
            Pattern matches known account-draining fraud with suspicious receiver profile.</span>
            <span class="result-score">Combined Risk Score: {combined*100:.1f}%</span>
        </div>""", unsafe_allow_html=True)
    elif risk == "SUSPICIOUS":
        st.markdown(f"""
        <div class="result-card result-suspicious">
            <span class="result-icon">⚠️</span>
            <span class="result-label">Suspicious Activity</span>
            <span class="result-title">Hold for Review</span>
            <span class="result-desc">Transaction shows irregular patterns that deviate from normal behaviour.
            Requires manual analyst review before processing.</span>
            <span class="result-score">Combined Risk Score: {combined*100:.1f}%</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-safe">
            <span class="result-icon">✅</span>
            <span class="result-label">All Clear</span>
            <span class="result-title">Transaction Approved</span>
            <span class="result-desc">No fraud indicators detected across both models.
            Transaction behaviour is consistent with legitimate activity.</span>
            <span class="result-score">Combined Risk Score: {combined*100:.1f}%</span>
        </div>""", unsafe_allow_html=True)

    # ── Model Scores ──
    st.markdown('<div class="sec-header" style="margin-top:1.8rem"><div class="sec-dot"></div><div class="sec-title">Model Scores</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="scores-row">
        <div class="score-pill">
            <div class="score-pill-label">🤖 Gradient Boosting</div>
            <div class="score-pill-value">{gb_proba*100:.1f}%</div>
            <div class="score-pill-sub">supervised · weight {int(GB_WEIGHT*100)}%</div>
        </div>
        <div class="score-pill">
            <div class="score-pill-label">🔍 Isolation Forest</div>
            <div class="score-pill-value">{iso_score*100:.1f}%</div>
            <div class="score-pill-sub">anomaly · weight {int(IF_WEIGHT*100)}%</div>
        </div>
        <div class="score-pill">
            <div class="score-pill-label">⚡ Combined Score</div>
            <div class="score-pill-value">{combined*100:.1f}%</div>
            <div class="score-pill-sub">final decision score</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Confidence Bars ──
    st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Confidence</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("✅ Legitimate probability")
        st.progress(safe_prob)
        st.markdown(f"**{safe_prob*100:.1f}%**")
    with c2:
        st.caption("🚨 Fraud probability")
        st.progress(combined)
        st.markdown(f"**{combined*100:.1f}%**")

    # ── Feature Breakdown ──
    def risk_class(val, hi, mid):
        return "high" if val > hi else "med" if val > mid else "low"

    drain_cls  = risk_class(drain_val, 0.8, 0.5)
    ratio_cls  = risk_class(amount_ratio_val, 5, 2)
    dest_cls   = "high" if dest_empty_val == 1 else "low"
    both_cls   = "high" if both_val == 1 else "low"
    hour_cls   = "med"  if 0 <= hour_val <= 6 else "low"
    gb_cls     = risk_class(gb_proba, 0.55, 0.15)
    if_cls     = risk_class(iso_score, 0.6, 0.3)

    st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Feature Breakdown</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="feat-grid">
        <div class="feat-cell {drain_cls}">
            <div class="feat-cell-label">Drain Score</div>
            <div class="feat-cell-value">{drain_val:.3f}</div>
            <div class="feat-cell-desc">{drain_val*100:.0f}% of sender balance drained. &gt;0.8 = account nearly emptied.</div>
        </div>
        <div class="feat-cell {ratio_cls}">
            <div class="feat-cell-label">Amount Ratio</div>
            <div class="feat-cell-value">{amount_ratio_val:.2f}×</div>
            <div class="feat-cell-desc">Transaction vs sender balance. &gt;5× = disproportionately large.</div>
        </div>
        <div class="feat-cell {dest_cls}">
            <div class="feat-cell-label">Receiver Empty</div>
            <div class="feat-cell-value">{"YES ⚠️" if dest_empty_val == 1 else "NO ✓"}</div>
            <div class="feat-cell-desc">{"Receiver had ₦0 before — classic mule account indicator." if dest_empty_val == 1 else "Receiver had existing balance — lower risk."}</div>
        </div>
        <div class="feat-cell {both_cls}">
            <div class="feat-cell-label">Both Accounts Suspicious</div>
            <div class="feat-cell-value">{"YES ⚠️" if both_val == 1 else "NO ✓"}</div>
            <div class="feat-cell-desc">{"Sender depleted + receiver empty = strongest fraud chain signal." if both_val == 1 else "Accounts do not match dual-zero fraud pattern."}</div>
        </div>
        <div class="feat-cell {hour_cls}">
            <div class="feat-cell-label">Hour of Day</div>
            <div class="feat-cell-value">{hour_val:02d}:00</div>
            <div class="feat-cell-desc">{"Off-hours (00–06). Elevated risk window for fraud." if 0 <= hour_val <= 6 else "Normal business hours — lower time-based risk."}</div>
        </div>
        <div class="feat-cell">
            <div class="feat-cell-label">Transaction Type</div>
            <div class="feat-cell-value">{type_val}</div>
            <div class="feat-cell-desc">{"Cash withdrawal — most common fraud vector in dataset." if type_val == "CASH_OUT" else "Inter-account transfer — second most common fraud vector."}</div>
        </div>
        <div class="feat-cell {gb_cls}">
            <div class="feat-cell-label">GB Fraud Probability</div>
            <div class="feat-cell-value">{gb_proba*100:.1f}%</div>
            <div class="feat-cell-desc">Supervised model trained on 6,498 labelled fraud cases.</div>
        </div>
        <div class="feat-cell {if_cls}">
            <div class="feat-cell-label">IF Anomaly Score</div>
            <div class="feat-cell-value">{iso_score*100:.1f}%</div>
            <div class="feat-cell-desc">Isolation Forest trained on 2.2M normal transactions.</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Triggered Signals ──
    st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Triggered Signals</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    flags_html = ""
    if drain_val > 0.8:
        flags_html += f'<div class="flag-item flag-red">🔴 <strong>Critical Drain ({drain_val:.2f})</strong> — {drain_val*100:.0f}% of sender balance removed in one transaction. Matches account-draining fraud.</div>'
    if both_val == 1:
        flags_html += '<div class="flag-item flag-red">🔴 <strong>Dual Zero Pattern</strong> — Sender exhausted AND receiver was empty. Classic two-account fraud chain.</div>'
    if iso_score > 0.6:
        flags_html += f'<div class="flag-item flag-red">🔴 <strong>Statistical Anomaly ({iso_score*100:.0f}%)</strong> — Isolation Forest flagged this as an outlier among 2.2M normal transactions.</div>'
    if gb_proba > 0.55:
        flags_html += f'<div class="flag-item flag-red">🔴 <strong>High GB Fraud Score ({gb_proba*100:.0f}%)</strong> — Gradient Boosting assigns high fraud probability based on 10 learned features.</div>'
    if dest_empty_val == 1 and not both_val:
        flags_html += '<div class="flag-item flag-yellow">🟡 <strong>Empty Receiver</strong> — Destination had zero balance. Possible newly created mule account.</div>'
    if amount_ratio_val >= 5:
        flags_html += f'<div class="flag-item flag-yellow">🟡 <strong>Oversized Transaction ({amount_ratio_val:.1f}×)</strong> — Amount is {amount_ratio_val:.1f}× the sender\'s balance. Highly disproportionate.</div>'
    if 0 <= hour_val <= 6:
        flags_html += f'<div class="flag-item flag-yellow">🟡 <strong>Off-Hours ({hour_val:02d}:00)</strong> — Transactions between midnight and 6am carry elevated fraud risk.</div>'
    if not flags_html:
        flags_html = '<div class="flag-item flag-green">🟢 <strong>No Fraud Signals Triggered</strong> — All 8 indicators are within normal range for legitimate transactions.</div>'
    st.markdown(flags_html, unsafe_allow_html=True)

    # ── Recommended Action ──
    st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Recommended Action</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    if risk == "FRAUD":
        st.markdown("""<div class="rec-box rec-fraud">
            <span class="rec-title">⛔ Fraud Response Protocol</span>
            <div class="rec-step"><div class="rec-num">1</div>Immediately block and reverse the transaction before funds are transferred.</div>
            <div class="rec-step"><div class="rec-num">2</div>Freeze both the sender and destination accounts pending investigation.</div>
            <div class="rec-step"><div class="rec-num">3</div>Escalate to the fraud operations team with this report as evidence.</div>
            <div class="rec-step"><div class="rec-num">4</div>Notify the account holder via a verified contact channel.</div>
            <div class="rec-step"><div class="rec-num">5</div>File a Suspicious Activity Report (SAR) per regulatory guidelines.</div>
        </div>""", unsafe_allow_html=True)
    elif risk == "SUSPICIOUS":
        st.markdown("""<div class="rec-box rec-suspicious">
            <span class="rec-title">⚠️ Review Protocol</span>
            <div class="rec-step"><div class="rec-num">1</div>Place the transaction on hold — do not process until reviewed.</div>
            <div class="rec-step"><div class="rec-num">2</div>Assign to a fraud analyst for manual inspection within 2 hours.</div>
            <div class="rec-step"><div class="rec-num">3</div>Request additional verification from the account holder if needed.</div>
            <div class="rec-step"><div class="rec-num">4</div>Document findings and approve or reject with justification.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="rec-box rec-safe">
            <span class="rec-title">✅ Approval Protocol</span>
            <div class="rec-step"><div class="rec-num">1</div>Transaction cleared — proceed with normal processing.</div>
            <div class="rec-step"><div class="rec-num">2</div>Log as approved for audit trail. No further action required.</div>
            <div class="rec-step"><div class="rec-num">3</div>Continue monitoring account for any unusual follow-up activity.</div>
        </div>""", unsafe_allow_html=True)


# ─── FOOTER ─────────────────────────────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    f'<div class="footer">FRAUDSHIELD · GROUP 4 · GB {int(GB_WEIGHT*100)}% + IF {int(IF_WEIGHT*100)}% · STREAMLIT</div>',
    unsafe_allow_html=True
)
