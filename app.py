import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FraudShield — Group 4",
    page_icon="🛡️",
    layout="centered",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; background-color: #080c14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 1rem 4rem 1rem !important; max-width: 740px !important; }

/* Hero */
.hero-wrap { text-align: center; padding: 2rem 0 1.5rem 0; }
.hero-badge {
    display: inline-block;
    background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.3);
    color: #00d4aa; font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 2px; padding: 4px 14px;
    border-radius: 20px; margin-bottom: 14px;
}
.hero-title { font-size: clamp(2rem,6vw,3rem); font-weight: 900; color: #fff; line-height:1.1; margin-bottom:8px; letter-spacing:-1.5px; }
.hero-title span { background: linear-gradient(135deg,#00d4aa,#0099ff); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-sub { color:#5a6478; font-size:clamp(0.8rem,3vw,0.95rem); }

/* Section headers */
.sec-header { display:flex; align-items:center; gap:10px; margin:2rem 0 0.75rem 0; }
.sec-dot { width:6px; height:6px; background:#00d4aa; border-radius:50%; flex-shrink:0; }
.sec-title { font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:#00d4aa; }
.sec-line { flex:1; height:1px; background:linear-gradient(to right,rgba(0,212,170,0.2),transparent); }

/* Info pill */
.info-pill { background:rgba(0,153,255,0.07); border:1px solid rgba(0,153,255,0.15); border-radius:10px; padding:10px 16px; font-size:0.8rem; color:#4a9eff; margin-bottom:16px; line-height:1.5; }

/* Divider */
.divider { border:none; border-top:1px solid #1a2035; margin:1.5rem 0; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg,#00d4aa,#0099ff) !important;
    color: #080c14 !important; border: none !important; border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 14px 0 !important; width: 100% !important;
    letter-spacing: 0.5px !important; transition: opacity 0.2s, transform 0.1s !important; margin-top: 8px !important;
}
.stButton > button:hover { opacity:0.9 !important; transform:translateY(-1px) !important; }

/* Result cards */
.result-card { border-radius:16px; padding:24px; margin:16px 0; position:relative; overflow:hidden; }
.result-card::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.result-fraud { background:linear-gradient(135deg,#1a0a0a,#120606); border:1px solid rgba(255,59,59,0.35); }
.result-fraud::before { background:linear-gradient(90deg,#ff3b3b,#ff8c00); }
.result-suspicious { background:linear-gradient(135deg,#1a0f00,#120b00); border:1px solid rgba(255,160,0,0.35); }
.result-suspicious::before { background:linear-gradient(90deg,#ffa000,#ffcc00); }
.result-safe { background:linear-gradient(135deg,#021a0f,#011009); border:1px solid rgba(0,212,170,0.35); }
.result-safe::before { background:linear-gradient(90deg,#00d4aa,#0099ff); }
.result-icon { font-size:2.2rem; margin-bottom:8px; }
.result-label { font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:3px; text-transform:uppercase; margin-bottom:4px; }
.result-fraud .result-label { color:#ff6b6b; }
.result-suspicious .result-label { color:#ffa000; }
.result-safe .result-label { color:#00d4aa; }
.result-title { font-size:clamp(1.4rem,4vw,1.9rem); font-weight:900; color:#fff; letter-spacing:-0.5px; margin-bottom:6px; }
.result-desc { color:#5a6478; font-size:0.85rem; line-height:1.6; margin-bottom:12px; }
.result-score { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.75rem; padding:5px 12px; border-radius:20px; }
.result-fraud .result-score { background:rgba(255,59,59,0.1); color:#ff6b6b; border:1px solid rgba(255,59,59,0.2); }
.result-suspicious .result-score { background:rgba(255,160,0,0.1); color:#ffa000; border:1px solid rgba(255,160,0,0.2); }
.result-safe .result-score { background:rgba(0,212,170,0.1); color:#00d4aa; border:1px solid rgba(0,212,170,0.2); }

/* Score pills */
.scores-row { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:16px 0; }
@media(max-width:480px){ .scores-row{ grid-template-columns:1fr; } }
.score-pill { background:#0d1117; border:1px solid #1a2035; border-radius:12px; padding:14px 16px; text-align:center; }
.score-pill-label { font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:1.5px; text-transform:uppercase; color:#5a6478; margin-bottom:6px; }
.score-pill-value { font-size:1.5rem; font-weight:700; color:#fff; }
.score-pill-sub { font-size:0.7rem; color:#5a6478; margin-top:2px; }

/* Feature detail table */
.feat-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; margin:12px 0; }
@media(max-width:480px){ .feat-grid{ grid-template-columns:1fr; } }
.feat-cell { background:#0d1117; border:1px solid #1a2035; border-radius:10px; padding:12px 14px; }
.feat-cell-label { font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:1px; text-transform:uppercase; color:#5a6478; margin-bottom:4px; }
.feat-cell-value { font-size:1rem; font-weight:700; color:#fff; }
.feat-cell-desc { font-size:0.72rem; color:#3a4455; margin-top:2px; line-height:1.4; }
.feat-cell.high .feat-cell-value { color:#ff6b6b; }
.feat-cell.med .feat-cell-value { color:#ffa000; }
.feat-cell.low .feat-cell-value { color:#00d4aa; }

/* Flags */
.flag-item { display:flex; align-items:flex-start; gap:10px; padding:10px 14px; border-radius:10px; margin-bottom:8px; font-size:0.85rem; line-height:1.5; }
.flag-red { background:rgba(255,59,59,0.07); border:1px solid rgba(255,59,59,0.15); color:#ff9090; }
.flag-yellow { background:rgba(255,160,0,0.07); border:1px solid rgba(255,160,0,0.15); color:#ffc04a; }
.flag-green { background:rgba(0,212,170,0.07); border:1px solid rgba(0,212,170,0.15); color:#00d4aa; }

/* Recommendation box */
.rec-box { border-radius:12px; padding:16px 20px; margin-top:16px; }
.rec-fraud { background:rgba(255,59,59,0.06); border:1px solid rgba(255,59,59,0.2); }
.rec-suspicious { background:rgba(255,160,0,0.06); border:1px solid rgba(255,160,0,0.2); }
.rec-safe { background:rgba(0,212,170,0.06); border:1px solid rgba(0,212,170,0.2); }
.rec-title { font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:2px; text-transform:uppercase; margin-bottom:10px; }
.rec-fraud .rec-title { color:#ff6b6b; }
.rec-suspicious .rec-title { color:#ffa000; }
.rec-safe .rec-title { color:#00d4aa; }
.rec-step { display:flex; align-items:flex-start; gap:10px; margin-bottom:8px; font-size:0.82rem; color:#8892a4; line-height:1.5; }
.rec-num { background:#1a2035; color:#00d4aa; font-family:'JetBrains Mono',monospace; font-size:0.65rem; font-weight:600; width:20px; height:20px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:1px; }

/* Progress */
.stProgress > div > div { background:linear-gradient(90deg,#00d4aa,#0099ff) !important; border-radius:4px !important; }
.stProgress > div { background:#1a2035 !important; border-radius:4px !important; height:6px !important; }

/* Footer */
.footer { text-align:center; padding:2rem 0 1rem 0; font-family:'JetBrains Mono',monospace; font-size:0.65rem; letter-spacing:1.5px; color:#2a3040; }

@media(max-width:640px){
    .block-container{ padding:1rem 0.5rem 3rem 0.5rem !important; }
    .result-card{ padding:18px; }
    .hero-wrap{ padding:1.5rem 0 1rem 0; }
}
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
    except FileNotFoundError:
        return None, None, None, None, None

gb_model, iso_model, iso_scaler, feature_list, weights = load_models()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">GROUP 4 · AI FRAUD DETECTION</div>
    <div class="hero-title">Fraud<span>Shield</span></div>
    <div class="hero-sub">Real-time digital banking fraud analysis<br>powered by Gradient Boosting + Isolation Forest</div>
</div>
""", unsafe_allow_html=True)

if gb_model is None:
    st.error("⚠️ Model files not found. Ensure all 5 files exist in `models/`.")
    st.stop()

GB_WEIGHT = weights['gb']
IF_WEIGHT = weights['if']

st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECTION 1 — TRANSACTION DETAILS
# ─────────────────────────────────────────────

st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Transaction Details</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="info-pill">ℹ️ Only pre-transaction values are used — post-transaction balances are excluded to simulate a real-time decision.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tx_type = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER"])
with col2:
    step = st.number_input("Time Step (Hour)", min_value=0, value=150, step=1)

amount = st.number_input("Transaction Amount (₦)", min_value=0.0, value=50000.0, step=100.0, format="%.2f")

col3, col4 = st.columns(2)
with col3:
    oldbalanceOrg = st.number_input("Sender Balance Before (₦)", min_value=0.0, value=100000.0, step=100.0, format="%.2f")
with col4:
    oldbalanceDest = st.number_input("Receiver Balance Before (₦)", min_value=0.0, value=0.0, step=100.0, format="%.2f")


# ─────────────────────────────────────────────
# SECTION 2 — SENDER PROFILE
# ─────────────────────────────────────────────

st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Sender Profile</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="info-pill">ℹ️ Sender historical data. Leave as defaults if unknown.</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    sender_avg_amount = st.number_input("Avg Transaction Amount (₦)", min_value=0.0, value=40000.0, step=100.0, format="%.2f")
with col6:
    sender_total_sent = st.number_input("Total Ever Sent (₦)", min_value=0.0, value=200000.0, step=100.0, format="%.2f")


# ─────────────────────────────────────────────
# COMPUTE FEATURES
# ─────────────────────────────────────────────

def compute_features():
    type_encoded             = 0 if tx_type == "CASH_OUT" else 1
    amount_ratio             = min(amount / (oldbalanceOrg + 1), 10)
    # ✅ Fixed: use < 0.01 instead of == 0 to handle float precision
    dest_was_empty           = 1 if oldbalanceDest < 0.01 else 0
    hour_of_day              = step % 24
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

with st.expander("🔬 View computed feature vector"):
    df_preview = compute_features()
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Derived Features")
        st.dataframe(df_preview[["amount_ratio","dest_was_empty","hour_of_day"]], width='stretch')
    with col_b:
        st.caption("Unique Features")
        st.dataframe(df_preview[["drain_score","both_accounts_suspicious"]], width='stretch')
    st.caption("Full Feature Vector")
    st.dataframe(df_preview[feature_list], width='stretch')


# ─────────────────────────────────────────────
# ANALYSE BUTTON
# ─────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)

if st.button("🛡️ Analyse Transaction"):
    input_df = compute_features()[feature_list]

    with st.spinner("Analysing transaction..."):
        gb_proba  = float(gb_model.predict_proba(input_df)[0][1])
        iso_raw   = iso_model.decision_function(input_df)
        iso_score = float(np.clip(iso_scaler.transform(-iso_raw.reshape(-1,1)).flatten()[0], 0, 1))
        combined  = float(np.clip((GB_WEIGHT * gb_proba) + (IF_WEIGHT * iso_score), 0, 1))
        safe_prob = float(np.clip(1 - combined, 0, 1))

    if combined >= 0.55:
        risk = "FRAUD"
    elif combined >= 0.15:
        risk = "SUSPICIOUS"
    else:
        risk = "SAFE"

    # ── Pull feature values for display ──
    drain_val        = input_df['drain_score'].values[0]
    both_val         = int(input_df['both_accounts_suspicious'].values[0])
    dest_empty_val   = int(input_df['dest_was_empty'].values[0])
    amount_ratio_val = input_df['amount_ratio'].values[0]
    hour_val         = int(input_df['hour_of_day'].values[0])
    type_val         = "CASH_OUT" if input_df['type_encoded'].values[0] == 0 else "TRANSFER"

    # ── Result Card ──
    if risk == "FRAUD":
        st.markdown(f"""<div class="result-card result-fraud">
            <div class="result-icon">🚨</div>
            <div class="result-label">High Confidence Fraud</div>
            <div class="result-title">Block This Transaction</div>
            <div class="result-desc">Both the supervised model and anomaly detector flagged this transaction.
            Pattern matches known fraud — account draining with suspicious receiver profile.</div>
            <div class="result-score">Combined Risk Score: {combined*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    elif risk == "SUSPICIOUS":
        st.markdown(f"""<div class="result-card result-suspicious">
            <div class="result-icon">⚠️</div>
            <div class="result-label">Suspicious Activity</div>
            <div class="result-title">Hold for Review</div>
            <div class="result-desc">Transaction shows irregular patterns that don't match normal behaviour.
            Requires manual analyst review before processing.</div>
            <div class="result-score">Combined Risk Score: {combined*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="result-card result-safe">
            <div class="result-icon">✅</div>
            <div class="result-label">All Clear</div>
            <div class="result-title">Transaction Approved</div>
            <div class="result-desc">No fraud indicators detected across both models.
            Transaction behaviour is consistent with legitimate activity.</div>
            <div class="result-score">Combined Risk Score: {combined*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Score Breakdown ──
    st.markdown('<div class="sec-header" style="margin-top:1.5rem"><div class="sec-dot"></div><div class="sec-title">Model Scores</div><div class="sec-line"></div></div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="scores-row">
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

    # ── Feature Detail Grid ──
    st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Feature Breakdown</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    def risk_class(val, high_thresh, med_thresh, invert=False):
        if invert:
            return "low" if val > high_thresh else "med" if val > med_thresh else "high"
        return "high" if val > high_thresh else "med" if val > med_thresh else "low"

    drain_cls    = risk_class(drain_val, 0.8, 0.5)
    ratio_cls    = risk_class(amount_ratio_val, 5, 2)
    dest_cls     = "high" if dest_empty_val == 1 else "low"
    both_cls     = "high" if both_val == 1 else "low"
    hour_cls     = "med" if 0 <= hour_val <= 6 else "low"
    gb_cls       = risk_class(gb_proba, 0.55, 0.15)
    if_cls       = risk_class(iso_score, 0.6, 0.3)

    st.markdown(f"""
    <div class="feat-grid">
        <div class="feat-cell {drain_cls}">
            <div class="feat-cell-label">Drain Score</div>
            <div class="feat-cell-value">{drain_val:.3f}</div>
            <div class="feat-cell-desc">Portion of sender balance removed. &gt;0.8 = account nearly emptied</div>
        </div>
        <div class="feat-cell {ratio_cls}">
            <div class="feat-cell-label">Amount Ratio</div>
            <div class="feat-cell-value">{amount_ratio_val:.2f}×</div>
            <div class="feat-cell-desc">Transaction size vs sender balance. &gt;5× = disproportionately large</div>
        </div>
        <div class="feat-cell {dest_cls}">
            <div class="feat-cell-label">Receiver Was Empty</div>
            <div class="feat-cell-value">{"YES ⚠️" if dest_empty_val == 1 else "NO ✓"}</div>
            <div class="feat-cell-desc">Receiver had zero balance before — classic mule account indicator</div>
        </div>
        <div class="feat-cell {both_cls}">
            <div class="feat-cell-label">Both Accounts Suspicious</div>
            <div class="feat-cell-value">{"YES ⚠️" if both_val == 1 else "NO ✓"}</div>
            <div class="feat-cell-desc">Sender depleted AND receiver empty — strongest fraud chain pattern</div>
        </div>
        <div class="feat-cell {hour_cls}">
            <div class="feat-cell-label">Hour of Day</div>
            <div class="feat-cell-value">{hour_val:02d}:00</div>
            <div class="feat-cell-desc">{"Off-hours transaction (00:00–06:00) — elevated risk window" if 0 <= hour_val <= 6 else "Normal business hours — lower time-based risk"}</div>
        </div>
        <div class="feat-cell">
            <div class="feat-cell-label">Transaction Type</div>
            <div class="feat-cell-value">{type_val}</div>
            <div class="feat-cell-desc">{"Cash withdrawal — most common fraud vector" if type_val == "CASH_OUT" else "Inter-account transfer — second most common fraud vector"}</div>
        </div>
        <div class="feat-cell {gb_cls}">
            <div class="feat-cell-label">GB Fraud Probability</div>
            <div class="feat-cell-value">{gb_proba*100:.1f}%</div>
            <div class="feat-cell-desc">Gradient Boosting learned from 6,498 labelled fraud cases in training</div>
        </div>
        <div class="feat-cell {if_cls}">
            <div class="feat-cell-label">IF Anomaly Score</div>
            <div class="feat-cell-value">{iso_score*100:.1f}%</div>
            <div class="feat-cell-desc">Isolation Forest trained on 2.2M normal transactions — flags statistical outliers</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Signal Flags ──
    st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Triggered Signals</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    flags_html = ""
    if drain_val > 0.8:
        flags_html += f'<div class="flag-item flag-red">🔴 <strong>Critical Drain ({drain_val:.2f})</strong> — {drain_val*100:.0f}% of sender balance removed. Matches account-draining fraud pattern.</div>'
    if both_val == 1:
        flags_html += '<div class="flag-item flag-red">🔴 <strong>Dual Zero Pattern</strong> — Sender balance exhausted AND receiver had no prior balance. Classic two-account fraud chain.</div>'
    if iso_score > 0.6:
        flags_html += f'<div class="flag-item flag-red">🔴 <strong>Statistical Anomaly ({iso_score*100:.0f}%)</strong> — Isolation Forest: this transaction is an outlier among 2.2M normal transactions.</div>'
    if gb_proba > 0.55:
        flags_html += f'<div class="flag-item flag-red">🔴 <strong>High GB Fraud Score ({gb_proba*100:.0f}%)</strong> — Gradient Boosting model assigns high fraud probability based on learned patterns.</div>'
    if dest_empty_val == 1:
        flags_html += '<div class="flag-item flag-yellow">🟡 <strong>Empty Receiver Account</strong> — Destination had zero balance. Could indicate a newly created mule account.</div>'
    if amount_ratio_val >= 5:
        flags_html += f'<div class="flag-item flag-yellow">🟡 <strong>Oversized Transaction ({amount_ratio_val:.1f}×)</strong> — Amount is {amount_ratio_val:.1f}× the sender\'s remaining balance. Highly disproportionate.</div>'
    if 0 <= hour_val <= 6:
        flags_html += f'<div class="flag-item flag-yellow">🟡 <strong>Off-Hours Transaction ({hour_val:02d}:00)</strong> — Transactions between midnight and 6am carry elevated fraud risk.</div>'
    if not flags_html:
        flags_html = '<div class="flag-item flag-green">🟢 <strong>No Fraud Signals Triggered</strong> — All 8 indicators are within normal range for legitimate transactions.</div>'

    st.markdown(flags_html, unsafe_allow_html=True)

    # ── Recommendation ──
    st.markdown('<div class="sec-header"><div class="sec-dot"></div><div class="sec-title">Recommended Action</div><div class="sec-line"></div></div>', unsafe_allow_html=True)

    if risk == "FRAUD":
        st.markdown(f"""<div class="rec-box rec-fraud">
            <div class="rec-title">⛔ Fraud Response Protocol</div>
            <div class="rec-step"><div class="rec-num">1</div>Immediately block and reverse the transaction before funds are transferred.</div>
            <div class="rec-step"><div class="rec-num">2</div>Freeze both the sender account and the destination account pending investigation.</div>
            <div class="rec-step"><div class="rec-num">3</div>Escalate to the fraud operations team with this report as supporting evidence.</div>
            <div class="rec-step"><div class="rec-num">4</div>Notify the account holder via verified contact channel of the attempted fraud.</div>
            <div class="rec-step"><div class="rec-num">5</div>File a Suspicious Activity Report (SAR) as required by regulatory guidelines.</div>
        </div>""", unsafe_allow_html=True)
    elif risk == "SUSPICIOUS":
        st.markdown(f"""<div class="rec-box rec-suspicious">
            <div class="rec-title">⚠️ Review Protocol</div>
            <div class="rec-step"><div class="rec-num">1</div>Place the transaction on hold — do not process until reviewed.</div>
            <div class="rec-step"><div class="rec-num">2</div>Assign to a fraud analyst for manual inspection within 2 hours.</div>
            <div class="rec-step"><div class="rec-num">3</div>Request additional verification from the account holder if needed.</div>
            <div class="rec-step"><div class="rec-num">4</div>Document findings and approve or reject with justification.</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="rec-box rec-safe">
            <div class="rec-title">✅ Approval Protocol</div>
            <div class="rec-step"><div class="rec-num">1</div>Transaction cleared — proceed with normal processing.</div>
            <div class="rec-step"><div class="rec-num">2</div>No further action required. Log as approved for audit trail.</div>
            <div class="rec-step"><div class="rec-num">3</div>Continue monitoring account for any unusual follow-up activity.</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    f'<div class="footer">FRAUDSHIELD · GROUP 4 · GB {int(GB_WEIGHT*100)}% + IF {int(IF_WEIGHT*100)}% · STREAMLIT</div>',
    unsafe_allow_html=True
)
