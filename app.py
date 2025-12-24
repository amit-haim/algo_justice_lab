import streamlit as st
import pandas as pd
import numpy as np

# Try to import plotly, but handle it gracefully if it's still installing
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 1. PAGE CONFIG (Must be at the very top)
st.set_page_config(page_title="Algorithmic Justice Lab", layout="wide")

# 2. BACKEND DATA (Hardcoded for stability)
CASE_STUDY = {
    "name": "Applicant #402: 'Sarah J.'",
    "details": """**Age:** 45 | **Occupation:** Warehouse Manager  
    **Claim:** Chronic Back Pain. MRI shows moderate degeneration.  
    **Note:** Social media shows a hiking trip from 6 months ago.""",
    "ai_risk_score": 72,
    "ground_truth": "Eligible"
}

@st.cache_data
def load_simulated_data():
    np.random.seed(42)
    scores = np.concatenate([np.random.normal(75, 12, 700), np.random.normal(35, 18, 300)])
    actual = np.concatenate([np.ones(700), np.zeros(300)])
    return pd.DataFrame({"score": np.clip(scores, 0, 100), "actual": actual})

# 3. UI LAYOUT
st.title("⚖️ Algorithmic Justice Lab")

if not HAS_PLOTLY:
    st.warning("The visualization library is still loading on the server. Please refresh in 10 seconds.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Stage 1: The Decision", "Stage 2: The Tradeoffs", "Stage 3: Policy"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Application File")
        st.info(CASE_STUDY['details'])
    with col2:
        # Needle Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = CASE_STUDY['ai_risk_score'],
            title = {'text': "AI Score (Eligibility %)"},
            gauge = {'axis': {'range': [0, 100]}, 'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 100], 'color': "green"}]}))
        st.plotly_chart(fig, use_container_width=True)
    
    choice = st.selectbox("Your Decision:", ["Pending", "Approve", "Deny"])
    if choice != "Pending":
        st.write(f"The ground truth for this case was: **{CASE_STUDY['ground_truth']}**")

with tab2:
    st.header("Model Tuning")
    threshold = st.slider("Strictness Threshold", 0, 100, 50)
    data = load_simulated_data()
    
    # Calculate FP/FN
    fps = len(data[(data['score'] >= threshold) & (data['actual'] == 0)])
    fns = len(data[(data['score'] < threshold) & (data['actual'] == 1)])
    
    c1, c2 = st.columns(2)
    c1.metric("Wrongful Denials (FN)", fns)
    c2.metric("Wrongful Approvals (FP)", fps)
    
    st.bar_chart(pd.DataFrame({"Error": ["FN (Harm)", "FP (Waste)"], "Count": [fns, fps]}).set_index("Error"))

with tab3:
    st.header("Policy Tradeoffs")
    appeal = st.checkbox("Enable Human Appeal")
    wait = 2 + (10 if appeal else 0)
    trust = 40 + (30 if appeal else 0)
    
    st.write(f"**Predicted Wait Time:** {wait} Weeks")
    st.progress(wait/20)
    st.write(f"**Public Trust Level:** {trust}%")
    st.progress(trust/100)
