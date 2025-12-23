import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ==========================================
# 1. PAGE CONFIG (MUST BE FIRST)
# ==========================================
st.set_page_config(
    page_title="Algorithmic Justice Lab", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. BACKEND CONFIGURATION
# ==========================================

# CASE STUDY: The specific applicant students review in Stage 1
CASE_STUDY = {
    "name": "Applicant #402: 'Sarah J.'",
    "details": """
    **Age:** 45  
    **Occupation:** Former Warehouse Logistics Manager  
    **Medical Claim:** Chronic Lumbar Radiculopathy.  
    **Evidence:** MRI shows moderate degeneration. Reports 'severe pain' lifting >10lbs.  
    **Social Data:** Posted photos of a hiking trip 6 months ago.
    """,
    "ai_risk_score": 72,  # 0-100.
    "ground_truth": "Eligible" 
}

# DATA GENERATION: Create the synthetic population for Stage 2
@st.cache_data # Caches this so it doesn't reload every click
def generate_data():
    np.random.seed(42)
    n_applicants = 1000
    true_eligibility = np.random.choice([0, 1], size=n_applicants, p=[0.3, 0.7])
    ai_scores = []
    for is_eligible in true_eligibility:
        if is_eligible == 1:
            score = np.random.normal(75, 15)
        else:
            score = np.random.normal(40, 20)
        ai_scores.append(np.clip(score, 0, 100))
    
    return pd.DataFrame({"eligible": true_eligibility, "score": ai_scores})

# Load the data
df = generate_data()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def create_gauge(value):
    """Creates the Red-to-Green Needle Gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI Probability of Eligibility"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcccb"},
                {'range': [50, 80], 'color': "#ffffcc"},
                {'range': [80, 100], 'color': "#90ee90"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': value}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def calculate_tradeoffs(threshold, include_social_data):
    """Calculates FP/FN based on student sliders"""
    adjusted_scores = df['score'].copy()
    
    if not include_social_data:
        # Adding noise to simulate lower accuracy without data
        noise = np.random.normal(0, 15, size=len(df))
        adjusted_scores = adjusted_scores + noise
    
    # Prediction logic
    predicted_approved = adjusted_scores >= threshold
    
    fp = len(df[(predicted_approved == 1) & (df['eligible'] == 0)])
    fn = len(df[(predicted_approved == 0) & (df['eligible'] == 1)])
    
    return fp, fn

# ==========================================
# 4. MAIN APP INTERFACE
# ==========================================

st.title("⚖️ The Black Box: Algorithms in Public Law")
st.markdown("Welcome to the **Benefits Allocation Simulator**.")

# TABS
tab1, tab2, tab3 = st.tabs(["1. The Human Assist", "2. Tuning the Model", "3. System Design"])

# --- STAGE 1 ---
with tab1:
    st.header("Stage 1: Automation Bias")
    st.write("Review the application. The AI has provided a risk score based on historical data.")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.info(f"**Name:** {CASE_STUDY['name']}")
        st.markdown(CASE_STUDY['details'])
    with c2:
        st.plotly_chart(create_gauge(CASE_STUDY['ai_risk_score']), use_container_width=True)

    decision = st.radio("Your Ruling:", ["Select...", "Approve", "Deny"], horizontal=True)
    
    if decision != "Select...":
        if st.button("Reveal Outcome"):
            result_color = "green" if CASE_STUDY['ground_truth'] == "Eligible" else "red"
            st.markdown(f"### Ground Truth: :{result_color}[{CASE_STUDY['ground_truth']}]")
            
            if (decision == "Approve" and CASE_STUDY['ground_truth'] == "Ineligible") or \
               (decision == "Deny" and CASE_STUDY['ground_truth'] == "Eligible"):
                st.error("You made an error regarding the ground truth.")
            else:
                st.success("Your decision matched the ground truth.")

# --- STAGE 2 ---
with tab2:
    st.header("Stage 2: The Sensitivity Tradeoff")
    
    col_controls, col_viz = st.columns([1, 2])
    with col_controls:
        st.subheader("Model Parameters")
        threshold = st.slider("Approval Score Threshold", 0, 100, 50)
        social_data = st.checkbox("Include Social Media Data", value=True)
        
    with col_viz:
        fp, fn = calculate_tradeoffs(threshold, social_data)
        
        st.subheader("Results per 1,000 Applicants")
        m1, m2 = st.columns(2)
        m1.metric("False Negatives (Harm)", fn, delta_color="inverse")
        m2.metric("False Positives (Waste)", fp, delta_color="inverse")
        
        # Simple Bar Chart
        chart_data = pd.DataFrame({
            "Error Type": ["False Negative (Harm)", "False Positive (Waste)"],
            "Count": [fn, fp]
        })
        st.bar_chart(chart_data, x="Error Type", y="Count")

# --- STAGE 3 ---
with tab3:
    st.header("Stage 3: Policy Design")
    
    appeal = st.checkbox("Add Human Appeal Process")
    auto = st.checkbox("Auto-Approve High Scores")
    
    wait_time = 2
    trust = 50
    budget = 100
    
    if appeal:
        wait_time += 4
        trust += 20
        budget += 30
    if auto:
        wait_time -= 1
        trust -= 10
        budget -= 10
        
    st.write("### System Projections")
    st.caption("Wait Time (Weeks)")
    st.progress(min(wait_time/10, 1.0))
    
    st.caption("Public Trust Score")
    st.progress(min(trust/100, 1.0))
    
    st.caption("Budget Utilized")
    st.progress(min(budget/200, 1.0))
