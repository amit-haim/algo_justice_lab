import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ==========================================
# 🔧 BACKEND CONFIGURATION (EDIT THIS AREA)
# ==========================================

# 1. THE CASE STUDY TEXT
# This is the specific case students review in Stage 1.
CASE_STUDY = {
    "name": "Applicant #402: 'Sarah J.'",
    "details": """
    **Age:** 45  
    **Occupation:** Former Warehouse Logistics Manager  
    **Medical Claim:** Chronic Lumbar Radiculopathy (Nerve damage in back).  
    **Evidence:** MRI shows moderate degeneration. Subject reports 'severe pain' preventing lifting >10lbs.  
    **Social Data:** Posted photos of a hiking trip 6 months ago (before latest surgery).
    """,
    "ai_risk_score": 72,  # 0 to 100. Higher = More likely to be legitimate/approved.
    "ground_truth": "Eligible" # Used for the 'Reveal' button
}

# 2. THE SYNTHETIC POPULATION (For Stage 2 Statistics)
# This simulates 1000 applicants so students can see "Big Picture" stats.
np.random.seed(42)
n_applicants = 1000
# Generate random "True Eligibility" (30% are truly fraudulent/ineligible, 70% are eligible)
true_eligibility = np.random.choice([0, 1], size=n_applicants, p=[0.3, 0.7])
# Generate AI scores. If eligible, score is higher on average, but with noise.
ai_scores = []
for is_eligible in true_eligibility:
    if is_eligible == 1:
        score = np.random.normal(75, 15) # Eligible people usually score high
    else:
        score = np.random.normal(40, 20) # Ineligible people usually score low
    ai_scores.append(np.clip(score, 0, 100))

df = pd.DataFrame({"eligible": true_eligibility, "score": ai_scores})

# ==========================================
# 🎨 UI & VISUALIZATION FUNCTIONS
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
                {'range': [0, 50], 'color': "#ffcccb"},  # Red/Pink (Low probability)
                {'range': [50, 80], 'color': "#ffffcc"}, # Yellow (Uncertain)
                {'range': [80, 100], 'color': "#90ee90"} # Green (High probability)
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def calculate_tradeoffs(threshold, include_social_data):
    """Calculates FP/FN based on student sliders"""
    
    # Simulate "Noise" if social data is excluded (Model becomes less accurate)
    adjusted_scores = df['score'] 
    if not include_social_data:
        # Adding random noise effectively lowers accuracy
        noise = np.random.normal(0, 15, size=n_applicants)
        adjusted_scores = adjusted_scores + noise
    
    # Determine who gets approved based on the threshold
    df['predicted_approved'] = adjusted_scores >= threshold
    
    # Calculate Confusion Matrix
    fp = len(df[(df['predicted_approved'] == 1) & (df['eligible'] == 0)]) # Fiscal Waste
    fn = len(df[(df['predicted_approved'] == 0) & (df['eligible'] == 1)]) # Human Suffering
    tp = len(df[(df['predicted_approved'] == 1) & (df['eligible'] == 1)])
    tn = len(df[(df['predicted_approved'] == 0) & (df['eligible'] == 0)])
    
    return fp, fn, tp, tn

# ==========================================
# 🚀 MAIN APP LAYOUT
# ==========================================

st.set_page_config(page_title="Algorithmic Justice Lab", layout="wide")

st.title("⚖️ The Black Box: Algorithms in Public Law")
st.markdown("""
Welcome to the Benefits Allocation Simulator. You will experience the tradeoffs inherent in automating legal decisions.
""")

# TABS FOR STAGES
tab1, tab2, tab3 = st.tabs(["Stage 1: The Human-in-the-Loop", "Stage 2: Tuning the Model", "Stage 3: System Design"])

# --- STAGE 1: THE ASSIST ---
with tab1:
    st.header("Stage 1: Automation Bias")
    st.write("Review the application below. The AI has provided a preliminary risk score.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Application File")
        st.info(f"**Name:** {CASE_STUDY['name']}")
        st.markdown(CASE_STUDY['details'])
        
    with col2:
        st.subheader("🤖 AI Recommendation")
        st.plotly_chart(create_gauge(CASE_STUDY['ai_risk_score']), use_container_width=True)
        st.caption("The algorithm compares this applicant to 10,000 historical cases.")

    st.divider()
    
    decision = st.radio("What is your ruling?", ["Select...", "Approve Benefits", "Deny Benefits"], horizontal=True)
    
    if decision != "Select...":
        if st.button("Submit Decision & Reveal Truth"):
            if decision == "Approve Benefits":
                st.success(f"You Approved. The Ground Truth was: **{CASE_STUDY['ground_truth']}**.")
            else:
                st.error(f"You Denied. The Ground Truth was: **{CASE_STUDY['ground_truth']}**.")
            
            st.markdown("""
            **Discussion Question:** Did the green needle make you feel safer approving the request? 
            If the needle had been red (20%), would you have denied it despite the medical evidence?
            """)

# --- STAGE 2: MODEL TUNING ---
with tab2:
    st.header("Stage 2: The Sensitivity Tradeoff")
    st.markdown("You are now the **System Administrator**. You control how strict the algorithm is.")
    
    col_controls, col_viz = st.columns([1, 2])
    
    with col_controls:
        st.subheader("🎛️ Model Parameters")
        threshold = st.slider("Approval Threshold (%)", 0, 100, 50, 
                              help="Higher = Stricter. Applicants need a higher score to get approved.")
        
        social_data = st.checkbox("Include 'Alternative Data' (Social Media, etc.)", value=True,
                                  help="Including this data increases accuracy but may introduce privacy/bias concerns.")
        
        st.markdown("---")
        st.markdown("**Your Policy Stance:**")
        if threshold > 70:
            st.warning("Strict: Protecting the Budget")
        elif threshold < 30:
            st.success("Lenient: Protecting the Vulnerable")
        else:
            st.info("Balanced Approach")

    with col_viz:
        fp, fn, tp, tn = calculate_tradeoffs(threshold, social_data)
        
        # VISUAL: CONFUSION MATRIX
        st.subheader("📊 Real-time Consequences (Per 1,000 Applicants)")
        
        col_metrics1, col_metrics2 = st.columns(2)
        with col_metrics1:
            st.metric("False Negatives (Wrongful Denials)", f"{fn}", delta="- Human Cost", delta_color="inverse")
            st.markdown("*People who need help but were rejected.*")
        with col_metrics2:
            st.metric("False Positives (Fraud/Waste)", f"{fp}", delta="- Fiscal Cost", delta_color="inverse")
            st.markdown("*People who don't qualify but got paid.*")
            
        # Simple Bar Chart
        error_df = pd.DataFrame({
            "Error Type": ["Wrongful Denial (Human Cost)", "Wrongful Approval (Fiscal Cost)"],
            "Count": [fn, fp],
            "Color": ["red", "orange"]
        })
        fig_bar = px.bar(error_df, x="Error Type", y="Count", color="Error Type", 
                         color_discrete_map={"Wrongful Denial (Human Cost)":"#ff4b4b", "Wrongful Approval (Fiscal Cost)":"#ffa500"})
        st.plotly_chart(fig_bar, use_container_width=True)

# --- STAGE 3: SYSTEM DESIGN ---
with tab3:
    st.header("Stage 3: Designing the Human Layer")
    st.write("Algorithms don't exist in a vacuum. Add human safeguards and see the cost.")
    
    # INITIAL VALUES
    base_wait = 2 # weeks
    base_trust = 50 # percent
    base_budget = 100 # index
    
    c1, c2, c3 = st.columns(3)
    with c1:
        human_appeal = st.checkbox("Mandatory Human Appeal for Denials")
    with c2:
        auto_approve = st.checkbox("Auto-Approve 'High Confidence' (>90%)")
    with c3:
        audit = st.checkbox("Third-Party Fairness Audit")
        
    # LOGIC ENGINE
    if human_appeal:
        base_wait += 12
        base_trust += 20
        base_budget += 30
    
    if auto_approve:
        base_wait -= 1
        base_budget -= 10
        # Trust hit because people fear "black box" approvals lacking oversight
        base_trust -= 5 
        
    if audit:
        base_trust += 15
        base_budget += 10
        base_wait += 1

    st.markdown("### Projected System Outcomes")
    
    # DISPLAY METRICS
    st.subheader(f"⏱️ Avg Wait Time: {base_wait} Weeks")
    st.progress(min(base_wait/20, 1.0))
    
    st.subheader(f"🤝 Public Trust Index: {base_trust}/100")
    st.progress(min(base_trust/100, 1.0))
    
    st.subheader(f"💰 Administrative Cost: {base_budget} (Index)")
    st.progress(min(base_budget/200, 1.0))
    
    st.info("Note: Notice how adding 'Fairness' mechanisms (Appeals/Audits) almost always increases Cost and Wait Times.")
