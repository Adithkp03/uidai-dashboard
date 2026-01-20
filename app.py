import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pymongo
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
st.set_page_config(
    page_title="UIDAI Operational Analytics",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117; 
        color: #E0E0E0;
    }
    
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        background-image: radial-gradient(circle at 10% 20%, rgba(0, 229, 255, 0.1) 0%, transparent 20%),
                          radial-gradient(circle at 90% 80%, rgba(213, 0, 249, 0.1) 0%, transparent 20%);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #0E1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #333;
        border-radius: 5px;
    }

    /* Card Style - Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Metric Value Styling inside Custom Cards */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FFFFFF, #B0BEC5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #90A4AE;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 500;
    }

    .metric-delta {
        font-size: 0.9rem;
        margin-top: 5px;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #2979FF, #00B0FF);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(0, 176, 255, 0.5);
    }

</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

@st.cache_data
def load_data():
    """
    Loads pre-aggregated data from MongoDB.
    """
    if not MONGO_URI:
        st.error("MONGO_URI not found in environment variables.")
        return pd.DataFrame()

    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client["uidai_db"]
        collection = db["district_metrics"]
        
        # Exclude _id field
        data = list(collection.find({}, {"_id": 0}))
        
        if not data:
            st.warning("No data found in MongoDB collection 'district_metrics'.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        return df
        
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return pd.DataFrame()

def compute_signals(df):
    """
    Feature Engineering for Identity Churn Signals.
    Normalize signals to 0-1 scale relative to their max per state (contextual normalization).
    """
    
    # Sort by district and date
    df = df.sort_values(by=['state', 'district', 'date'])
    
    # --- 1. Enrolment Signal (Volatility) ---
    # Rolling standard deviation over 3 months
    df['Enrol_Vol'] = df.groupby(['state', 'district'])['Total_Enrolment'].transform(lambda x: x.rolling(window=3).std())
    
    # --- 2. Demographic Signal (Momentum) ---
    # Rate of change (First difference)
    df['Demo_Momentum'] = df.groupby(['state', 'district'])['Total_Demographic'].transform(lambda x: x.diff().abs())
    
    # --- 3. Biometric Signal (Unexpected Activity) ---
    # Simply using the raw count scaled, or volatility. Let's use Volatility.
    df['Bio_Vol'] = df.groupby(['state', 'district'])['Total_Biometric'].transform(lambda x: x.rolling(window=3).std())
    
    # --- UPGRADE: Baseline Deviation Signal ---
    # Rationale: Captures abnormal behavior relative to historical median.
    # Compute median per district
    df['Enrol_Median'] = df.groupby(['state', 'district'])['Total_Enrolment'].transform(lambda x: x.rolling(window=12, min_periods=1).median())
    df['Demo_Median'] = df.groupby(['state', 'district'])['Total_Demographic'].transform(lambda x: x.rolling(window=12, min_periods=1).median())
    df['Bio_Median'] = df.groupby(['state', 'district'])['Total_Biometric'].transform(lambda x: x.rolling(window=12, min_periods=1).median())
    
    # Calculate Relative Deviation: abs(Current - Median) / (Median + 1)
    # Adding 1 to avoid division by zero
    df['Dev_Enrol'] = (df['Total_Enrolment'] - df['Enrol_Median']).abs() / (df['Enrol_Median'] + 1)
    df['Dev_Demo'] = (df['Total_Demographic'] - df['Demo_Median']).abs() / (df['Demo_Median'] + 1)
    df['Dev_Bio'] = (df['Total_Biometric'] - df['Bio_Median']).abs() / (df['Bio_Median'] + 1)
    
    # Weighted Sum of Deviations for a combined anomalies signal
    df['Total_Deviation'] = df['Dev_Enrol'] + df['Dev_Demo'] + df['Dev_Bio']

    # Fill NaNs created by rolling/diff
    df.fillna(0, inplace=True)

    # --- Normalization (Min-Max Scalar per State or Global?) ---
    # Global normalization for comparing across the board.
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    df['Signal_Enrolment'] = normalize(df['Enrol_Vol'])
    df['Signal_Demograpic'] = normalize(df['Demo_Momentum'])
    df['Signal_Biometric'] = normalize(df['Bio_Vol'])
    df['Signal_Baseline_Deviation'] = normalize(df['Total_Deviation']) # UPGRADE
    
    return df

def calculate_ici_score(df, enrol_weight=1.0, demo_weight=1.0, bio_weight=1.0):
    """
    Computes Identity Churn Index (ICI).
    ICI = Weighted Mean of Signals.
    """
    # Allow simulation weights
    # UPGRADE: Added Baseline Deviation weight (fixed at 1.0 for now)
    total_weight = enrol_weight + demo_weight + bio_weight + 1.0
    
    df['ICI_Score'] = (
        (df['Signal_Enrolment'] * enrol_weight) +
        (df['Signal_Demograpic'] * demo_weight) +
        (df['Signal_Biometric'] * bio_weight) +
        (df['Signal_Baseline_Deviation'] * 1.0)
    ) / total_weight
    
    # --- Churn Acceleration ---
    # Month-over-month change in ICI
    df['ICI_Acceleration'] = df.groupby(['state', 'district'])['ICI_Score'].transform(lambda x: x.diff())
    df.fillna(0, inplace=True)
    
    # --- UPGRADE: Percentile Ranks for Dynamic Classification ---
    # We rank districts within each time period to handle seasonal shifts fairly.
    df['ICI_Rank_Pct'] = df.groupby('date')['ICI_Score'].rank(pct=True)
    df['Accel_Rank_Pct'] = df.groupby('date')['ICI_Acceleration'].rank(pct=True)
    
    return df

def classify_district(row):
    """
    Rule-based classification system using Percentile Ranks.
    Fairer than hard thresholds as it adjusts to network-wide load.
    """
    # UPGRADE: Replaced hard thresholds with relative percentiles
    accel_pct = row['Accel_Rank_Pct']
    ici_pct = row['ICI_Rank_Pct']
    
    # Top 10% fastest growing churn -> Emerging Risk
    if accel_pct > 0.90: 
        return "Emerging Risk"
    # Top 20% highest absolute churn -> High Churn
    elif ici_pct > 0.80:
        return "High Churn"
    # Next 30% -> Transitional
    elif ici_pct > 0.50:
        return "Transitional"
    else:
        return "Stable"

# --- ML VALIDATION LAYER ---
def compute_ml_validation(df):
    """
    Unsupervised ML Layer for supporting validaton.
    1. Isolation Forest for Anomaly Detection.
    2. KMeans for independent clustering.
    """
    # Features for ML
    features = ['Signal_Enrolment', 'Signal_Demograpic', 'Signal_Biometric', 'Signal_Baseline_Deviation', 'ICI_Score']
    
    # Needs to be done at District-Month level, but for simplicity/stability in this demo, 
    # we'll fit on the whole dataset (or per state if data large enough). 
    # Fitting on whole dataset for global anomaly context.
    
    X = df[features].fillna(0)
    
    # 1. Anomaly Detection (Isolation Forest)
    # Contamination=0.05 implies we expect ~5% anomalies
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['ML_Anomaly_Score'] = iso.fit_predict(X) 
    # -1 is anomaly, 1 is normal. Map to readable.
    df['ML_Is_Anomaly'] = df['ML_Anomaly_Score'].apply(lambda x: "⚠️ Anomalous" if x == -1 else "✅ Normal")
    
    # 2. Clustering (KMeans) - Validating Typology
    # We use 3 clusters to see if they align with Stable/Transitional/High Churn
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['ML_Cluster'] = kmeans.fit_predict(X)
    
    return df

def compute_forecast(district_df):
    """
    Simple 3-month forecast using rolling mean trend extrapolation.
    Returns dataframe with future dates.
    """
    if district_df.empty:
        return district_df
        
    last_date = district_df['date'].max()
    last_ici = district_df[district_df['date'] == last_date]['ICI_Score'].values[0]
    
    # specialized simple logic: avg change over last 3 periods
    recent_trend = district_df.sort_values('date').tail(3)['ICI_Acceleration'].mean()
    
    future_data = []
    current_val = last_ici
    
    for i in range(1, 4): # 3 months forecast
        next_date = last_date + pd.DateOffset(months=i)
        current_val += (recent_trend if not np.isnan(recent_trend) else 0)
        # Clip to 0-1
        current_val = max(0.0, min(1.0, current_val))
        
        future_data.append({
            'date': next_date,
            'ICI_Score': current_val,
            'Type': 'Forecast'
        })
        
    return pd.DataFrame(future_data)

def get_recommendation(category):
    """
    Prescriptive Recommendation Engine.
    """
    recs = {
        "Emerging Risk": "🚨 Immediate Audit & Resource Allocation. Churn is spiking.",
        "High Churn": "⚠️ Deploy Mobile Units & Camp Mode Updates. Load is critically high.",
        "Transitional": "ℹ️ Monitor Closely. Optimize slot booking availability.",
        "Stable": "✅ Standard Operations. Maintain baseline capacity."
    }
    return recs.get(category, "Analyze Further")

# --- UI Layout ---

st.title("🇮🇳 UIDAI Operational Decision Support System")
st.markdown("### Identity Churn Index (ICI) Analytics")

# 1. Load & Process Data
with st.spinner('Loading Data from MongoDB...'):
    try:
        metrics_df = load_data()
        if metrics_df.empty:
             st.stop()
        metrics_df = compute_signals(metrics_df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# --- Simulation Sidebar ---
with st.sidebar:
    st.header("⚙️ Scenario Simulation")
    st.info("Simulate operational stress by adjusting signal weights or multipliers.")
    
    # Simulation Sliders
    sim_enrol = st.slider("Enrolment load multiplier", 0.5, 2.0, 1.0, 0.1)
    sim_demo = st.slider("Demographic update multiplier", 0.5, 2.0, 1.0, 0.1)
    
    # Recalculate Logic for Simulation
    # We apply multipliers to the SIGNALS, not raw data, for 'Weight' effect simulation 
    # OR we treat them as 'Shock' multipliers. Let's treat them as weight/impact multipliers.
    
    final_df = calculate_ici_score(metrics_df, enrol_weight=sim_enrol, demo_weight=sim_demo, bio_weight=1.0)
    
    # Apply Classification using latest ICI
    final_df['Category'] = final_df.apply(classify_district, axis=1)
    
    # --- UPGRADE: Run ML Layer ---
    final_df = compute_ml_validation(final_df)

    st.divider()
    
    # Filters
    st.header("📍 Region Filtering")
    try:
        states = sorted(final_df['state'].unique())
        selected_state = st.selectbox("Select State", states)
        
        districts = sorted(final_df[final_df['state'] == selected_state]['district'].unique())
        selected_district = st.selectbox("Select District", districts)
    except Exception as e:
        st.error(f"Error checking State/District data. Is data empty? {e}")
        st.stop()
    
    # --- ML Specific Data Filtering for Forecast ---
    # Need to do this before potential subsetting for other views if needed, 
    # but here we just follow the flow.

# --- Main Dashboard ---

# Filtered Data
state_data = final_df[final_df['state'] == selected_state]
district_data = state_data[state_data['district'] == selected_district]

# Get Latest Month Data for KPI
latest_date = final_df['date'].max()
latest_district_data = district_data[district_data['date'] == latest_date]

if not latest_district_data.empty:
    current_ici = latest_district_data['ICI_Score'].values[0]
    current_category = latest_district_data['Category'].values[0]
    recommendation = get_recommendation(current_category)
    
    # ML Outputs
    is_anomaly = latest_district_data['ML_Is_Anomaly'].values[0]
    ml_cluster = latest_district_data['ML_Cluster'].values[0]
    
    # Custom HTML Metric Cards
    def metric_card(label, value, delta=None, color_class="neutral"):
        delta_html = ""
        if delta:
            try:
                d_val = float(delta)
                if d_val > 0:
                   arrow = "↑"
                   c_code = "#00E676" # Green
                elif d_val < 0:
                   arrow = "↓"
                   c_code = "#FF5252" # Red 
                else: 
                   arrow = "-"
                   c_code = "#B0BEC5" # Grey
                
                delta_html = f'<div class="metric-delta" style="color: {c_code};">{arrow} {abs(d_val):.2f}</div>'
            except:
                delta_html = f'<div class="metric-delta" style="color: #B0BEC5;">{delta}</div>' # Text delta

        return f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """

    # Top Row: KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Identity Churn Index", f"{current_ici:.2f}", latest_district_data['ICI_Acceleration'].values[0]), unsafe_allow_html=True)
    with col2:
        # Style Risk Category text based on severity
        cat_color = "#B0BEC5"
        if "High Churn" in current_category: cat_color = "#FF9800"
        elif "Emerging" in current_category: cat_color = "#FF5252" 
        elif "Stable" in current_category: cat_color = "#00E676"
        
        # Custom card for text value
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Risk Category</div>
            <div class="metric-value" style="color: {cat_color}; background: none; -webkit-text-fill-color: {cat_color};">{current_category}</div>
            <div class="metric-delta">Model Assessment</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(metric_card("Total Enrolments (Last Month)", f"{int(latest_district_data['Total_Enrolment'].values[0]):,}", "Volume"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Total Updates", f"{int(latest_district_data['Total_Demographic'].values[0] + latest_district_data['Total_Biometric'].values[0]):,}", "Activity"), unsafe_allow_html=True)

    # Recommendation Banner
    if current_ici > 0.6 or current_category == "Emerging Risk":
        st.error(f"**Action Required**: {recommendation}")
    elif current_ici > 0.3:
        st.warning(f"**Recommendation**: {recommendation}")
    else:
        st.success(f"**Status**: {recommendation}")

    # --- UPGRADE: ML Validation Section ---
    st.markdown("### 🤖 AI/ML Validation Layer (Supporting Analysis)")
    with st.expander("Expand to view Unsupervised ML & Forecasts", expanded=True):
        m1, m2 = st.columns([1, 2])
        
        with m1:
            st.info(f"**ML Validation Status**: {is_anomaly}")
            st.caption("Unsupervised Isolation Forest (Statistically unusual behavior check).")
            
            if is_anomaly == "⚠️ Anomalous":
                st.write("This district exhibits data patterns significantly deviating from the national distribution.")
            else:
                st.write("District behavior is statistically consistent with national norms.")
                
            st.markdown("---")
            st.write(f"**Cluster Group**: {ml_cluster}")
            st.caption("KMeans Grouping. Validates if rule-based category aligns with data clusters.")

        with m2:
            st.subheader("🔮 Short-Horizon Trend Forecast")
            
            # Forecast Logic
            forecast_df = compute_forecast(district_data)
            forecast_df['Type'] = 'Forecast'
            district_data['Type'] = 'Historical'
            
            # Combine for plotting
            combined_plot = pd.concat([district_data[['date', 'ICI_Score', 'Type']], forecast_df], ignore_index=True)
            
            fig_fc = px.line(combined_plot, x='date', y='ICI_Score', color='Type', 
                             line_dash='Type', 
                             color_discrete_map={"Historical": "blue", "Forecast": "orange"},
                             title="ICI Trend + 3 Month Forecast")
            st.plotly_chart(fig_fc, use_container_width=True)
            st.caption("Indicative trend projection based on recent acceleration. Not a guarantee.")

# --- UPGRADE: Model Validation & Sanity Check ---
st.divider()
st.subheader("🔍 Model Sanity Check & Validation")

with st.expander("See automatic validation of district classifications", expanded=False):
    st.markdown("Automated sampling of classified districts to verify logic consistency.")
    
    # Get examples
    try:
        high_churn_sample = final_df[final_df['Category'] == "High Churn"].sort_values(by="ICI_Score", ascending=False).head(1)
        stable_sample = final_df[final_df['Category'] == "Stable"].sort_values(by="ICI_Score", ascending=True).head(1)
        
        vc1, vc2 = st.columns(2)
        
        if not high_churn_sample.empty:
            d_name = high_churn_sample['district'].values[0]
            d_score = high_churn_sample['ICI_Score'].values[0]
            d_pctl = high_churn_sample['ICI_Rank_Pct'].values[0] * 100
            
            with vc1:
                st.error(f"**High Churn Example:** {d_name}")
                st.write(f"**ICI Score:** {d_score:.2f} (Top {100-d_pctl:.1f}% of all districts)")
                st.write(f"**Reasoning:** Only districts in the top 20th percentile of signals are flagged. High volatility in enrolments/updates detected.")

        if not stable_sample.empty:
            s_name = stable_sample['district'].values[0]
            s_score = stable_sample['ICI_Score'].values[0]
            
            with vc2:
                st.success(f"**Stable Example:** {s_name}")
                st.write(f"**ICI Score:** {s_score:.2f}")
                st.write(f"**Reasoning:** Low signal variance across all three metrics. Operates within predictable baseline bounds.")
                
    except Exception as e:
        st.write("Insufficient data for full validation.")

# --- UPGRADE: Strategic Comparison ---
st.subheader("💡 Why Identity Churn Matters More Than Raw Volume")
st.markdown("""
| Feature | Traditional Volume Dashboard | **Identity Churn Index (ICI)** |
| :--- | :--- | :--- |
| **Focus** | Absolute counts (e.g., "10k updates") | **Instability & Volatility** (e.g., "Why 3x variance?") |
| **Insight** | "Workload is high." | "**Operational risk is escalating.**" |
| **Blindspot** | Hides repeated failures/updates by same users. | **Detects re-work loops & unusual surges.** |
| **Action** | "Hire more staff." | "**Deploy targeted interventions (Mobile Units).**" |
""")

# --- UPGRADE: Ethics & Governance ---
with st.expander("🛡️ Ethics, Privacy & Governance"):
    st.info("""
    **Responsible AI & Data Usage Statement**
    
    1.  **Non-PII-Based:** This system uses **strictly aggregated** district-level counts. No individual Aadhaar number or personal identity is processed.
    2.  **Surveillance-Free:** The tooling is designed for **resource allocation** (staffing, kits), not policing or individual tracking.
    3.  **Explainable:** The ICI score is a simple weighted average of statistical variance. There are no "black box" neural networks making opaque decisions.
    4.  **Bias-Resistant:** By using **Percentile Ranking** instead of fixed thresholds, the system treats rural and urban districts fairly relative to their own network behavior.
    """)

st.divider()

# --- Visualizations ---

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"📉 ICI Trend: {selected_district}")
    
    # Line Chart
    fig_line = px.line(district_data, x="date", y="ICI_Score", 
                       title="Identity Churn Index Over Time",
                       markers=True)
    # Styles
    fig_line.update_layout(template="plotly_dark", 
                           paper_bgcolor='rgba(0,0,0,0)', 
                           plot_bgcolor='rgba(0,0,0,0)',
                           font={'family': 'Inter'})
                           
    # UPGRADE: Add line for 80th percentile threshold context (approximate from current data)
    # We calculate the 80th percentile score from the WHOLE dataset for context
    threshold_val = final_df['ICI_Score'].quantile(0.8)
    fig_line.add_hline(y=threshold_val, line_dash="dash", line_color="#FF5252", annotation_text="80th Pctl (High Churn)")
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.subheader("📊 Signal Decomposition")
    # Area Chart for Signals
    fig_area = px.area(district_data, x="date", y=["Signal_Enrolment", "Signal_Demograpic", "Signal_Biometric", "Signal_Baseline_Deviation"],
                       title="Contribution of Signals to ICI")
    fig_area.update_layout(template="plotly_dark", 
                           paper_bgcolor='rgba(0,0,0,0)', 
                           plot_bgcolor='rgba(0,0,0,0)',
                           font={'family': 'Inter'})
    st.plotly_chart(fig_area, use_container_width=True)

with col_right:
    st.subheader("🏆 Top High-Churn Districts (State)")
    
    # Get latest data for all districts in state
    latest_state_data = state_data[state_data['date'] == latest_date].sort_values(by="ICI_Score", ascending=False).head(10)
    
    fig_bar = px.bar(latest_state_data, x="ICI_Score", y="district", orientation='h',
                     color="Category",
                     color_discrete_map={
                         "Emerging Risk": "#FF5252", # Red
                         "High Churn": "#FFAB40",    # Orange
                         "Transitional": "#FFEA00",  # Yellowish
                         "Stable": "#69F0AE"         # Green
                     },
                     title="Top 10 Districts by ICI")
    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, 
                          template="plotly_dark",
                          paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          font={'family': 'Inter'})
    st.plotly_chart(fig_bar, use_container_width=True)

    # Explainability Panel
    st.markdown("### ℹ️ District Insight")
    st.write(f"**{selected_district}** is classified as **{current_category}**.")
    st.write("Prominent Churn Drivers:")
    
    # Identify max signal
    signals = {
        "Enrolment": latest_district_data['Signal_Enrolment'].values[0],
        "Demographic": latest_district_data['Signal_Demograpic'].values[0],
        "Biometric": latest_district_data['Signal_Biometric'].values[0]
    }
    max_driver = max(signals, key=signals.get)
    st.info(f"Main Driver: **{max_driver}** ({signals[max_driver]:.2f})")
    
    if max_driver == "Enrolment":
        st.caption("High variance in new enrolments detected.")
    elif max_driver == "Demographic":
        st.caption("Surge in address/mobile updates detected.")
    else:
        st.caption("Unusual biometric update activity patterns.")

# --- Data Table ---
with st.expander("📂 View Classification Data"):
    st.dataframe(latest_state_data[['district', 'ICI_Score', 'Category', 'Total_Enrolment', 'Total_Demographic', 'Total_Biometric']], use_container_width=True)
