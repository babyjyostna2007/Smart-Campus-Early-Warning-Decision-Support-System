import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Smart Campus Early-Warning System",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# FILES
# =========================================================

MODEL_FILE = "risk_model.pkl"
DATASET_FILE = "smart_campus.csv"

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

/* =========================================================
   SMART CAMPUS — GREEN + MINT + CREAM LIGHT THEME
   ========================================================= */

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(125, 211, 168, 0.18), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(159, 214, 207, 0.20), transparent 30%),
        linear-gradient(135deg, #FFFDF5 0%, #F2FAF4 52%, #ECF8F2 100%);
    color: #244238;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Main text */
h1, h2, h3, h4, h5, h6 {
    color: #174C3A !important;
}

p, label, .stMarkdown, .stCaption {
    color: #587067;
}

/* Hero */
.hero {
    padding: 38px;
    border-radius: 26px;
    text-align: center;
    color: white;
    background:
        linear-gradient(135deg, rgba(34, 139, 90, 0.96), rgba(21, 154, 156, 0.92));
    border: 1px solid rgba(34, 139, 90, 0.20);
    box-shadow: 0 12px 30px rgba(38, 91, 68, 0.18);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 5px;
    color: #FFFFFF !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.12);
}

/* Metric cards */
.metric-card {
    background: #FFFFFF;
    padding: 13px 10px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #D8E9DF;
    box-shadow: 0 5px 16px rgba(50, 90, 70, 0.08);
}

.metric-title {
    color: #6B8078;
    font-size: 11px;
    font-weight: 600;
}

.metric-value {
    color: #177B5A;
    font-size: 19px;
    font-weight: 800;
}

/* General cards */
.card {
    background: #FFFFFF;
    padding: 15px 18px;
    border-radius: 15px;
    border: 1px solid #D8E9DF;
    box-shadow: 0 5px 18px rgba(50, 90, 70, 0.08);
    margin-bottom: 14px;
}

.card h2 {
    font-size: 20px;
    margin: 0 0 5px 0;
    color: #174C3A !important;
}

.card p {
    font-size: 14px;
    margin: 2px 0;
    color: #60766D;
}

/* Risk cards */
.low-card {
    background: #EAF8EF;
    border-left: 6px solid #228B5A;
    border-radius: 14px;
    padding: 14px 18px;
    color: #174C3A;
}

.medium-card {
    background: #FFF6D9;
    border-left: 6px solid #D99A00;
    border-radius: 14px;
    padding: 14px 18px;
    color: #684F00;
}

.high-card {
    background: #FFECEC;
    border-left: 6px solid #D64545;
    border-radius: 14px;
    padding: 14px 18px;
    color: #7A2525;
}

.low-card h1, .medium-card h1, .high-card h1 {
    color: #174C3A !important;
}

.low-card h2, .medium-card h2, .high-card h2 {
    color: #4E655C !important;
}

/* Buttons */
div.stButton > button {
    width: 100%;
    min-height: 45px;
    border-radius: 12px;
    border: 1px solid #BFDCCF;
    background: #F7FCF8;
    color: #176B4D;
    font-weight: 650;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #E1F4EA;
    border-color: #228B5A;
    color: #12563E;
    box-shadow: 0 0 12px rgba(34, 139, 90, 0.12);
}

/* Tabs */
div[data-testid="stTabs"] button {
    font-weight: 700;
    font-size: 18px;
    padding: 16px 24px;
    min-height: 55px;
    color: #648077;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #177B5A !important;
}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 10px;
    background: #F5FBF7;
    border: 1px solid #D8E9DF;
    border-radius: 14px;
    padding: 6px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F5FBF7 0%, #FFFDF5 100%);
    border-right: 1px solid #D8E9DF;
}

section[data-testid="stSidebar"] hr {
    border-color: #D8E9DF;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    color: #244238 !important;
    border-color: #C9DFD4 !important;
}

div[data-testid="stSlider"] [role="slider"] {
    background: #228B5A !important;
}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {
    background: #F9FCF9;
    border: 1px dashed #A9CFC0;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    border: 1px solid #D8E9DF;
    border-radius: 12px;
}

/* Progress */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #228B5A, #159A9C) !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = None

    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                model = pickle.load(f)
        except:
            model = None

    return model


model = load_model()

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if os.path.exists(DATASET_FILE):
        try:
            return pd.read_csv(DATASET_FILE)
        except:
            return None

    return None


dataset = load_dataset()

records = len(dataset) if dataset is not None else 0

# =========================================================
# AUTO-SCAN DATASET FOR AT-RISK RECORDS
# (no manual input required)
# =========================================================

if dataset is not None and "Risk_Level" in dataset.columns:

    dataset_high_risk = dataset[
        dataset["Risk_Level"].astype(str).str.strip().str.lower() == "high"
    ]

    dataset_medium_risk = dataset[
        dataset["Risk_Level"].astype(str).str.strip().str.lower() == "medium"
    ]

else:

    dataset_high_risk = pd.DataFrame()
    dataset_medium_risk = pd.DataFrame()

dataset_high_risk_count = len(dataset_high_risk)
dataset_medium_risk_count = len(dataset_medium_risk)

if dataset_high_risk_count > 0 and "Risk_Factors" in dataset_high_risk.columns:

    dataset_top_factors = (
        dataset_high_risk["Risk_Factors"]
        .dropna()
        .str.split(";")
        .explode()
        .str.strip()
        .value_counts()
        .head(3)
        .index
        .tolist()
    )

else:

    dataset_top_factors = []

# =========================================================
# SESSION STATE
# =========================================================

if "menu" not in st.session_state:
    st.session_state.menu = "Home"

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
<div style="text-align:center;padding:10px;">

<div style="font-size:45px;">🏫</div>

<h2 style="color:#063b91;margin:0;">
Smart Campus
</h2>

<p style="color:#667085;">
Early-Warning System
</p>

</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# =========================================================
# QUICK INSIGHTS
# =========================================================

st.sidebar.markdown(
    "<h3 style='color:#063b91;'>📌 Quick Insights</h3>",
    unsafe_allow_html=True
)

if st.sidebar.button(
    "🎯 Prediction",
    use_container_width=True
):
    st.session_state.menu = "Prediction"

if st.sidebar.button(
    "🛡️ System",
    use_container_width=True
):
    st.session_state.menu = "System"

# =========================================================
# ANALYTICS
# =========================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "<h3 style='color:#063b91;'>📈 Analytics</h3>",
    unsafe_allow_html=True
)

if st.sidebar.button(
    "📊 Performance Analysis",
    use_container_width=True
):
    st.session_state.menu = "Performance Analysis"

if st.sidebar.button(
    "⚠️ Risk Factor Analysis",
    use_container_width=True
):
    st.session_state.menu = "Risk Factor Analysis"

if st.sidebar.button(
    "🎯 Risk Score",
    use_container_width=True
):
    st.session_state.menu = "Risk Score"

if st.sidebar.button(
    "💡 Recommendations",
    use_container_width=True
):
    st.session_state.menu = "Recommendations"

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🏫 Smart Campus</h1>

<p>
<b>AI-Based Early-Warning Decision Support System</b>
</p>

<p>
📊 Monitor &nbsp; • &nbsp;
🎯 Predict &nbsp; • &nbsp;
⚠️ Prevent &nbsp; • &nbsp;
🛡️ Protect
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# TOP METRICS
# =========================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">📂 Dataset Records</div>
        <div class="metric-value">{records}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-title">🤖 ML Model</div>
        <div class="metric-value">
        {"READY" if model else "ON"}
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="metric-card">
        <div class="metric-title">🎯 Prediction</div>
        <div class="metric-value">LIVE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        """
        <div class="metric-card">
        <div class="metric-title">📊 Analytics</div>
        <div class="metric-value">ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c5:
    st.markdown(
        """
        <div class="metric-card">
        <div class="metric-title">🛡️ System</div>
        <div class="metric-value">ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# =========================================================
# SHARED RESULTS RENDERER
# (Prediction / System / Performance Analysis /
#  Risk Factor Analysis / Risk Score / Recommendations)
# =========================================================

def render_results(page, risk, risk_score, factors, recommendation, performance_data):

    if st.session_state.menu == "Prediction":

        st.markdown(
            f"## 🎯 {page} - Prediction"
        )

        if risk == "HIGH":

            st.error(
                f"🔴 HIGH RISK — Risk Score: {risk_score}%"
            )

        elif risk == "MEDIUM":

            st.warning(
                f"🟡 MEDIUM RISK — Risk Score: {risk_score}%"
            )

        else:

            st.success(
                f"🟢 LOW RISK — Risk Score: {risk_score}%"
            )

    elif st.session_state.menu == "System":

        st.markdown(
            f"## 🛡️ {page} - System"
        )

        # =====================================================
        # NEW ENTRY DETECTION
        # (fires the moment a new user's values are entered
        #  via the sliders — real-time, per-entry check)
        # =====================================================

        st.markdown(
            "### 🆕 New Entry Check"
        )

        if risk == "HIGH":

            st.toast(
                f"🚨 New {page} entry detected — HIGH RISK "
                f"({risk_score}%)",
                icon="🚨"
            )

            st.markdown(
                f"""
                <div style="
                    background:#ffe9e9;
                    border:2px solid #dc2626;
                    padding:14px 18px;
                    border-radius:12px;
                    margin-bottom:14px;
                    animation: pulse 1.5s infinite;
                ">
                <b>🚨 New entry flagged as HIGH RISK</b><br>
                A newly entered {page.lower()} record scored
                {risk_score}%.
                {("<br>Issues found: " + ", ".join(factors)) if factors else ""}
                </div>
                <style>
                @keyframes pulse {{
                    0% {{ box-shadow: 0 0 0 0 rgba(220,38,38,0.4); }}
                    70% {{ box-shadow: 0 0 0 12px rgba(220,38,38,0); }}
                    100% {{ box-shadow: 0 0 0 0 rgba(220,38,38,0); }}
                }}
                </style>
                """,
                unsafe_allow_html=True
            )

        elif risk == "MEDIUM":

            st.toast(
                f"⚠️ New {page} entry shows MEDIUM risk "
                f"({risk_score}%)",
                icon="⚠️"
            )

            st.warning(
                f"🟡 New entry flagged as MEDIUM RISK — "
                f"Score: {risk_score}%."
                + ((" Issues: " + ", ".join(factors)) if factors else "")
            )

        else:

            st.success(
                f"🟢 New entry looks fine — LOW RISK "
                f"({risk_score}%)."
            )

        st.write("")

        s1, s2, s3 = st.columns(3)

        with s1:
            st.metric(
                "Prediction",
                "LIVE"
            )

        with s2:
            st.metric(
                "Analytics",
                "ACTIVE"
            )

        with s3:
            st.metric(
                "System",
                "ALERT" if risk == "HIGH" else "ACTIVE"
            )

        st.write("")
        st.markdown("---")

        # =====================================================
        # UPLOAD NEW FILE FOR PREDICTION
        # (runs the trained ML model on uploaded records)
        # =====================================================

        st.markdown(
            "### 📤 Upload New File for Prediction"
        )

        st.caption(
            "Upload a CSV with student/campus/environmental records "
            "and the trained ML model will predict a risk level for "
            "each row."
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            key=f"system_upload_predict_{page}"
        )

        REQUIRED_FEATURES = [
            "Attendance", "Current_GPA", "Previous_GPA",
            "Assignment_Rate", "Backlogs", "Occupancy_Rate",
            "Electricity_Usage", "Internet_Usage",
            "Maintenance_Complaints", "Temperature", "Humidity",
            "Rainfall", "Water_Level", "Air_Quality_Index"
        ]

        if uploaded_file is not None:

            try:
                new_data = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                new_data = None

            if new_data is not None:

                missing_cols = [
                    c for c in REQUIRED_FEATURES
                    if c not in new_data.columns
                ]

                if missing_cols:

                    st.error(
                        "Uploaded file is missing required column(s): "
                        + ", ".join(missing_cols)
                    )

                elif model is None:

                    st.error(
                        "ML model is not loaded — cannot run "
                        "predictions. Make sure risk_model.pkl is "
                        "in the app folder."
                    )

                else:

                    predictions = model.predict(
                        new_data[REQUIRED_FEATURES]
                    )

                    new_data["Predicted_Risk_Level"] = predictions

                    up_high = int((predictions == "High").sum())
                    up_medium = int((predictions == "Medium").sum())
                    up_low = int((predictions == "Low").sum())

                    if up_high > 0:

                        st.toast(
                            f"🚨 {up_high} uploaded record(s) "
                            f"predicted HIGH RISK",
                            icon="🚨"
                        )

                        st.markdown(
                            f"""
                            <div style="
                                background:#ffe9e9;
                                border:2px solid #dc2626;
                                padding:14px 18px;
                                border-radius:12px;
                                margin:14px 0;
                                animation: pulse 1.5s infinite;
                            ">
                            <b>🚨 {up_high} record(s) flagged HIGH RISK
                            </b><br>
                            out of {len(new_data)} uploaded records.
                            </div>
                            <style>
                            @keyframes pulse {{
                                0% {{ box-shadow: 0 0 0 0 rgba(220,38,38,0.4); }}
                                70% {{ box-shadow: 0 0 0 12px rgba(220,38,38,0); }}
                                100% {{ box-shadow: 0 0 0 0 rgba(220,38,38,0); }}
                            }}
                            </style>
                            """,
                            unsafe_allow_html=True
                        )

                    elif up_medium > 0:

                        st.toast(
                            f"⚠️ {up_medium} uploaded record(s) "
                            f"predicted MEDIUM RISK",
                            icon="⚠️"
                        )

                        st.warning(
                            f"🟡 {up_medium} record(s) out of "
                            f"{len(new_data)} predicted MEDIUM RISK."
                        )

                    else:

                        st.success(
                            f"🟢 All {len(new_data)} uploaded "
                            f"record(s) predicted LOW RISK."
                        )

                    u1, u2, u3 = st.columns(3)

                    with u1:
                        st.metric("High Risk", up_high)

                    with u2:
                        st.metric("Medium Risk", up_medium)

                    with u3:
                        st.metric("Low Risk", up_low)

                    st.dataframe(
                        new_data,
                        use_container_width=True
                    )

                    st.download_button(
                        "⬇️ Download Predictions as CSV",
                        data=new_data.to_csv(index=False).encode("utf-8"),
                        file_name="predicted_results.csv",
                        mime="text/csv",
                        key=f"download_predictions_{page}"
                    )

    elif st.session_state.menu == "Performance Analysis":

        st.markdown(
            f"## 📊 {page} - Performance Analysis"
        )

        if page == "🌦️ Environmental Risk":

            chart = px.bar(
                performance_data,
                x="Parameter",
                y="Current",
                text="Current",
                title="🌦️ Environmental Performance"
            )

        else:

            long_data = performance_data.melt(
                id_vars="Parameter",
                var_name="Type",
                value_name="Value"
            )

            chart = px.bar(
                long_data,
                x="Parameter",
                y="Value",
                color="Type",
                barmode="group",
                text="Value",
                title=f"{page} Performance Analysis"
            )

        chart.update_layout(
            height=450
        )

        st.plotly_chart(
            chart,
            use_container_width=True
        )

    elif st.session_state.menu == "Risk Factor Analysis":

        st.markdown(
            f"## ⚠️ {page} - Risk Factor Analysis"
        )

        # -----------------------------------------------
        # SUMMARY ROW
        # -----------------------------------------------

        f1, f2 = st.columns(2)

        with f1:
            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-title">⚠️ Factors Detected</div>
                <div class="metric-value">{len(factors)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with f2:
            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-title">🎯 Risk Score</div>
                <div class="metric-value">{risk_score}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # -----------------------------------------------
        # DETECTED FACTORS — 2-COLUMN CARD GRID
        # -----------------------------------------------

        if factors:

            st.markdown(
                "### 🔍 Detected Risk Factors"
            )

            grid_cols = st.columns(2)

            for i, factor in enumerate(factors):

                with grid_cols[i % 2]:

                    st.markdown(
                        f"""
                        <div style="
                            background:#fff8df;
                            padding:14px 16px;
                            border-radius:12px;
                            margin-bottom:10px;
                            border-left:5px solid #e0a000;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                        ">
                        <div style="font-size:18px;">⚠️</div>
                        <div style="
                            font-weight:600;
                            color:#7a5b00;
                            margin-top:4px;
                            font-size:14px;
                        ">
                        {factor}
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        else:

            st.markdown(
                """
                <div style="
                    background:#eafaf0;
                    padding:18px;
                    border-radius:12px;
                    border-left:5px solid #16a34a;
                    text-align:center;
                ">
                <div style="font-size:26px;">✅</div>
                <div style="
                    font-weight:700;
                    color:#146c2e;
                    margin-top:6px;
                ">
                No significant risk factors detected
                </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # -----------------------------------------------
        # OVERALL RISK — COLOR CARD + PROGRESS BAR
        # -----------------------------------------------

        st.markdown(
            "### 🎯 Overall Risk"
        )

        if risk == "HIGH":
            card_class = "high-card"
            emoji = "🔴"

        elif risk == "MEDIUM":
            card_class = "medium-card"
            emoji = "🟡"

        else:
            card_class = "low-card"
            emoji = "🟢"

        st.markdown(
            f"""
            <div class="{card_class}">
            <h1>{emoji} {risk} RISK</h1>
            <h2>Risk Score: {risk_score}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(
            risk_score / 100
        )

    elif st.session_state.menu == "Risk Score":

        st.markdown(
            f"## 🎯 {page} - Risk Score"
        )

        if risk == "HIGH":

            st.markdown(
                f"""
                <div class="high-card">

                <h1>🔴 HIGH RISK</h1>

                <h2>
                Risk Score: {risk_score}%
                </h2>

                </div>
                """,
                unsafe_allow_html=True
            )

        elif risk == "MEDIUM":

            st.markdown(
                f"""
                <div class="medium-card">

                <h1>🟡 MEDIUM RISK</h1>

                <h2>
                Risk Score: {risk_score}%
                </h2>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="low-card">

                <h1>🟢 LOW RISK</h1>

                <h2>
                Risk Score: {risk_score}%
                </h2>

                </div>
                """,
                unsafe_allow_html=True
            )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={
                    "text": f"{page} Risk Score"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "steps": [
                        {
                            "range": [0, 40],
                            "color": "#d9f7df"
                        },
                        {
                            "range": [40, 70],
                            "color": "#fff0b8"
                        },
                        {
                            "range": [70, 100],
                            "color": "#ffd6d6"
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=400
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    elif st.session_state.menu == "Recommendations":

        st.markdown(
            f"## 💡 {page} - Recommendations"
        )

        if risk == "HIGH":

            st.error(
                recommendation
            )

        elif risk == "MEDIUM":

            st.warning(
                recommendation
            )

        else:

            st.success(
                recommendation
            )

    else:

        st.markdown("""
        <div class="card">

        <h2>
        👈 Select an option from the sidebar
        </h2>

        <p>
        Choose Prediction, System,
        Performance Analysis, Risk Factor Analysis,
        Risk Score or Recommendations.
        </p>

        <p>
        The selected option will show results
        for this risk category.
        </p>

        </div>
        """, unsafe_allow_html=True)



# =========================================================
# RECOMMENDATION ENGINE
# (builds specific, factor-driven recommendation text
#  instead of generic boilerplate)
# =========================================================

STUDENT_ACTIONS = {
    "attendance": (
        "Schedule a meeting with the student and guardians "
        "to address the attendance shortfall."
    ),
    "current gpa": (
        "Assign a faculty mentor for subject-wise academic support."
    ),
    "previous gpa": (
        "Review historical performance trends with the "
        "academic advisor."
    ),
    "assignment": (
        "Set up a structured assignment-completion plan with "
        "weekly check-ins."
    ),
    "backlog": (
        "Enroll the student in remedial classes to clear "
        "pending backlogs."
    ),
}

CAMPUS_ACTIONS = {
    "attendance": (
        "Investigate causes of low campus-wide attendance and "
        "notify department heads."
    ),
    "infrastructure": (
        "Raise a maintenance work order to inspect and repair "
        "campus infrastructure."
    ),
    "faculty": (
        "Coordinate with HR to address faculty shortages or "
        "scheduling gaps."
    ),
    "safety": (
        "Conduct a health and safety audit across campus facilities."
    ),
    "security": (
        "Increase security patrols and review access control systems."
    ),
    "academic": (
        "Launch academic support programs to improve overall "
        "performance."
    ),
}

ENV_ACTIONS = {
    "temperature": (
        "Activate heat-safety protocols and ensure adequate "
        "cooling and shade on campus."
    ),
    "humidity": (
        "Increase ventilation in indoor spaces to manage high "
        "humidity levels."
    ),
    "rainfall": (
        "Alert the facilities team to check drainage systems "
        "ahead of heavy rainfall."
    ),
    "air quality": (
        "Advise limiting outdoor activities and monitor the AQI "
        "closely."
    ),
    "wind": (
        "Secure loose outdoor structures and issue a "
        "high-wind advisory."
    ),
    "flood": (
        "Coordinate with local authorities on flood-preparedness "
        "measures."
    ),
}


def generate_recommendation(risk, factors, action_map, low_message):

    if risk == "LOW":
        return low_message

    intro = (
        "Immediate action required — "
        if risk == "HIGH"
        else "Preventive action recommended — "
    )

    actions = []

    for factor in factors:

        for keyword, action in action_map.items():

            if keyword.lower() in factor.lower():

                if action not in actions:
                    actions.append(action)

                break

    if not actions:

        actions = [
            "Increase monitoring frequency and review the "
            "relevant performance indicators."
        ]

    return intro + " ".join(actions)


# =========================================================
# RISK CATEGORY TABS
# =========================================================

tab_student, tab_campus, tab_env = st.tabs(
    [
        "👨‍🎓 Student Risk",
        "🏫 Campus Risk",
        "🌦️ Environmental Risk"
    ]
)

# =========================================================
# STUDENT RISK
# =========================================================

with tab_student:

    page = "👨‍🎓 Student Risk"

    risk_score = 0
    risk = "LOW"
    factors = []
    recommendation = ""

    st.markdown("""
    <div class="card">

    <h2>👨‍🎓 Student Risk Prediction</h2>

    <p>
    Enter student academic information.
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.slider(
            "👥 Attendance (%)",
            0,
            100,
            75,
            key="student_attendance"
        )

        current_gpa = st.slider(
            "⭐ Current GPA",
            0.0,
            10.0,
            7.0,
            0.1,
            key="student_current_gpa"
        )

        previous_gpa = st.slider(
            "📚 Previous GPA",
            0.0,
            10.0,
            7.0,
            0.1,
            key="student_previous_gpa"
        )

    with col2:

        assignment_rate = st.slider(
            "📝 Assignment Completion (%)",
            0,
            100,
            75,
            key="student_assignment_rate"
        )

        backlogs = st.number_input(
            "⚠️ Number of Backlogs",
            0,
            20,
            0,
            key="student_backlogs"
        )

    if attendance < 75:
        risk_score += 20
        factors.append("Attendance is below 75%")

    if current_gpa < 7:
        risk_score += 20
        factors.append("Current GPA is below 7")

    if previous_gpa < 7:
        risk_score += 10
        factors.append("Previous GPA is below 7")

    if assignment_rate < 75:
        risk_score += 20
        factors.append("Assignment completion is below 75%")

    if backlogs > 0:
        risk_score += 30
        factors.append(
            f"{int(backlogs)} backlog(s) detected"
        )

    risk_score = min(risk_score, 100)

    performance_data = pd.DataFrame({

        "Parameter": [
            "Attendance",
            "Current GPA",
            "Previous GPA",
            "Assignment Rate"
        ],

        "Current": [
            attendance,
            current_gpa * 10,
            previous_gpa * 10,
            assignment_rate
        ],

        "Normal": [
            75,
            70,
            70,
            75
        ]

    })

    if risk_score <= 20:
        risk = "LOW"

    elif risk_score <= 50:
        risk = "MEDIUM"

    else:
        risk = "HIGH"

    recommendation = generate_recommendation(
        risk,
        factors,
        STUDENT_ACTIONS,
        "Student performance is within the normal range. "
        "Continue regular monitoring."
    )

    render_results(page, risk, risk_score, factors, recommendation, performance_data)

# =========================================================
# CAMPUS RISK
# =========================================================

with tab_campus:

    page = "🏫 Campus Risk"

    risk_score = 0
    risk = "LOW"
    factors = []
    recommendation = ""

    st.markdown("""
    <div class="card">

    <h2>🏫 Campus Risk Prediction</h2>

    <p>
    Enter campus performance information.
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        campus_attendance = st.slider(
            "👥 Student Attendance (%)",
            0,
            100,
            80,
            key="campus_attendance"
        )

        infrastructure = st.slider(
            "🏢 Infrastructure Condition (%)",
            0,
            100,
            85,
            key="campus_infrastructure"
        )

        faculty = st.slider(
            "👨‍🏫 Faculty Availability (%)",
            0,
            100,
            90,
            key="campus_faculty"
        )

    with col2:

        safety = st.slider(
            "🏥 Health & Safety (%)",
            0,
            100,
            90,
            key="campus_safety"
        )

        security = st.slider(
            "🔐 Campus Security (%)",
            0,
            100,
            90,
            key="campus_security"
        )

        academic = st.slider(
            "📚 Academic Performance (%)",
            0,
            100,
            80,
            key="campus_academic"
        )

    campus_names = [
        "Attendance",
        "Infrastructure",
        "Faculty",
        "Health & Safety",
        "Security",
        "Academic"
    ]

    campus_values = [
        campus_attendance,
        infrastructure,
        faculty,
        safety,
        security,
        academic
    ]

    # Convert good performance into risk score
    risk_score = int(
        100 - np.mean(campus_values)
    )

    risk_score = max(
        0,
        min(risk_score, 100)
    )

    if campus_attendance < 75:
        factors.append(
            "Student attendance is below 75%"
        )

    if infrastructure < 70:
        factors.append(
            "Infrastructure condition needs attention"
        )

    if faculty < 70:
        factors.append(
            "Faculty availability is low"
        )

    if safety < 70:
        factors.append(
            "Health and safety score is low"
        )

    if security < 70:
        factors.append(
            "Campus security score is low"
        )

    if academic < 70:
        factors.append(
            "Academic performance is low"
        )

    performance_data = pd.DataFrame({

        "Parameter": campus_names,

        "Current": campus_values,

        "Normal": [
            75,
            75,
            75,
            75,
            75,
            75
        ]

    })

    if risk_score <= 20:
        risk = "LOW"

    elif risk_score <= 40:
        risk = "MEDIUM"

    else:
        risk = "HIGH"

    recommendation = generate_recommendation(
        risk,
        factors,
        CAMPUS_ACTIONS,
        "Campus performance is within an acceptable range."
    )

    render_results(page, risk, risk_score, factors, recommendation, performance_data)

# =========================================================
# ENVIRONMENTAL RISK
# =========================================================

with tab_env:

    page = "🌦️ Environmental Risk"

    risk_score = 0
    risk = "LOW"
    factors = []
    recommendation = ""

    st.markdown("""
    <div class="card">

    <h2>🌦️ Environmental Risk Prediction</h2>

    <p>
    Enter environmental condition information.
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.slider(
            "🌡️ Temperature (°C)",
            0,
            50,
            28,
            key="env_temperature"
        )

        humidity = st.slider(
            "💧 Humidity (%)",
            0,
            100,
            60,
            key="env_humidity"
        )

        rainfall = st.slider(
            "🌧️ Rainfall (mm)",
            0,
            300,
            20,
            key="env_rainfall"
        )

    with col2:

        air_quality = st.slider(
            "🌬️ Air Quality Index",
            0,
            500,
            80,
            key="env_air_quality"
        )

        wind = st.slider(
            "🌪️ Wind Speed (km/h)",
            0,
            150,
            20,
            key="env_wind"
        )

        flood = st.slider(
            "🌊 Flood Risk (%)",
            0,
            100,
            10,
            key="env_flood"
        )

    if temperature > 40:
        risk_score += 25
        factors.append(
            "Temperature is above 40°C"
        )

    if humidity > 85:
        risk_score += 15
        factors.append(
            "Humidity is very high"
        )

    if rainfall > 100:
        risk_score += 20
        factors.append(
            "Heavy rainfall detected"
        )

    if air_quality > 150:
        risk_score += 20
        factors.append(
            "Air Quality Index is high"
        )

    if wind > 80:
        risk_score += 10
        factors.append(
            "High wind speed detected"
        )

    if flood > 50:
        risk_score += 20
        factors.append(
            "High flood risk detected"
        )

    risk_score = min(
        risk_score,
        100
    )

    performance_data = pd.DataFrame({

        "Parameter": [
            "Temperature",
            "Humidity",
            "Rainfall",
            "Air Quality",
            "Wind Speed",
            "Flood Risk"
        ],

        "Current": [
            temperature,
            humidity,
            rainfall,
            air_quality,
            wind,
            flood
        ]

    })

    if risk_score <= 20:
        risk = "LOW"

    elif risk_score <= 40:
        risk = "MEDIUM"

    else:
        risk = "HIGH"

    recommendation = generate_recommendation(
        risk,
        factors,
        ENV_ACTIONS,
        "Environmental conditions are currently within an "
        "acceptable range."
    )

    render_results(page, risk, risk_score, factors, recommendation, performance_data)
