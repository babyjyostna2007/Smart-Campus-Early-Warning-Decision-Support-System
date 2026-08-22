import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Campus Risk Dashboard",
    page_icon="🏫",
    layout="wide"
)

# ----------------------------------------------------------------------
# Load model + data (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("smart_campus_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_smart_campus.csv")

model = load_model()
df = load_data()

FEATURES = [
    "Attendance", "Current_GPA", "Previous_GPA", "Assignment_Rate", "Backlogs",
    "Occupancy_Rate", "Electricity_Usage", "Internet_Usage", "Maintenance_Complaints",
    "Temperature", "Humidity", "Rainfall", "Water_Level", "Air_Quality_Index"
]

ACADEMIC_FEATURES = ["Attendance", "Current_GPA", "Previous_GPA", "Assignment_Rate", "Backlogs"]
FACILITY_FEATURES = ["Occupancy_Rate", "Electricity_Usage", "Internet_Usage", "Maintenance_Complaints"]
ENV_FEATURES = ["Temperature", "Humidity", "Rainfall", "Water_Level", "Air_Quality_Index"]

RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

# Recommended_Action maps 1:1 with Risk_Level in the training data
RECOMMENDED_ACTION = {
    "Low": "Continue regular monitoring",
    "Medium": "Monitor performance and attendance",
    "High": "Faculty intervention; Academic counselling; Campus monitoring"
}


def get_key_risk_factors(inputs: dict) -> list:
    """Rule-based key risk factors, mirroring the logic used to label the training data."""
    factors = []
    if inputs["Attendance"] < 60:
        factors.append("Low attendance")
    if inputs["Current_GPA"] < inputs["Previous_GPA"]:
        factors.append("Declining GPA")
    if inputs["Assignment_Rate"] < 60:
        factors.append("Low assignment completion")
    if inputs["Backlogs"] >= 1:
        factors.append("Backlogs")
    if inputs["Rainfall"] > 80:
        factors.append("Heavy rainfall")
    if inputs["Water_Level"] > 80:
        factors.append("High water level")
    if inputs["Air_Quality_Index"] > 180:
        factors.append("Poor air quality")
    if inputs["Maintenance_Complaints"] >= 10:
        factors.append("High maintenance complaints")
    if inputs["Occupancy_Rate"] > 90:
        factors.append("Overcrowding")
    if inputs["Electricity_Usage"] > 1300:
        factors.append("High electricity usage")
    if not factors:
        factors.append("No major risk factors")
    return factors


def render_prediction_section(key_prefix: str):
    """Reusable 'Predict Risk Level' block, dropped into any tab.
    Uses ALL 14 features from session_state (so values entered in other
    tabs are included), since the model needs the full feature set."""
    st.markdown("---")
    st.markdown("### 🔮 Predict Risk Level")
    st.caption("Uses the current values from all three tabs (Academic, Facility, Environment).")

    if st.button("Predict Risk Level", key=f"{key_prefix}_predict_btn", use_container_width=True):
        input_dict = {f: st.session_state[f] for f in FEATURES}
        input_df = pd.DataFrame([input_dict])[FEATURES]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        classes = model.classes_

        color = RISK_COLORS.get(prediction, "#3498db")
        st.markdown(
            f"<div style='padding:20px;border-radius:10px;background-color:{color}22;"
            f"border:2px solid {color};text-align:center;'>"
            f"<h2 style='color:{color};margin:0;'>Predicted Risk Level: {prediction}</h2></div>",
            unsafe_allow_html=True
        )

        prob_df = pd.DataFrame({"Risk_Level": classes, "Probability": proba})
        fig = px.bar(prob_df, x="Risk_Level", y="Probability", color="Risk_Level",
                     color_discrete_map=RISK_COLORS, range_y=[0, 1], title="Prediction Confidence")
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_proba_chart")

        key_factors = get_key_risk_factors(input_dict)
        recommended_action = RECOMMENDED_ACTION.get(prediction, "Continue regular monitoring")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### 🚩 Key Risk Factors")
            if key_factors == ["No major risk factors"]:
                st.success("No major risk factors detected")
            else:
                for f in key_factors:
                    st.markdown(f"- {f}")
        with colB:
            st.markdown("#### ✅ Recommended Action")
            st.info(recommended_action)

# ----------------------------------------------------------------------
# Session state for input values (persists across tabs)
# ----------------------------------------------------------------------
DEFAULTS = {
    "Attendance": 71, "Current_GPA": 7.0, "Previous_GPA": 7.0, "Assignment_Rate": 66,
    "Backlogs": 2, "Occupancy_Rate": 71, "Electricity_Usage": 937.0, "Internet_Usage": 66,
    "Maintenance_Complaints": 8, "Temperature": 31.0, "Humidity": 64, "Rainfall": 58.0,
    "Water_Level": 56, "Air_Quality_Index": 133
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🏫 Smart Campus Risk Prediction Dashboard")
st.caption("Explore campus data, enter live values, and predict student/campus risk level using a trained Random Forest model.")

tab_overview, tab_academic, tab_facility, tab_env, tab_about = st.tabs(
    ["📊 Overview", "🎓 Academic", "🏢 Facility", "🌦️ Environment", "ℹ️ About"]
)

# ----------------------------------------------------------------------
# TAB: Overview
# ----------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Avg Risk Score", round(df["Risk_Score"].mean(), 2))
    c3.metric("High Risk %", f"{(df['Risk_Level'] == 'High').mean()*100:.1f}%")
    c4.metric("Avg Attendance", f"{df['Attendance'].mean():.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        risk_counts = df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"])
        fig = px.pie(
            names=risk_counts.index, values=risk_counts.values,
            title="Risk Level Distribution",
            color=risk_counts.index, color_discrete_map=RISK_COLORS
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="Risk_Score", nbins=30, title="Risk Score Distribution")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Feature Importance (from trained model)")
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    fig3 = px.bar(importance, x="Importance", y="Feature", orientation="h", title="What drives the model's predictions")
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### Most Common Key Risk Factors")
    all_factors = (
        df["Risk_Factors"]
        .str.split(";")
        .explode()
        .str.strip()
    )
    factor_counts = all_factors[all_factors != "No major risk factors"].value_counts().head(10)
    fig4 = px.bar(
        x=factor_counts.values, y=factor_counts.index, orientation="h",
        labels={"x": "Occurrences", "y": "Risk Factor"},
        title="Top 10 recurring risk factors across all records"
    )
    fig4.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Recommended Actions Breakdown")
    action_counts = df["Recommended_Action"].value_counts()
    fig5 = px.bar(
        x=action_counts.index, y=action_counts.values,
        labels={"x": "Recommended Action", "y": "Count"},
        title="How many records fall under each recommended action"
    )
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("View raw dataset"):
        st.dataframe(df, use_container_width=True)

# ----------------------------------------------------------------------
# TAB: Academic
# ----------------------------------------------------------------------
with tab_academic:
    st.subheader("🎓 Academic Metrics")
    st.write("Enter academic values below, then predict the risk level right here.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["Attendance"] = st.slider("Attendance (%)", 0, 100, int(st.session_state["Attendance"]))
        st.session_state["Current_GPA"] = st.number_input("Current GPA", 0.0, 10.0, float(st.session_state["Current_GPA"]), step=0.01)
        st.session_state["Previous_GPA"] = st.number_input("Previous GPA", 0.0, 10.0, float(st.session_state["Previous_GPA"]), step=0.01)
    with col2:
        st.session_state["Assignment_Rate"] = st.slider("Assignment Completion Rate (%)", 0, 100, int(st.session_state["Assignment_Rate"]))
        st.session_state["Backlogs"] = st.number_input("Backlogs", 0, 20, int(st.session_state["Backlogs"]), step=1)

    st.markdown("#### Academic Trends in Dataset")
    fig = px.scatter(
        df, x="Current_GPA", y="Attendance", color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        title="Attendance vs Current GPA colored by Risk Level"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(df, x="Risk_Level", y="Backlogs", color="Risk_Level",
                   color_discrete_map=RISK_COLORS, title="Backlogs by Risk Level")
    st.plotly_chart(fig2, use_container_width=True)

    render_prediction_section("academic")

# ----------------------------------------------------------------------
# TAB: Facility
# ----------------------------------------------------------------------
with tab_facility:
    st.subheader("🏢 Facility Metrics")
    st.write("Enter facility/infrastructure values below, then predict the risk level right here.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["Occupancy_Rate"] = st.slider("Occupancy Rate (%)", 0, 100, int(st.session_state["Occupancy_Rate"]))
        st.session_state["Electricity_Usage"] = st.number_input("Electricity Usage (kWh)", 0.0, 3000.0, float(st.session_state["Electricity_Usage"]), step=10.0)
    with col2:
        st.session_state["Internet_Usage"] = st.slider("Internet Usage (%)", 0, 100, int(st.session_state["Internet_Usage"]))
        st.session_state["Maintenance_Complaints"] = st.number_input("Maintenance Complaints", 0, 50, int(st.session_state["Maintenance_Complaints"]), step=1)

    st.markdown("#### Facility Trends in Dataset")
    fig = px.scatter(
        df, x="Occupancy_Rate", y="Electricity_Usage", color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        title="Electricity Usage vs Occupancy Rate colored by Risk Level"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(df, x="Risk_Level", y="Maintenance_Complaints", color="Risk_Level",
                   color_discrete_map=RISK_COLORS, title="Maintenance Complaints by Risk Level")
    st.plotly_chart(fig2, use_container_width=True)

    render_prediction_section("facility")

# ----------------------------------------------------------------------
# TAB: Environment
# ----------------------------------------------------------------------
with tab_env:
    st.subheader("🌦️ Environmental Metrics")
    st.write("Enter environmental sensor values below, then predict the risk level right here.")

    col1, col2 = st.columns(2)
    with col1:
        st.session_state["Temperature"] = st.number_input("Temperature (°C)", 0.0, 60.0, float(st.session_state["Temperature"]), step=0.1)
        st.session_state["Humidity"] = st.slider("Humidity (%)", 0, 100, int(st.session_state["Humidity"]))
        st.session_state["Rainfall"] = st.number_input("Rainfall (mm)", 0.0, 300.0, float(st.session_state["Rainfall"]), step=0.5)
    with col2:
        st.session_state["Water_Level"] = st.slider("Water Level (%)", 0, 100, int(st.session_state["Water_Level"]))
        st.session_state["Air_Quality_Index"] = st.number_input("Air Quality Index (AQI)", 0, 500, int(st.session_state["Air_Quality_Index"]), step=1)

    st.markdown("#### Environmental Trends in Dataset")
    fig = px.scatter(
        df, x="Rainfall", y="Water_Level", color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        title="Water Level vs Rainfall colored by Risk Level"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(df, x="Risk_Level", y="Air_Quality_Index", color="Risk_Level",
                   color_discrete_map=RISK_COLORS, title="Air Quality Index by Risk Level")
    st.plotly_chart(fig2, use_container_width=True)

    render_prediction_section("environment")

# ----------------------------------------------------------------------
# TAB: About
# ----------------------------------------------------------------------
with tab_about:
    st.subheader("ℹ️ About this Dashboard")
    st.markdown("""
This dashboard predicts a campus/student **Risk Level** (`Low`, `Medium`, `High`) using a
**Random Forest Classifier** trained on 14 academic, facility, and environmental features.

**Model details**
- Algorithm: `RandomForestClassifier` (`n_estimators=100`, `class_weight="balanced"`)
- Test accuracy: ~84.7%
- Training data: `cleaned_smart_campus.csv` (750 records, cleaned via `Data_cleaning.ipynb`)

**How to use**
1. Go to the **Academic**, **Facility**, and **Environment** tabs and enter values relevant to that category.
2. Each tab has its own **Predict Risk Level** button at the bottom — it uses the values from
   all three tabs together (the model needs all 14 features), so you can predict from wherever
   you're already working.
3. Review the predicted risk level, confidence chart, key risk factors, and recommended action.
    """)
