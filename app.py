import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio

pio.templates["clean_minimal"] = pio.templates["plotly_white"]
pio.templates["clean_minimal"].layout.update(
    font=dict(family="Inter, -apple-system, sans-serif", color="#3a3f47"),
    title=dict(font=dict(size=15, color="#1a1a1a")),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=10, r=10, t=45, b=10),
    colorway=["#2ecc71", "#f39c12", "#e74c3c", "#3498db", "#9b59b6"],
)
pio.templates.default = "clean_minimal"

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Campus Risk Dashboard",
    page_icon="🏫",
    layout="wide"
)

# ----------------------------------------------------------------------
# Minimal, clean styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }
    #MainMenu, footer, header { visibility: hidden; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #eef0f2;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: left;
    }
    .metric-label {
        font-size: 13px;
        color: #8a8f98;
        font-weight: 500;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1a1a1a;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        color: #b0b4ba;
        margin-top: 4px;
    }

    .result-card {
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        border: 1px solid #eef0f2;
        margin-bottom: 6px;
    }
    .result-label {
        font-size: 13px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.75;
        margin-bottom: 6px;
    }
    .result-value {
        font-size: 34px;
        font-weight: 800;
        margin: 0;
    }

    .factor-pill {
        display: inline-block;
        background: #f5f6f8;
        color: #3a3f47;
        border-radius: 999px;
        padding: 5px 14px;
        margin: 4px 6px 4px 0;
        font-size: 13px;
        border: 1px solid #e7e9ec;
    }

    div[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

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


def metric_card(label, value, sub=""):
    st.markdown(
        f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True
    )


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


def live_alert_banner(key_prefix: str):
    """Auto-senses the current combined values (from all 3 tabs) on every
    rerun — no button needed — and shows an alert banner immediately.
    Fires a one-time toast when risk newly crosses into High."""
    input_dict = {f: st.session_state[f] for f in FEATURES}
    input_df = pd.DataFrame([input_dict])[FEATURES]

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    confidence = proba[classes.index(prediction)] * 100
    factors = get_key_risk_factors(input_dict)

    # Fire a one-time toast only when the level changes to High
    if prediction == "High" and st.session_state.get("_last_alert_level") != "High":
        st.toast(f"⚠️ Risk just became HIGH ({confidence:.0f}% confidence)", icon="🚨")
    st.session_state["_last_alert_level"] = prediction

    if prediction == "High":
        st.error(
            f"🚨 **Live Alert — HIGH risk** ({confidence:.0f}% confidence). "
            f"Triggers: {', '.join(f for f in factors if f != 'No major risk factors') or 'model-detected pattern'}. "
            f"Action: {RECOMMENDED_ACTION['High']}",
            icon="🚨"
        )
    elif prediction == "Medium":
        st.warning(
            f"⚠️ **Live status — MEDIUM risk** ({confidence:.0f}% confidence). "
            f"Keep an eye on: {', '.join(f for f in factors if f != 'No major risk factors') or 'borderline values'}.",
            icon="⚠️"
        )
    else:
        st.success(f"✅ **Live status — LOW risk** ({confidence:.0f}% confidence). No action needed.", icon="✅")


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
        classes = list(model.classes_)
        confidence = proba[classes.index(prediction)] * 100

        color = RISK_COLORS.get(prediction, "#3498db")

        # ---- Big result card + gauge, side by side ----
        colR, colG = st.columns([1, 1.2])
        with colR:
            st.markdown(
                f"""<div class="result-card" style="background:{color}14;border-color:{color}55;">
                        <div class="result-label" style="color:{color};">Predicted Risk Level</div>
                        <p class="result-value" style="color:{color};">{prediction}</p>
                    </div>""",
                unsafe_allow_html=True
            )
            m1, m2 = st.columns(2)
            with m1:
                metric_card("Confidence", f"{confidence:.1f}%")
            with m2:
                metric_card("Recommended Action", RECOMMENDED_ACTION.get(prediction, "—"), sub="")

        with colG:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"suffix": "%", "font": {"size": 34}},
                title={"text": f"Confidence in '{prediction}'", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": color, "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "#f5f6f8"},
                        {"range": [50, 80], "color": "#eef0f2"},
                        {"range": [80, 100], "color": "#e7e9ec"},
                    ],
                }
            ))
            gauge.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True, key=f"{key_prefix}_gauge")

        # ---- Key risk factors as pills ----
        key_factors = get_key_risk_factors(input_dict)
        st.markdown("##### 🚩 Key Risk Factors")
        if key_factors == ["No major risk factors"]:
            st.success("No major risk factors detected")
        else:
            pills = "".join(f'<span class="factor-pill">{f}</span>' for f in key_factors)
            st.markdown(pills, unsafe_allow_html=True)


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

st.markdown("## 🏫 Smart Campus Risk Dashboard")
st.caption("Enter live values by category and predict risk using a trained Random Forest model.")

tab_overview, tab_academic, tab_facility, tab_env, tab_bulk, tab_about = st.tabs(
    ["📊 Overview", "🎓 Academic", "🏢 Facility", "🌦️ Environment", "🚨 Bulk Scan", "ℹ️ About"]
)

# ----------------------------------------------------------------------
# TAB: Overview
# ----------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Records", f"{len(df):,}")
    with c2:
        metric_card("Avg Risk Score", f"{df['Risk_Score'].mean():.2f}")
    with c3:
        metric_card("High Risk %", f"{(df['Risk_Level'] == 'High').mean()*100:.1f}%")
    with c4:
        metric_card("Avg Attendance", f"{df['Attendance'].mean():.1f}%")

    st.markdown("")

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

    live_alert_banner("academic")

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

    live_alert_banner("facility")

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

    live_alert_banner("environment")

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
# TAB: Bulk Scan
# ----------------------------------------------------------------------
with tab_bulk:
    st.subheader("🚨 Bulk Risk Scan")
    st.write(
        "Upload a CSV with the 14 feature columns (one row per student/campus record) "
        "to scan them all at once and flag the risky ones."
    )

    with st.expander("Required columns"):
        st.code(", ".join(FEATURES))

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="bulk_uploader")

    if uploaded is not None:
        try:
            scan_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")
            scan_df = None

        if scan_df is not None:
            missing = [f for f in FEATURES if f not in scan_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                X = scan_df[FEATURES].copy()
                preds = model.predict(X)
                probs = model.predict_proba(X)
                classes = list(model.classes_)

                scan_df["Predicted_Risk_Level"] = preds
                scan_df["Confidence_%"] = [
                    round(probs[i][classes.index(preds[i])] * 100, 1) for i in range(len(preds))
                ]
                scan_df["Key_Risk_Factors"] = [
                    "; ".join(get_key_risk_factors(row._asdict()))
                    for row in X.itertuples(index=False)
                ]
                scan_df["Recommended_Action"] = scan_df["Predicted_Risk_Level"].map(RECOMMENDED_ACTION)

                n_total = len(scan_df)
                n_high = int((scan_df["Predicted_Risk_Level"] == "High").sum())
                n_medium = int((scan_df["Predicted_Risk_Level"] == "Medium").sum())
                n_low = int((scan_df["Predicted_Risk_Level"] == "Low").sum())

                if n_high > 0:
                    st.error(f"🚨 {n_high} of {n_total} records flagged **HIGH risk** — review below.", icon="🚨")
                elif n_medium > 0:
                    st.warning(f"⚠️ No high-risk records, but {n_medium} are Medium risk.", icon="⚠️")
                else:
                    st.success("✅ No risky records found in this batch.", icon="✅")

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    metric_card("Total Scanned", f"{n_total:,}")
                with m2:
                    metric_card("High Risk", f"{n_high:,}", sub=f"{n_high/n_total*100:.1f}%" if n_total else "")
                with m3:
                    metric_card("Medium Risk", f"{n_medium:,}", sub=f"{n_medium/n_total*100:.1f}%" if n_total else "")
                with m4:
                    metric_card("Low Risk", f"{n_low:,}", sub=f"{n_low/n_total*100:.1f}%" if n_total else "")

                st.markdown("#### 🚩 Flagged Records (High Risk)")
                high_df = scan_df[scan_df["Predicted_Risk_Level"] == "High"]
                if len(high_df) > 0:
                    id_col = "Student_ID" if "Student_ID" in scan_df.columns else None
                    show_cols = ([id_col] if id_col else []) + [
                        "Predicted_Risk_Level", "Confidence_%", "Key_Risk_Factors", "Recommended_Action"
                    ]
                    st.dataframe(high_df[show_cols], use_container_width=True)
                else:
                    st.info("No high-risk records in this batch.")

                with st.expander("View full scanned results"):
                    st.dataframe(scan_df, use_container_width=True)

                csv_out = scan_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download full results as CSV",
                    data=csv_out,
                    file_name="bulk_risk_scan_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("No file uploaded yet. You can also try it on the existing dataset:")
        if st.button("Scan the loaded dataset (cleaned_smart_campus.csv)", use_container_width=True):
            X = df[FEATURES].copy()
            preds = model.predict(X)
            n_high = int((preds == "High").sum())
            st.error(f"🚨 {n_high} of {len(df)} records in the dataset are HIGH risk.", icon="🚨") \
                if n_high else st.success("No high-risk records found.")

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
2. Each tab shows a **live alert banner** that senses risk automatically as you type — no button needed.
3. Click **Predict Risk Level** in any tab for the full breakdown (gauge, confidence, key factors, recommended action).
4. Use **🚨 Bulk Scan** to upload a CSV of many records at once and flag every high-risk one automatically.

**Alerting**
- **Live alerts**: every tab auto-evaluates the current values on each interaction and shows a red/yellow/green banner instantly; a toast pops up the moment risk crosses into High.
- **Bulk scanning**: upload a CSV with the 14 feature columns to scan hundreds of records at once, see counts by risk level, view the flagged high-risk rows, and download full results.
    """)
