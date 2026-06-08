"""
Heart Disease Prediction — Streamlit Web Application
=====================================================
Interactive prediction app for the fedesoriano Heart Failure dataset.
11 features: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
             RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
"""

import os
import sys
import json
import glob
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils import MODELS_DIR, EDA_PLOTS_DIR, MODEL_METRICS_DIR, CLEANED_DATA_PATH

# ══════════════════════════════════════════════
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center; padding: 1.5rem 0;
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
        border-radius: 16px; margin-bottom: 2rem;
        border: 1px solid #333366; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #E63946, #FF6B6B, #F4A261);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p { color: #AAAACC; font-size: 1.1rem; margin: 0; }

    .metric-card {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 1px solid #333366; border-radius: 12px; padding: 1.5rem;
        text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(230,57,70,0.15); }
    .metric-card .metric-value { font-size: 2rem; font-weight: 800; color: #E63946; }
    .metric-card .metric-label { font-size: 0.9rem; color: #AAAACC; margin-top: 0.3rem; }

    .prediction-result {
        text-align: center; padding: 2rem; border-radius: 16px;
        margin: 1.5rem 0; border: 2px solid;
    }
    .prediction-positive { background: linear-gradient(135deg, #0a2a1f, #0d3b2a); border-color: #2A9D8F; }
    .prediction-negative { background: linear-gradient(135deg, #2a0a0f, #3b0d15); border-color: #E63946; }
    .prediction-result h2 { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .prediction-result p { font-size: 1.1rem; color: #CCCCDD; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0E1117 0%, #1A1A2E 100%); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}

    .section-header {
        font-size: 1.5rem; font-weight: 700; color: #E0E0E0;
        margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #333366;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(model_path):
        return None, None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    return model, scaler, metadata


@st.cache_data
def load_data():
    if os.path.exists(CLEANED_DATA_PATH):
        return pd.read_csv(CLEANED_DATA_PATH)
    return None


# ══════════════════════════════════════════════
def sidebar():
    with st.sidebar:
        st.markdown("## 🫀 Navigation")
        st.markdown("---")
        page = st.radio("Go to",
                        ["🏠 Home", "📊 EDA Dashboard", "🤖 Predict", "📈 Model Performance"],
                        label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Predicts heart disease risk using ML models trained on 918 clinical records.")
        st.markdown("---")
        st.markdown("<div style='text-align:center;color:#666688;font-size:0.8rem;'>"
                    "Built with ❤️ using Streamlit</div>", unsafe_allow_html=True)
    return page


# ══════════════════════════════════════════════
def page_home():
    st.markdown("<div class='main-header'><h1>🫀 Heart Disease Predictor</h1>"
                "<p>End-to-End ML Project — From Data to Deployment</p></div>",
                unsafe_allow_html=True)

    df = load_data()
    _, _, metadata = load_model()

    col1, col2, col3, col4 = st.columns(4)
    vals = [
        (df.shape[0] if df is not None else "N/A", "Total Samples"),
        (df.shape[1]-1 if df is not None else "N/A", "Features"),
        (f"{metadata.get('accuracy',0)*100:.1f}%" if metadata else "N/A", "Best Accuracy"),
        ("5", "Models Trained"),
    ]
    for col, (val, label) in zip([col1, col2, col3, col4], vals):
        with col:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{val}</div>"
                        f"<div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("<div class='section-header'>📋 Project Overview</div>", unsafe_allow_html=True)
        st.markdown("""
        A **complete end-to-end data science project** predicting heart disease from clinical features:
        1. **📥 Data Collection** — 918 records from Kaggle (fedesoriano)
        2. **🧹 Data Cleaning** — Duplicate removal, zero-cholesterol fix, encoding
        3. **📊 EDA** — 11 publication-quality visualizations
        4. **🤖 Model Building** — 5 ML models with hyperparameter tuning
        5. **🚀 Deployment** — Interactive Streamlit web app
        """)
    with col_r:
        st.markdown("<div class='section-header'>🔬 Key Features</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            ("Age", "Patient age (years)"), ("Sex", "Male/Female"),
            ("ChestPainType", "4 types"), ("RestingBP", "mm Hg"),
            ("Cholesterol", "mg/dl"), ("MaxHR", "Max heart rate"),
            ("Oldpeak", "ST depression"), ("ST_Slope", "Up/Flat/Down"),
        ], columns=["Feature", "Description"]), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>🛠️ Tech Stack</div>", unsafe_allow_html=True)
    for col, (icon, name, desc) in zip(st.columns(5), [
        ("🐍","Python","Core"), ("🐼","Pandas","Data"), ("📊","Matplotlib","Plots"),
        ("🤖","Scikit-learn","ML"), ("🚀","Streamlit","Deploy")]):
        with col:
            st.markdown(f"<div class='metric-card'><div style='font-size:2rem'>{icon}</div>"
                        f"<div style='font-weight:700;color:#E0E0E0;margin-top:0.5rem'>{name}</div>"
                        f"<div class='metric-label'>{desc}</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
def page_eda():
    st.markdown("<div class='main-header'><h1>📊 Exploratory Data Analysis</h1>"
                "<p>11 publication-quality visualizations</p></div>", unsafe_allow_html=True)

    plot_files = sorted(glob.glob(os.path.join(EDA_PLOTS_DIR, "*.png")))
    if not plot_files:
        st.warning("⚠️ No plots found. Run: `python src/eda.py`")
        return

    descs = {
        "01_target_distribution": "~55% positive (disease) vs ~45% negative. Fairly balanced.",
        "02_age_distribution": "Older patients (55+) have higher heart disease prevalence.",
        "03_correlation_heatmap": "MaxHR, ExerciseAngina, and Oldpeak strongly correlate with target.",
        "04_chest_pain_analysis": "Asymptomatic chest pain has the highest disease rate (~80%).",
        "05_gender_analysis": "Males have significantly higher heart disease rates than females.",
        "06_numerical_boxplots": "Disease patients have lower MaxHR and higher Oldpeak.",
        "07_heart_rate_vs_age": "Disease patients cluster in the low-MaxHR region with clear trend separation.",
        "08_cholesterol_bp_distribution": "Cholesterol and BP distributions overlap — weaker predictors alone.",
        "09_exercise_features": "Exercise angina + Flat ST slope are the strongest risk indicators.",
        "10_oldpeak_analysis": "Oldpeak is significantly elevated in heart disease patients.",
        "11_target_correlations": "ST_Slope, Oldpeak, ExerciseAngina, ChestPainType are top predictors.",
    }

    tabs = st.tabs([os.path.basename(p).replace(".png","").replace("_"," ").title() for p in plot_files])
    for tab, path in zip(tabs, plot_files):
        with tab:
            name = os.path.basename(path).replace(".png","")
            if name in descs:
                st.info(f"💡 {descs[name]}")
            st.image(path, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>📋 Dataset Preview</div>", unsafe_allow_html=True)
    df = load_data()
    if df is not None:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        with c2:
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
            st.metric("Missing", df.isnull().sum().sum())


# ══════════════════════════════════════════════
def page_predict():
    st.markdown("<div class='main-header'><h1>🤖 Heart Disease Prediction</h1>"
                "<p>Enter clinical features to get a risk assessment</p></div>",
                unsafe_allow_html=True)

    model, scaler, metadata = load_model()
    if model is None:
        st.error("⚠️ No model found. Run: `python src/model_building.py`")
        return

    if metadata:
        st.markdown(f"**Model:** {metadata.get('model_name','?')} | "
                    f"**Accuracy:** {metadata.get('accuracy',0)*100:.1f}% | "
                    f"**F1:** {metadata.get('f1_score',0)*100:.1f}%")

    st.markdown("---")
    st.markdown("<div class='section-header'>📝 Patient Information</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("🎂 Age", 20, 90, 50)
        sex = st.selectbox("👤 Sex", [1, 0], format_func=lambda x: "Male" if x else "Female")
        chest_pain = st.selectbox("💔 Chest Pain Type", [0, 1, 2, 3],
            format_func=lambda x: {0:"Typical Angina",1:"Atypical Angina",
                                   2:"Non-anginal Pain",3:"Asymptomatic"}[x])
        resting_bp = st.slider("🩸 Resting BP (mm Hg)", 80, 200, 130)

    with col2:
        cholesterol = st.slider("🧪 Cholesterol (mg/dl)", 100, 600, 230)
        fasting_bs = st.selectbox("🍬 Fasting BS > 120", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")
        resting_ecg = st.selectbox("📊 Resting ECG", [0, 1, 2],
            format_func=lambda x: {0:"Normal",1:"ST-T Abnormality",2:"LV Hypertrophy"}[x])

    with col3:
        max_hr = st.slider("❤️ Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("🏃 Exercise Angina", [0, 1],
                                       format_func=lambda x: "Yes" if x else "No")
        oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 6.2, 1.0, step=0.1)
        st_slope = st.selectbox("📈 ST Slope", [0, 1, 2],
            format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])

    st.markdown("---")

    if st.button("🔮 Predict Heart Disease Risk", use_container_width=True, type="primary"):
        # Build input in the EXACT feature order from training
        feature_names = metadata.get("feature_names", [
            "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
        ])

        input_values = {
            "Age": age, "Sex": sex, "ChestPainType": chest_pain,
            "RestingBP": resting_bp, "Cholesterol": cholesterol,
            "FastingBS": fasting_bs, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak, "ST_Slope": st_slope,
        }

        # Create DataFrame with correct column order
        input_df = pd.DataFrame([[input_values[f] for f in feature_names]], columns=feature_names)

        # Scale and predict
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("---")

        if prediction == 0:
            conf = probability[0] * 100
            st.markdown(
                "<div class='prediction-result prediction-positive'>"
                "<h2>🟢 Low Risk — No Heart Disease Detected</h2>"
                f"<p>The model predicts this patient is <strong>unlikely</strong> to have heart disease.</p>"
                f"<p style='font-size:1.3rem;color:#2A9D8F;font-weight:700;'>Confidence: {conf:.1f}%</p>"
                "</div>", unsafe_allow_html=True)
        else:
            conf = probability[1] * 100
            st.markdown(
                "<div class='prediction-result prediction-negative'>"
                "<h2>🔴 High Risk — Heart Disease Detected</h2>"
                f"<p>The model predicts this patient is <strong>at risk</strong>. Consult a cardiologist.</p>"
                f"<p style='font-size:1.3rem;color:#E63946;font-weight:700;'>Confidence: {conf:.1f}%</p>"
                "</div>", unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=probability[1]*100,
            title={"text": "Heart Disease Probability", "font": {"size": 18, "color": "#E0E0E0"}},
            number={"suffix": "%", "font": {"size": 36, "color": "#E0E0E0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#AAAACC"},
                "bar": {"color": "#E63946" if prediction == 1 else "#2A9D8F"},
                "bgcolor": "#1A1A2E", "borderwidth": 2, "bordercolor": "#333366",
                "steps": [
                    {"range": [0, 30], "color": "#0a2a1f"},
                    {"range": [30, 60], "color": "#2a2a0f"},
                    {"range": [60, 100], "color": "#2a0a0f"},
                ],
                "threshold": {"line": {"color": "#FFD700", "width": 3}, "thickness": 0.75, "value": 50},
            },
        ))
        fig.update_layout(paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                          font={"color": "#E0E0E0"}, height=300,
                          margin={"t": 60, "b": 20, "l": 30, "r": 30})
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Input Summary"):
            st.dataframe(input_df.T.reset_index().rename(columns={"index": "Feature", 0: "Value"}),
                         use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** For educational purposes only. Not a substitute for medical advice.")


# ══════════════════════════════════════════════
def page_model_performance():
    st.markdown("<div class='main-header'><h1>📈 Model Performance</h1>"
                "<p>Comparison of all trained models</p></div>", unsafe_allow_html=True)

    _, _, metadata = load_model()
    csv_path = os.path.join(MODEL_METRICS_DIR, "model_comparison.csv")

    if os.path.exists(csv_path):
        st.markdown("<div class='section-header'>📊 Model Comparison</div>", unsafe_allow_html=True)
        df = pd.read_csv(csv_path)
        st.dataframe(df.style.highlight_max(
            subset=["Accuracy","Precision","Recall","F1-Score","ROC-AUC"], color="#2A9D8F44"),
            use_container_width=True, hide_index=True)

    if metadata:
        st.markdown("---")
        st.markdown("<div class='section-header'>🏆 Best Model</div>", unsafe_allow_html=True)
        for col, (label, key) in zip(st.columns(4), [
            ("Model", "model_name"), ("Accuracy", "accuracy"),
            ("F1-Score", "f1_score"), ("ROC-AUC", "roc_auc")]):
            with col:
                v = metadata.get(key, "N/A")
                st.metric(label, v if isinstance(v, str) else f"{v*100:.1f}%")

    st.markdown("---")
    st.markdown("<div class='section-header'>📈 Visualizations</div>", unsafe_allow_html=True)
    plots = sorted(glob.glob(os.path.join(MODEL_METRICS_DIR, "*.png")))
    if plots:
        names = {"model_comparison": "Comparison", "confusion_matrices": "Confusion Matrices",
                 "roc_curves": "ROC Curves", "feature_importance": "Feature Importance",
                 "cv_comparison": "Cross-Validation"}
        tabs = st.tabs([names.get(os.path.basename(p).replace(".png",""), "Plot") for p in plots])
        for tab, path in zip(tabs, plots):
            with tab:
                st.image(path, use_container_width=True)


# ══════════════════════════════════════════════
def main():
    page = sidebar()
    if page == "🏠 Home": page_home()
    elif page == "📊 EDA Dashboard": page_eda()
    elif page == "🤖 Predict": page_predict()
    elif page == "📈 Model Performance": page_model_performance()


if __name__ == "__main__":
    main()
