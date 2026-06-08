"""
Utility constants and helper functions for the Heart Disease Prediction project.
"""

import os

# ──────────────────────────────────────────────
# Project Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EDA_PLOTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "eda_plots")
MODEL_METRICS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "model_metrics")

# Create directories if they don't exist
for d in [DATA_DIR, MODELS_DIR, EDA_PLOTS_DIR, MODEL_METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Dataset Configuration
# ──────────────────────────────────────────────
RAW_DATA_PATH = os.path.join(DATA_DIR, "heart.csv")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "heart_cleaned.csv")
TARGET_COL = "target"

# ──────────────────────────────────────────────
# Feature Definitions
# ──────────────────────────────────────────────
FEATURE_INFO = {
    "age": {
        "description": "Age of the patient (years)",
        "type": "numerical",
        "min": 29, "max": 77,
    },
    "sex": {
        "description": "Sex of the patient",
        "type": "categorical",
        "values": {1: "Male", 0: "Female"},
    },
    "cp": {
        "description": "Chest Pain Type",
        "type": "categorical",
        "values": {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic",
        },
    },
    "trestbps": {
        "description": "Resting Blood Pressure (mm Hg)",
        "type": "numerical",
        "min": 94, "max": 200,
    },
    "chol": {
        "description": "Serum Cholesterol (mg/dl)",
        "type": "numerical",
        "min": 126, "max": 564,
    },
    "fbs": {
        "description": "Fasting Blood Sugar > 120 mg/dl",
        "type": "categorical",
        "values": {0: "No (≤120)", 1: "Yes (>120)"},
    },
    "restecg": {
        "description": "Resting ECG Results",
        "type": "categorical",
        "values": {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy",
        },
    },
    "thalach": {
        "description": "Maximum Heart Rate Achieved",
        "type": "numerical",
        "min": 71, "max": 202,
    },
    "exang": {
        "description": "Exercise Induced Angina",
        "type": "categorical",
        "values": {0: "No", 1: "Yes"},
    },
    "oldpeak": {
        "description": "ST Depression Induced by Exercise",
        "type": "numerical",
        "min": 0.0, "max": 6.2,
    },
    "slope": {
        "description": "Slope of Peak Exercise ST Segment",
        "type": "categorical",
        "values": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    },
    "ca": {
        "description": "Number of Major Vessels (0-4) Colored by Fluoroscopy",
        "type": "numerical",
        "min": 0, "max": 4,
    },
    "thal": {
        "description": "Thalassemia",
        "type": "categorical",
        "values": {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"},
    },
}

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ──────────────────────────────────────────────
# Visualization Theme
# ──────────────────────────────────────────────
COLORS = {
    "primary": "#E63946",      # Red accent
    "secondary": "#457B9D",    # Steel blue
    "background": "#1D3557",   # Dark navy
    "surface": "#F1FAEE",      # Light surface
    "text": "#F1FAEE",         # Light text
    "positive": "#2A9D8F",     # Teal green (no disease)
    "negative": "#E63946",     # Red (disease)
    "palette": ["#2A9D8F", "#E63946", "#457B9D", "#F4A261", "#264653"],
}

PLOT_STYLE = {
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#1A1A2E",
    "axes.edgecolor": "#333366",
    "axes.labelcolor": "#E0E0E0",
    "text.color": "#E0E0E0",
    "xtick.color": "#AAAACC",
    "ytick.color": "#AAAACC",
    "grid.color": "#333366",
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 12,
}


def get_feature_label(feature_name: str) -> str:
    """Return a human-readable label for a feature."""
    info = FEATURE_INFO.get(feature_name, {})
    return info.get("description", feature_name.replace("_", " ").title())


def print_section(title: str, char: str = "═", width: int = 60) -> None:
    """Print a formatted section header."""
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")
