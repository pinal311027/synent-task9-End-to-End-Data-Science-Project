<p align="center">
  <h1 align="center">🫀 Heart Disease Prediction</h1>
  <p align="center">
    <strong>End-to-End Data Science Project — From Raw Data to Deployed Web Application</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-2.1+-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"></a>
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
    <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge" alt="Status">
  </p>
</p>

---

## 📖 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [💡 Motivation](#-motivation)
- [📊 Dataset](#-dataset)
- [🔄 Workflow](#-workflow)
- [📈 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🤖 Model Building & Results](#-model-building--results)
- [🚀 Deployment](#-deployment)
- [⚙️ Installation & Setup](#️-installation--setup)
- [📁 Project Structure](#-project-structure)
- [🛠️ Tech Stack](#️-tech-stack)
- [🔮 Future Improvements](#-future-improvements)
- [📜 License](#-license)

---

## 🎯 Project Overview

This project implements a **complete end-to-end data science pipeline** for predicting heart disease risk from patient clinical parameters. It covers the entire data science lifecycle:

| Phase | Description | Status |
|-------|-------------|--------|
| 📥 **Data Collection** | Merge 3 Kaggle datasets + augmentation (3,478 records) | ✅ Complete |
| 🧹 **Data Cleaning** | Fix zero cholesterol, validate ranges, encode categoricals, augment | ✅ Complete |
| 📊 **EDA** | 11 publication-quality dark-themed visualizations | ✅ Complete |
| 🤖 **Model Building** | 5 ML models with cross-validation & hyperparameter tuning | ✅ Complete |
| 🚀 **Deployment** | Interactive Streamlit web app with prediction UI | ✅ Complete |

The deployed application allows users to input patient clinical data and receive an **instant heart disease risk prediction** with a confidence score.

---

## 💡 Motivation

Cardiovascular disease is the **#1 cause of death globally**, responsible for approximately **17.9 million deaths per year** (WHO). Early detection and prediction of heart disease can significantly improve patient outcomes and reduce mortality rates.

This project demonstrates how machine learning can assist healthcare professionals in identifying at-risk patients based on readily available clinical measurements, enabling:

- **Early Intervention** — Identify high-risk patients before symptoms worsen
- **Decision Support** — Provide data-driven insights alongside clinical judgment
- **Screening Tool** — Enable rapid risk assessment during routine checkups

> ⚠️ **Disclaimer:** This tool is for educational purposes only and should NOT replace professional medical diagnosis.

---

## 📊 Dataset

**Sources (3 Kaggle datasets merged):**

| # | Dataset | Records | Source |
|---|---------|---------|--------|
| 1 | [fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | 918 | Combined 5 UCI databases |
| 2 | [johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) | 1,025 | UCI Heart Disease |
| 3 | [redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data) | 920 | Multi-source UCI |

After merging, deduplication, cleaning, and **Gaussian-noise augmentation**:

| Property | Details |
|----------|---------|
| **Raw Merged** | 2,863 records |
| **After Dedup** | 1,220 unique records |
| **After Augmentation** | **3,478 records** |
| **Features** | 11 clinical attributes |
| **Target** | Binary — Heart Disease (1) / No Disease (0) |
| **Class Balance** | 55.1% positive, 44.9% negative |
| **Missing Values** | 0 |

### Feature Description

| # | Feature | Description | Type |
|---|---------|-------------|------|
| 1 | **Age** | Patient age in years | Numerical |
| 2 | **Sex** | Gender (M/F → 1/0) | Categorical |
| 3 | **ChestPainType** | TA, ATA, NAP, ASY → 0–3 | Categorical |
| 4 | **RestingBP** | Resting blood pressure (mm Hg) | Numerical |
| 5 | **Cholesterol** | Serum cholesterol (mg/dl) | Numerical |
| 6 | **FastingBS** | Fasting blood sugar > 120 mg/dl | Categorical |
| 7 | **RestingECG** | Normal, ST, LVH → 0–2 | Categorical |
| 8 | **MaxHR** | Maximum heart rate achieved | Numerical |
| 9 | **ExerciseAngina** | Exercise-induced angina (N/Y → 0/1) | Categorical |
| 10 | **Oldpeak** | ST depression by exercise vs rest | Numerical |
| 11 | **ST_Slope** | Up, Flat, Down → 0–2 | Categorical |

---

## 🔄 Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    END-TO-END DATA SCIENCE PIPELINE                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📥 Data Collection ──→ 🧹 Data Cleaning ──→ 📊 EDA               │
│       │                      │                    │                 │
│  Merge 3 Kaggle        Fix 0-cholesterol    11 publication-        │
│  datasets              Validate ranges      quality plots          │
│  1220 unique rows      Gaussian augment     Dark-themed            │
│                         3478 final rows     visualizations          │
│                                                                     │
│  📊 EDA ──→ 🤖 Model Building ──→ 🚀 Deployment                   │
│                   │                      │                          │
│              5 ML models            Streamlit App                   │
│              Cross-validation       4 interactive pages             │
│              Hyperparameter tuning  Real-time predictions           │
│              Best: RF (97.3% F1)    Confidence gauges               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 1: Data Collection (`src/data_collection.py`)
- Downloads and merges **3 Kaggle datasets** into a unified schema
- Includes a synthetic data generator as fallback
- Provides comprehensive dataset inspection with statistics

### Step 2: Data Cleaning (`src/data_cleaning.py`)
- **Fixed 172 zero-cholesterol** values (biologically impossible → replaced with median)
- **Validated ranges** — fixed zero BP, negative Oldpeak, and out-of-range values
- **Encoded 5 categorical** columns (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope)
- **Gaussian-noise augmentation** — expanded 1,220 real rows to **3,478 rows** for robust training

### Step 3: Exploratory Data Analysis (`src/eda.py`)
- Generated **11 publication-quality visualizations** with a professional dark theme
- See [EDA Highlights](#-exploratory-data-analysis) below

### Step 4: Model Building (`src/model_building.py`)
- Trained **5 machine learning models** with standardized preprocessing
- **5-fold cross-validation** for robust evaluation
- **GridSearchCV hyperparameter tuning** for top 3 models
- See [Model Results](#-model-building--results) below

### Step 5: Deployment (`app.py`)
- Interactive **Streamlit web application** with 4 pages
- Real-time predictions with Plotly confidence gauge
- EDA dashboard and model performance visualization

---

## 📈 Exploratory Data Analysis

All visualizations use a consistent **dark theme** with a professional color palette for publication quality.

### Key Findings

| Finding | Insight |
|---------|---------|
| **Class Balance** | ~55% positive (heart disease) vs ~45% negative — fairly balanced |
| **Age Factor** | Patients aged 55+ show significantly higher heart disease prevalence |
| **Gender Disparity** | Males have significantly higher heart disease rates than females |
| **Chest Pain** | Asymptomatic patients (ASY) have the highest disease rate (~80%) |
| **Heart Rate** | Disease patients tend to have lower maximum heart rates |
| **ST Depression** | Oldpeak values are significantly higher in disease patients |
| **Exercise Angina** | Patients with exercise angina have ~75% heart disease rate |
| **Top Predictors** | ST_Slope, ExerciseAngina, Oldpeak, and ChestPainType |

### Visualizations Generated

| # | Plot | Purpose |
|---|------|---------|
| 1 | Target Distribution (Bar + Pie) | Class balance analysis |
| 2 | Age Distribution by Target | Age as a risk factor |
| 3 | Correlation Heatmap | Feature relationship matrix |
| 4 | Chest Pain Analysis | Categorical feature impact |
| 5 | Gender Analysis | Demographic risk comparison |
| 6 | Numerical Box Plots | Outlier detection & group comparison |
| 7 | Heart Rate vs Age Scatter | Multi-variable interaction with trend lines |
| 8 | Cholesterol & BP Distribution | Health metrics comparison |
| 9 | Exercise Features Analysis | Exercise-related predictors |
| 10 | Oldpeak (ST Depression) Analysis | Histogram + Violin plot |
| 11 | Feature-Target Correlations | Ranked correlation bar chart |

---

## 🤖 Model Building & Results

### Models Trained

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **🏆 Random Forest (tuned)** | **96.98%** | **96.65%** | **97.91%** | **97.28%** | **99.24%** |
| KNN (tuned) | 96.70% | 96.63% | 97.39% | 97.01% | 99.46% |
| Gradient Boosting (tuned) | 96.26% | 96.36% | 96.87% | 96.61% | 98.95% |
| SVM | 86.21% | 84.25% | 92.17% | 88.03% | 92.91% |
| Logistic Regression | 79.89% | 79.85% | 84.86% | 82.28% | 86.95% |

### Best Model: Random Forest (Tuned)

```
🏆 Best Model: Random Forest (tuned)
   ├── Accuracy:    96.98%
   ├── F1-Score:    97.28%
   ├── ROC-AUC:     99.24%
   ├── Recall:      97.91%  ← High recall is critical for medical diagnosis!
   └── Precision:   96.65%
```

> 💡 **Why Random Forest?** After hyperparameter tuning on 3,478 training samples, the Random Forest achieved the highest F1-score (97.28%) with near-perfect recall (97.91%). Validated on extreme cases — correctly predicts healthy patients (96% confidence) and at-risk patients (87-93% confidence).

### Hyperparameter Tuning Results (Top 3 Models)

| Model | Best Parameters | Best CV F1 |
|-------|----------------|------------|
| KNN | `metric=manhattan, n_neighbors=11, weights=distance` | 94.95% |
| Gradient Boosting | `learning_rate=0.2, max_depth=7, n_estimators=300, subsample=1.0` | 94.39% |
| Random Forest | `max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=300` | 94.35% |

### Visualizations Generated

- **Model Comparison** — Grouped bar chart of all metrics
- **Confusion Matrices** — For all 5 models
- **ROC Curves** — Comparison with AUC scores
- **Feature Importance** — From tree-based models
- **Cross-Validation Box Plot** — 5-fold CV accuracy distribution

---

## 🚀 Deployment

The application is deployed as an interactive **Streamlit web app** with 4 pages:

### 🏠 Home Page
- Project overview with key statistics
- Dataset feature table
- Tech stack showcase with icons

### 📊 EDA Dashboard
- Tabbed interface with all 11 EDA visualizations
- Contextual descriptions for each plot
- Dataset preview with stats

### 🤖 Prediction Page
- Interactive input form with sliders and dropdowns
- **Real-time heart disease risk prediction**
- Color-coded result (🟢 Low Risk / 🔴 High Risk)
- Plotly gauge chart showing confidence percentage
- Input summary table

### 📈 Model Performance
- Model comparison table with highlighted best scores
- Best model metrics dashboard
- Performance visualization gallery (ROC curves, confusion matrices, etc.)

### Run the App Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the complete pipeline
python src/data_collection.py    # Download dataset
python src/data_cleaning.py      # Clean & preprocess
python src/eda.py                # Generate EDA plots
python src/model_building.py     # Train & evaluate models

# 5. Launch the web app
streamlit run app.py
```

### One-Line Pipeline (After Setup)

```bash
python src/data_collection.py && python src/data_cleaning.py && python src/eda.py && python src/model_building.py && streamlit run app.py
```

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── 📊 data/
│   ├── heart.csv                    # Merged raw dataset (1,220 records)
│   └── heart_cleaned.csv           # Cleaned + augmented dataset (3,478 records)
│
├── 🐍 src/
│   ├── __init__.py                 # Package initialization
│   ├── utils.py                    # Constants, paths, feature definitions
│   ├── data_collection.py          # Dataset download & inspection
│   ├── data_cleaning.py            # Cleaning & preprocessing pipeline
│   ├── eda.py                      # 11 EDA visualizations
│   └── model_building.py           # ML training, tuning, evaluation
│
├── 🤖 models/
│   ├── best_model.pkl              # Serialized best model (Random Forest)
│   ├── scaler.pkl                  # StandardScaler for inference
│   └── model_metadata.json         # Model performance metadata
│
├── 📈 artifacts/
│   ├── eda_plots/                  # 11 EDA visualization PNGs
│   │   ├── 01_target_distribution.png
│   │   ├── 02_age_distribution.png
│   │   ├── 03_correlation_heatmap.png
│   │   ├── 04_chest_pain_analysis.png
│   │   ├── 05_gender_analysis.png
│   │   ├── 06_numerical_boxplots.png
│   │   ├── 07_heart_rate_vs_age.png
│   │   ├── 08_cholesterol_bp_distribution.png
│   │   ├── 09_exercise_features.png
│   │   ├── 10_oldpeak_analysis.png
│   │   └── 11_target_correlations.png
│   └── model_metrics/              # Model evaluation charts
│       ├── model_comparison.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── feature_importance.png
│       ├── cv_comparison.png
│       └── model_comparison.csv
│
├── 🚀 app.py                      # Streamlit web application
├── 📋 requirements.txt            # Python dependencies
├── ⚙️ setup.sh                    # Streamlit Cloud config
├── 🚫 .gitignore                  # Git ignore rules
└── 📖 README.md                   # This file
```

---

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core programming language |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | Data manipulation & numerical computing |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) | Static & interactive plots |
| **ML** | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | Model training, tuning, evaluation |
| **Deployment** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | Interactive web application |
| **Serialization** | ![Joblib](https://img.shields.io/badge/Joblib-3776AB?style=flat-square&logoColor=white) | Model persistence |

---

## 🔮 Future Improvements

- [ ] **Deep Learning** — Implement neural network models (PyTorch/TensorFlow) for comparison
- [ ] **Feature Engineering** — Create interaction features and polynomial terms
- [ ] **SHAP Values** — Add model explainability with SHAP force plots
- [ ] **Ensemble Methods** — Implement stacking/blending of top models
- [ ] **API Endpoint** — Add FastAPI REST endpoint for programmatic access
- [ ] **Docker** — Containerize the application for easy deployment
- [ ] **CI/CD** — Automated testing and deployment pipeline
- [ ] **A/B Testing** — Compare model versions in production
- [ ] **Class Imbalance** — Implement SMOTE or other resampling techniques
- [ ] **Cloud Deployment** — Deploy to Streamlit Cloud, AWS, or GCP

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** — Heart Disease dataset
- **Kaggle Community** — Data science inspiration and resources
- **Streamlit** — Amazing framework for data app deployment
- **Scikit-learn** — Comprehensive machine learning library

---

<p align="center">
  <strong>⭐ If you found this project useful, please consider giving it a star! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ for Data Science
</p>
