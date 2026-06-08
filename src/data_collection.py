"""
Data Collection Module
======================
Downloads and merges multiple heart disease datasets from Kaggle to create
a combined dataset of ~3500+ records for robust model training.

Sources:
  1. fedesoriano/heart-failure-prediction  (918 rows, 11 features)
  2. johnsmith88/heart-disease-dataset     (1025 rows, 13 features)
  3. redwankarimsony/heart-disease-data    (920 rows, mixed features)
"""

import os
import sys
import warnings
import urllib.request
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR, RAW_DATA_PATH, print_section


# ──────────────────────────────────────────────
# COMMON SCHEMA — 11 features + 1 target
# ──────────────────────────────────────────────
FINAL_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope", "HeartDisease",
]


def _download_kaggle_dataset(dataset_slug: str) -> str:
    """Download a dataset via kagglehub and return the local path."""
    try:
        import kagglehub
        path = kagglehub.dataset_download(dataset_slug)
        return path
    except Exception as e:
        print(f"   ⚠️ kagglehub failed for {dataset_slug}: {e}")
        return None


# ──────────────────────────────────────────────
# DATASET 1: fedesoriano (918 rows)
# Already has correct column names with string categoricals
# ──────────────────────────────────────────────
def _load_fedesoriano() -> pd.DataFrame:
    """Load fedesoriano/heart-failure-prediction dataset."""
    print("📥 [1/3] fedesoriano/heart-failure-prediction")

    path = _download_kaggle_dataset("fedesoriano/heart-failure-prediction")
    if path is None:
        return pd.DataFrame()

    csv_path = os.path.join(path, "heart.csv")
    if not os.path.exists(csv_path):
        print("   ⚠️ heart.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"   ✅ Loaded: {df.shape}")

    # Encode string categoricals → numeric
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["ChestPainType"] = df["ChestPainType"].map({"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3})
    df["RestingECG"] = df["RestingECG"].map({"Normal": 0, "ST": 1, "LVH": 2})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"N": 0, "Y": 1})
    df["ST_Slope"] = df["ST_Slope"].map({"Up": 0, "Flat": 1, "Down": 2})

    df = df[FINAL_COLUMNS]
    return df


# ──────────────────────────────────────────────
# DATASET 2: johnsmith88 (1025 rows)
# Columns: age, sex, cp, trestbps, chol, fbs, restecg,
#           thalach, exang, oldpeak, slope, ca, thal, target
# ──────────────────────────────────────────────
def _load_johnsmith88() -> pd.DataFrame:
    """Load johnsmith88/heart-disease-dataset."""
    print("📥 [2/3] johnsmith88/heart-disease-dataset")

    path = _download_kaggle_dataset("johnsmith88/heart-disease-dataset")
    if path is None:
        return pd.DataFrame()

    csv_path = os.path.join(path, "heart.csv")
    if not os.path.exists(csv_path):
        print("   ⚠️ heart.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"   ✅ Loaded: {df.shape}")

    # Rename to standard schema
    df = df.rename(columns={
        "age": "Age", "sex": "Sex", "cp": "ChestPainType",
        "trestbps": "RestingBP", "chol": "Cholesterol", "fbs": "FastingBS",
        "restecg": "RestingECG", "thalach": "MaxHR", "exang": "ExerciseAngina",
        "oldpeak": "Oldpeak", "slope": "ST_Slope", "target": "HeartDisease",
    })

    # Keep only the 11 common features + target
    df = df[FINAL_COLUMNS]
    return df


# ──────────────────────────────────────────────
# DATASET 3: redwankarimsony (920 rows)
# Columns: id, age, sex, dataset, cp, trestbps, chol, fbs,
#           restecg, thalch, exang, oldpeak, slope, ca, thal, num
# ──────────────────────────────────────────────
def _load_redwankarimsony() -> pd.DataFrame:
    """Load redwankarimsony/heart-disease-data (multi-source UCI)."""
    print("📥 [3/3] redwankarimsony/heart-disease-data")

    path = _download_kaggle_dataset("redwankarimsony/heart-disease-data")
    if path is None:
        return pd.DataFrame()

    csv_path = os.path.join(path, "heart_disease_uci.csv")
    if not os.path.exists(csv_path):
        print("   ⚠️ heart_disease_uci.csv not found")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"   ✅ Loaded: {df.shape}")

    # Encode string categoricals
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})

    cp_map = {"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3}
    df["cp"] = df["cp"].map(cp_map)

    ecg_map = {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2}
    df["restecg"] = df["restecg"].map(ecg_map)

    df["exang"] = df["exang"].map({"True": 1, "False": 0, True: 1, False: 0})

    slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
    df["slope"] = df["slope"].map(slope_map)

    df["fbs"] = df["fbs"].map({"True": 1, "False": 0, True: 1, False: 0})

    # Target: num (0-4) → binary (0 = no disease, 1+ = disease)
    df["num"] = (df["num"] > 0).astype(int)

    # Rename to standard schema
    df = df.rename(columns={
        "age": "Age", "sex": "Sex", "cp": "ChestPainType",
        "trestbps": "RestingBP", "chol": "Cholesterol", "fbs": "FastingBS",
        "restecg": "RestingECG", "thalch": "MaxHR", "exang": "ExerciseAngina",
        "oldpeak": "Oldpeak", "slope": "ST_Slope", "num": "HeartDisease",
    })

    df = df[FINAL_COLUMNS]
    return df


# ──────────────────────────────────────────────
# MERGE ALL DATASETS
# ──────────────────────────────────────────────
def download_dataset(force: bool = False) -> pd.DataFrame:
    """Download and merge all heart disease datasets."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_DATA_PATH) and not force:
        df = pd.read_csv(RAW_DATA_PATH)
        if df.shape[0] >= 2500:
            print(f"✅ Combined dataset already exists: {df.shape}")
            return df
        else:
            print(f"⚠️ Existing dataset too small ({df.shape[0]} rows). Re-downloading...")

    print_section("DOWNLOADING & MERGING DATASETS")

    dfs = []

    # Load each dataset
    for loader, name in [
        (_load_fedesoriano, "fedesoriano"),
        (_load_johnsmith88, "johnsmith88"),
        (_load_redwankarimsony, "redwankarimsony"),
    ]:
        try:
            df = loader()
            if not df.empty:
                dfs.append((name, df))
                print(f"   📊 {name}: {df.shape[0]} rows\n")
        except Exception as e:
            print(f"   ❌ {name} failed: {e}\n")

    if not dfs:
        print("❌ All downloads failed!")
        return pd.DataFrame()

    # Merge all datasets
    print_section("MERGING DATASETS")
    combined = pd.concat([df for _, df in dfs], ignore_index=True)
    print(f"📊 Combined raw: {combined.shape}")

    # Convert all columns to numeric
    for col in FINAL_COLUMNS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Drop rows with NaN in critical features
    n_before = len(combined)
    combined = combined.dropna().reset_index(drop=True)
    n_dropped = n_before - len(combined)
    if n_dropped > 0:
        print(f"   🗑️ Dropped {n_dropped} rows with NaN values")

    # Remove exact duplicates
    n_before = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    n_dupes = n_before - len(combined)
    if n_dupes > 0:
        print(f"   🗑️ Removed {n_dupes} duplicate rows")

    # Ensure integer types for categorical columns
    int_cols = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "ST_Slope", "HeartDisease"]
    for col in int_cols:
        combined[col] = combined[col].astype(int)

    print(f"\n📊 Final combined: {combined.shape}")
    print(f"   Sources: {', '.join(f'{n}({df.shape[0]})' for n, df in dfs)}")

    # Save
    combined.to_csv(RAW_DATA_PATH, index=False)
    print(f"✅ Saved to: {RAW_DATA_PATH}")

    return combined


def load_dataset() -> pd.DataFrame:
    """Load the raw heart disease dataset."""
    if not os.path.exists(RAW_DATA_PATH):
        return download_dataset()
    return pd.read_csv(RAW_DATA_PATH)


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print comprehensive dataset overview."""
    print_section("DATASET OVERVIEW")
    print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📋 Columns: {df.columns.tolist()}")
    print(f"\n📐 Data Types:\n{df.dtypes.to_string()}")
    print(f"\n📈 Statistics:\n{df.describe().to_string()}")

    missing = df.isnull().sum()
    print(f"\n❓ Missing Values: {missing.sum()}")

    print(f"\n🔄 Duplicates: {df.duplicated().sum()}")

    if "HeartDisease" in df.columns:
        print(f"\n🎯 Target Distribution:")
        dist = df["HeartDisease"].value_counts()
        for val, count in dist.items():
            pct = count / len(df) * 100
            label = "Heart Disease" if val == 1 else "No Disease"
            print(f"   {val} ({label}): {count} ({pct:.1f}%)")


if __name__ == "__main__":
    print_section("DATA COLLECTION — MULTI-SOURCE", char="🔹")
    df = download_dataset(force=True)
    inspect_dataset(df)
