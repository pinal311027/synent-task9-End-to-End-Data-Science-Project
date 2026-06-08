"""
Model Building Module
=====================
Trains, evaluates, and compares multiple ML models for heart disease prediction.
Includes proper preprocessing, cross-validation, hyperparameter tuning,
and thorough validation to ensure predictions are correct.
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
)

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import MODELS_DIR, MODEL_METRICS_DIR, COLORS, PLOT_STYLE, print_section
from src.data_cleaning import get_clean_data

plt.rcParams.update(PLOT_STYLE)
DPI = 150


# ══════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════
def prepare_data(df: pd.DataFrame = None, test_size: float = 0.2):
    """Split and scale the data for training."""
    if df is None:
        df = get_clean_data()

    TARGET = "HeartDisease"
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # Store feature names for later
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index
    )

    print(f"📊 Training set: {X_train_scaled.shape[0]} samples")
    print(f"📊 Test set:     {X_test_scaled.shape[0]} samples")
    print(f"📊 Features:     {len(feature_names)}: {feature_names}")
    print(f"📊 Train target: 0={int((y_train==0).sum())}, 1={int((y_train==1).sum())}")
    print(f"📊 Test target:  0={int((y_test==0).sum())}, 1={int((y_test==1).sum())}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ══════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42, C=1.0, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, max_depth=10
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42, learning_rate=0.1, max_depth=5
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42, C=1.0, gamma="scale"
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, n_jobs=-1, weights="distance"
        ),
    }


# ══════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════
def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """Train all models and evaluate."""
    models = get_models()
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print_section("MODEL TRAINING & EVALUATION")

    for name, model in models.items():
        print(f"\n🤖 Training: {name}")
        print(f"   {'─' * 40}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "y_pred": y_pred,
            "y_prob": y_prob,
            "model": model,
        }
        results[name] = metrics

        print(f"   ✅ Accuracy:    {metrics['accuracy']:.4f}")
        print(f"   📏 Precision:   {metrics['precision']:.4f}")
        print(f"   🔄 Recall:      {metrics['recall']:.4f}")
        print(f"   ⚖️  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"   📈 ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"   🔀 CV Accuracy: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

        # Sanity check: make sure model predicts both classes
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            print(f"   ⚠️  WARNING: Model only predicts class {unique_preds[0]}!")
        else:
            print(f"   ✅ Predicts both classes: 0={int((y_pred==0).sum())}, 1={int((y_pred==1).sum())}")

    return results


# ══════════════════════════════════════════════
# HYPERPARAMETER TUNING
# ══════════════════════════════════════════════
def tune_best_models(X_train, y_train, results: dict) -> dict:
    """Tune hyperparameters for the top 3 models."""
    print_section("HYPERPARAMETER TUNING (Top 3 Models)")

    sorted_models = sorted(results.items(), key=lambda x: x[1]["f1_score"], reverse=True)
    top_3 = [name for name, _ in sorted_models[:3]]

    param_grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 0.5, 1, 5, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 8, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 1.0],
        },
        "SVM": {
            "C": [0.1, 0.5, 1, 5, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto", 0.01, 0.1],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuned_models = {}

    for name in top_3:
        if name not in param_grids:
            continue

        print(f"\n🔧 Tuning: {name}")
        base_class = results[name]["model"].__class__

        # Build base estimator with random_state if supported
        base_params = {}
        if hasattr(results[name]["model"], "random_state"):
            base_params["random_state"] = 42
        if name == "SVM":
            base_params["probability"] = True

        grid = GridSearchCV(
            base_class(**base_params),
            param_grids[name],
            cv=cv, scoring="f1", n_jobs=-1, verbose=0, refit=True,
        )
        grid.fit(X_train, y_train)

        print(f"   🏆 Best params: {grid.best_params_}")
        print(f"   📈 Best CV F1: {grid.best_score_:.4f}")

        tuned_models[name] = grid.best_estimator_

    return tuned_models


# ══════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ══════════════════════════════════════════════
def plot_model_comparison(results: dict) -> str:
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=PLOT_STYLE["figure.facecolor"])

    model_names = list(results.keys())
    metrics_list = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    colors = ["#2A9D8F", "#E63946", "#457B9D", "#F4A261", "#264653"]

    x = np.arange(len(model_names))
    width = 0.15

    for i, (metric, label, color) in enumerate(zip(metrics_list, labels, colors)):
        vals = [results[n][metric] for n in model_names]
        offset = (i - len(metrics_list) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=label, color=color,
               edgecolor="#ffffff22", linewidth=0.8, zorder=3)

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.3, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(MODEL_METRICS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 Saved: model_comparison.png")
    return path


def plot_confusion_matrices(results: dict) -> str:
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), facecolor=PLOT_STYLE["figure.facecolor"])
    axes = axes.flatten()

    for idx, (name, m) in enumerate(results.items()):
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"],
                    ax=axes[idx], linewidths=1, linecolor="#333366",
                    annot_kws={"fontsize": 14, "fontweight": "bold"})
        axes[idx].set_title(name, fontsize=13, fontweight="bold", pad=10)
        axes[idx].set_xlabel("Predicted", fontsize=11)
        axes[idx].set_ylabel("Actual", fontsize=11)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", color="#E0E0E0", y=1.02)
    plt.tight_layout()
    path = os.path.join(MODEL_METRICS_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 Saved: confusion_matrices.png")
    return path


def plot_roc_curves(results: dict, y_test) -> str:
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PLOT_STYLE["figure.facecolor"])
    colors = ["#2A9D8F", "#E63946", "#457B9D", "#F4A261", "#264653"]

    for idx, (name, m) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, m["y_prob"])
        roc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[idx % len(colors)],
                linewidth=2.5, label=f"{name} (AUC={roc_val:.3f})", zorder=3)

    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1.5, label="Random")
    ax.set_title("ROC Curves", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.legend(fontsize=11, framealpha=0.3, loc="lower right")
    ax.grid(alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(MODEL_METRICS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 Saved: roc_curves.png")
    return path


def plot_feature_importance(model, feature_names: list) -> str:
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PLOT_STYLE["figure.facecolor"])
    imp = model.feature_importances_
    idx = np.argsort(imp)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))

    ax.barh(range(len(idx)), imp[idx], color=colors, edgecolor="#ffffff22",
            linewidth=0.8, height=0.7, zorder=3)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=11)
    ax.set_title("Feature Importance (Best Model)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Importance Score", fontsize=13)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    for i, (ii, v) in enumerate(zip(idx, imp[idx])):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=10, color="#E0E0E0")

    plt.tight_layout()
    path = os.path.join(MODEL_METRICS_DIR, "feature_importance.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 Saved: feature_importance.png")
    return path


def plot_cv_comparison(X_train, y_train) -> str:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=PLOT_STYLE["figure.facecolor"])
    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = scores

    bp = ax.boxplot(list(cv_results.values()), labels=list(cv_results.keys()),
                    patch_artist=True, widths=0.5,
                    medianprops={"color": "#FFD700", "linewidth": 2},
                    whiskerprops={"color": "#AAAACC"}, capprops={"color": "#AAAACC"})
    box_colors = ["#2A9D8F", "#E63946", "#457B9D", "#F4A261", "#264653"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("#ffffff44")

    ax.set_title("5-Fold Cross-Validation", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_xticklabels(list(cv_results.keys()), rotation=15, ha="right", fontsize=11)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(MODEL_METRICS_DIR, "cv_comparison.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 Saved: cv_comparison.png")
    return path


# ══════════════════════════════════════════════
# SAVE MODEL + VALIDATION
# ══════════════════════════════════════════════
def save_best_model(results, tuned_models, scaler, feature_names, X_test, y_test):
    """Save the best model and validate it predicts correctly."""
    best_name = max(results, key=lambda k: results[k]["f1_score"])

    if best_name in tuned_models:
        best_model = tuned_models[best_name]
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"\n🏆 Best model (tuned): {best_name}")
        print(f"   Accuracy: {acc:.4f}, F1: {f1:.4f}")
    else:
        best_model = results[best_name]["model"]
        acc = results[best_name]["accuracy"]
        f1 = results[best_name]["f1_score"]
        print(f"\n🏆 Best model: {best_name}")
        print(f"   Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # ── VALIDATION: Test with extreme cases ──
    print(f"\n🔬 Validation — testing with extreme cases:")

    # Young healthy person → should predict NO disease
    healthy = pd.DataFrame([{
        "Age": 25, "Sex": 0, "ChestPainType": 0, "RestingBP": 110,
        "Cholesterol": 180, "FastingBS": 0, "RestingECG": 0,
        "MaxHR": 190, "ExerciseAngina": 0, "Oldpeak": 0.0, "ST_Slope": 0,
    }])
    # Old high-risk person → should predict disease
    risky = pd.DataFrame([{
        "Age": 70, "Sex": 1, "ChestPainType": 3, "RestingBP": 180,
        "Cholesterol": 350, "FastingBS": 1, "RestingECG": 2,
        "MaxHR": 90, "ExerciseAngina": 1, "Oldpeak": 4.0, "ST_Slope": 1,
    }])

    for name, case in [("Healthy 25F", healthy), ("Risky 70M", risky)]:
        X_case = scaler.transform(case)
        pred = best_model.predict(X_case)[0]
        prob = best_model.predict_proba(X_case)[0]
        label = "Heart Disease" if pred == 1 else "No Disease"
        print(f"   {name}: {label} (prob: {prob[0]:.2f}/{prob[1]:.2f})")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    # Save feature order for the app
    metadata = {
        "model_name": best_name,
        "model_type": type(best_model).__name__,
        "accuracy": float(acc),
        "precision": float(results[best_name]["precision"]),
        "recall": float(results[best_name]["recall"]),
        "f1_score": float(f1),
        "roc_auc": float(results[best_name]["roc_auc"]),
        "cv_mean": float(results[best_name]["cv_mean"]),
        "cv_std": float(results[best_name]["cv_std"]),
        "feature_names": feature_names,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n💾 Model:    {model_path}")
    print(f"💾 Scaler:   {scaler_path}")
    print(f"💾 Metadata: {meta_path}")
    return model_path


def save_metrics_table(results: dict) -> str:
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1_score"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
            "CV Mean": round(m["cv_mean"], 4),
            "CV Std": round(m["cv_std"], 4),
        })

    df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
    path = os.path.join(MODEL_METRICS_DIR, "model_comparison.csv")
    df.to_csv(path, index=False)
    print(f"\n📊 Model Comparison:\n{df.to_string(index=False)}")
    print(f"\n💾 Saved: {path}")
    return path


# ══════════════════════════════════════════════
# MASTER PIPELINE
# ══════════════════════════════════════════════
def run_model_pipeline(df=None):
    """Execute the complete model building pipeline."""
    if df is None:
        df = get_clean_data()

    print_section("DATA PREPARATION")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)

    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    tuned_models = tune_best_models(X_train, y_train, results)

    # Re-evaluate tuned models on test set
    print_section("TUNED MODEL EVALUATION")
    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"\n🔧 {name} (tuned):")
        print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
        print(f"   F1-Score:  {f1_score(y_test, y_pred):.4f}")
        print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
        # Update results with tuned model
        results[name]["model"] = model
        results[name]["accuracy"] = accuracy_score(y_test, y_pred)
        results[name]["precision"] = precision_score(y_test, y_pred)
        results[name]["recall"] = recall_score(y_test, y_pred)
        results[name]["f1_score"] = f1_score(y_test, y_pred)
        results[name]["roc_auc"] = roc_auc_score(y_test, y_prob)
        results[name]["y_pred"] = y_pred
        results[name]["y_prob"] = y_prob
        results[name]["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    # Visualizations
    print_section("GENERATING VISUALIZATIONS")
    os.makedirs(MODEL_METRICS_DIR, exist_ok=True)
    plot_model_comparison(results)
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    plot_cv_comparison(X_train, y_train)

    for name in ["Gradient Boosting", "Random Forest"]:
        if name in results:
            plot_feature_importance(results[name]["model"], feature_names)
            break

    # Save
    print_section("SAVING BEST MODEL")
    save_best_model(results, tuned_models, scaler, feature_names, X_test, y_test)
    save_metrics_table(results)

    print_section("✅ MODEL BUILDING COMPLETE")
    return results


if __name__ == "__main__":
    run_model_pipeline()
