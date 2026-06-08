"""
Exploratory Data Analysis Module
=================================
Generates 11 publication-quality visualizations for the Heart Failure dataset.
Designed for the fedesoriano dataset (918 rows, 11 features + 1 target).
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import EDA_PLOTS_DIR, COLORS, PLOT_STYLE, print_section
from src.data_cleaning import get_clean_data

plt.rcParams.update(PLOT_STYLE)
PALETTE = [COLORS["positive"], COLORS["negative"]]
TARGET_LABELS = {0: "No Disease", 1: "Heart Disease"}
DPI = 150
FIG_SIZE = (12, 7)


def _save(fig, name):
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)
    path = os.path.join(EDA_PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 Saved: {name}.png")
    return path


# ── 1. Target Distribution ──
def plot_target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PLOT_STYLE["figure.facecolor"])
    counts = df["HeartDisease"].value_counts().sort_index()

    bars = axes[0].bar([TARGET_LABELS[i] for i in counts.index], counts.values,
                        color=PALETTE, edgecolor="#ffffff22", width=0.6, zorder=3)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height()+5,
                     str(val), ha="center", fontweight="bold", fontsize=14, color="#E0E0E0")
    axes[0].set_title("Class Distribution", fontsize=16, fontweight="bold", pad=15)
    axes[0].set_ylabel("Count", fontsize=13)
    axes[0].grid(axis="y", alpha=0.2, zorder=0)
    axes[0].set_axisbelow(True)

    axes[1].pie(counts.values, labels=[TARGET_LABELS[i] for i in counts.index],
                colors=PALETTE, autopct="%1.1f%%", startangle=90, explode=(0.03, 0.03),
                textprops={"fontsize": 13, "color": "#E0E0E0"},
                wedgeprops={"edgecolor": "#0E1117", "linewidth": 2})
    axes[1].set_title("Class Proportion", fontsize=16, fontweight="bold", pad=15)

    fig.suptitle("Heart Disease — Target Variable", fontsize=18, fontweight="bold", color="#E0E0E0", y=1.02)
    plt.tight_layout()
    return _save(fig, "01_target_distribution")


# ── 2. Age Distribution ──
def plot_age_distribution(df):
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=PLOT_STYLE["figure.facecolor"])
    for label, color in zip([0, 1], PALETTE):
        ax.hist(df[df["HeartDisease"]==label]["Age"], bins=25, alpha=0.65,
                label=TARGET_LABELS[label], color=color, edgecolor="#ffffff33", zorder=3)
    ax.set_title("Age Distribution by Heart Disease", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Age (years)", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.legend(fontsize=12, framealpha=0.3)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return _save(fig, "02_age_distribution")


# ── 3. Correlation Heatmap ──
def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=PLOT_STYLE["figure.facecolor"])
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
                center=0, annot=True, fmt=".2f", square=True, linewidths=1,
                linecolor="#333366", cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 10}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    return _save(fig, "03_correlation_heatmap")


# ── 4. Chest Pain Analysis ──
def plot_chest_pain_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=PLOT_STYLE["figure.facecolor"])
    cp_labels = {0: "Typical\nAngina", 1: "Atypical\nAngina", 2: "Non-anginal\nPain", 3: "Asymptomatic"}
    df_plot = df.copy()
    df_plot["CP_Label"] = df_plot["ChestPainType"].map(cp_labels)

    ct = pd.crosstab(df_plot["CP_Label"], df_plot["HeartDisease"])
    ct.columns = [TARGET_LABELS[c] for c in ct.columns]
    ct.plot(kind="bar", ax=axes[0], color=PALETTE, edgecolor="#ffffff22", width=0.7, zorder=3)
    axes[0].set_title("Chest Pain Type vs Heart Disease", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Chest Pain Type", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(fontsize=11, framealpha=0.3)
    axes[0].grid(axis="y", alpha=0.2, zorder=0)
    axes[0].set_axisbelow(True)

    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_norm.plot(kind="barh", stacked=True, ax=axes[1], color=PALETTE, edgecolor="#ffffff22", zorder=3)
    axes[1].set_title("Heart Disease Rate by CP Type", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Percentage (%)", fontsize=12)
    axes[1].legend(fontsize=11, framealpha=0.3, loc="lower right")
    axes[1].grid(axis="x", alpha=0.2, zorder=0)
    axes[1].set_axisbelow(True)

    plt.tight_layout()
    return _save(fig, "04_chest_pain_analysis")


# ── 5. Gender Analysis ──
def plot_gender_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PLOT_STYLE["figure.facecolor"])
    sex_labels = {1: "Male", 0: "Female"}
    df_plot = df.copy()
    df_plot["Sex_Label"] = df_plot["Sex"].map(sex_labels)

    ct = pd.crosstab(df_plot["Sex_Label"], df_plot["HeartDisease"])
    ct.columns = [TARGET_LABELS[c] for c in ct.columns]
    ct.plot(kind="bar", ax=axes[0], color=PALETTE, edgecolor="#ffffff22", width=0.6, zorder=3)
    axes[0].set_title("Gender vs Heart Disease", fontsize=14, fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(fontsize=11, framealpha=0.3)
    axes[0].grid(axis="y", alpha=0.2, zorder=0)
    axes[0].set_axisbelow(True)

    rates = df.groupby("Sex")["HeartDisease"].mean() * 100
    bars = axes[1].bar([sex_labels[i] for i in rates.index], rates.values,
                        color=[COLORS["secondary"], COLORS["primary"]],
                        edgecolor="#ffffff22", width=0.5, zorder=3)
    for bar, val in zip(bars, rates.values):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                     f"{val:.1f}%", ha="center", fontweight="bold", fontsize=13, color="#E0E0E0")
    axes[1].set_title("Heart Disease Rate by Gender", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Rate (%)", fontsize=12)
    axes[1].grid(axis="y", alpha=0.2, zorder=0)
    axes[1].set_axisbelow(True)
    plt.tight_layout()
    return _save(fig, "05_gender_analysis")


# ── 6. Numerical Box Plots ──
def plot_numerical_boxplots(df):
    feats = [f for f in ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"] if f in df.columns]
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 6), facecolor=PLOT_STYLE["figure.facecolor"])
    if n == 1: axes = [axes]

    for ax, feat in zip(axes, feats):
        groups = [df[df["HeartDisease"]==0][feat], df[df["HeartDisease"]==1][feat]]
        bp = ax.boxplot(groups, labels=["No Disease", "Disease"], patch_artist=True, widths=0.5,
                        medianprops={"color": "#FFD700", "linewidth": 2},
                        whiskerprops={"color": "#AAAACC"}, capprops={"color": "#AAAACC"},
                        flierprops={"markerfacecolor": "#FF6B6B", "markersize": 4, "alpha": 0.6})
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("#ffffff44")
        ax.set_title(feat, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Numerical Features — Box Plots by Disease Status",
                 fontsize=16, fontweight="bold", color="#E0E0E0", y=1.02)
    plt.tight_layout()
    return _save(fig, "06_numerical_boxplots")


# ── 7. Heart Rate vs Age ──
def plot_heart_rate_vs_age(df):
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor=PLOT_STYLE["figure.facecolor"])
    for label, color, marker in zip([0, 1], PALETTE, ["o", "X"]):
        sub = df[df["HeartDisease"]==label]
        ax.scatter(sub["Age"], sub["MaxHR"], c=color, label=TARGET_LABELS[label],
                   alpha=0.55, s=50, edgecolors="#ffffff33", linewidth=0.5, marker=marker, zorder=3)
        z = np.polyfit(sub["Age"], sub["MaxHR"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sub["Age"].min(), sub["Age"].max(), 100)
        ax.plot(x_line, p(x_line), color=color, linestyle="--", linewidth=2, alpha=0.8, zorder=4)

    ax.set_title("Max Heart Rate vs Age", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Age (years)", fontsize=13)
    ax.set_ylabel("Max Heart Rate (bpm)", fontsize=13)
    ax.legend(fontsize=12, framealpha=0.3)
    ax.grid(alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return _save(fig, "07_heart_rate_vs_age")


# ── 8. Cholesterol & BP Distribution ──
def plot_cholesterol_bp(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=PLOT_STYLE["figure.facecolor"])
    features = [("Cholesterol", "Serum Cholesterol (mg/dl)"), ("RestingBP", "Resting BP (mm Hg)")]
    for ax, (feat, title) in zip(axes, features):
        if feat not in df.columns: continue
        for label, color in zip([0, 1], PALETTE):
            ax.hist(df[df["HeartDisease"]==label][feat], bins=30, alpha=0.6,
                    label=TARGET_LABELS[label], color=color, edgecolor="#ffffff22", zorder=3)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(feat, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(fontsize=11, framealpha=0.3)
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.set_axisbelow(True)
    fig.suptitle("Health Metrics Distribution", fontsize=17, fontweight="bold", color="#E0E0E0", y=1.02)
    plt.tight_layout()
    return _save(fig, "08_cholesterol_bp_distribution")


# ── 9. Exercise Features ──
def plot_exercise_features(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PLOT_STYLE["figure.facecolor"])

    angina_labels = {0: "No", 1: "Yes"}
    df_plot = df.copy()
    df_plot["Angina"] = df_plot["ExerciseAngina"].map(angina_labels)
    ct = pd.crosstab(df_plot["Angina"], df_plot["HeartDisease"])
    ct.columns = [TARGET_LABELS[c] for c in ct.columns]
    ct.plot(kind="bar", ax=axes[0], color=PALETTE, edgecolor="#ffffff22", width=0.6, zorder=3)
    axes[0].set_title("Exercise-Induced Angina", fontsize=14, fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(fontsize=11, framealpha=0.3)
    axes[0].grid(axis="y", alpha=0.2, zorder=0)
    axes[0].set_axisbelow(True)

    slope_labels = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    df_plot["Slope"] = df_plot["ST_Slope"].map(slope_labels)
    ct2 = pd.crosstab(df_plot["Slope"], df_plot["HeartDisease"])
    ct2.columns = [TARGET_LABELS[c] for c in ct2.columns]
    ct2.plot(kind="bar", ax=axes[1], color=PALETTE, edgecolor="#ffffff22", width=0.6, zorder=3)
    axes[1].set_title("ST Slope Segment", fontsize=14, fontweight="bold")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(fontsize=11, framealpha=0.3)
    axes[1].grid(axis="y", alpha=0.2, zorder=0)
    axes[1].set_axisbelow(True)

    fig.suptitle("Exercise-Related Features", fontsize=17, fontweight="bold", color="#E0E0E0", y=1.02)
    plt.tight_layout()
    return _save(fig, "09_exercise_features")


# ── 10. Oldpeak Analysis ──
def plot_oldpeak_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=PLOT_STYLE["figure.facecolor"])
    for label, color in zip([0, 1], PALETTE):
        axes[0].hist(df[df["HeartDisease"]==label]["Oldpeak"], bins=25, alpha=0.6,
                     label=TARGET_LABELS[label], color=color, edgecolor="#ffffff22", density=True, zorder=3)
    axes[0].set_title("Oldpeak Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("ST Depression", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].legend(fontsize=11, framealpha=0.3)
    axes[0].grid(axis="y", alpha=0.2, zorder=0)
    axes[0].set_axisbelow(True)

    parts = axes[1].violinplot([df[df["HeartDisease"]==0]["Oldpeak"], df[df["HeartDisease"]==1]["Oldpeak"]],
                                positions=[0, 1], showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("#FFD700")
    parts["cmedians"].set_color("#FFFFFF")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["No Disease", "Heart Disease"])
    axes[1].set_title("Oldpeak Violin Plot", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Oldpeak Value", fontsize=12)
    axes[1].grid(axis="y", alpha=0.2, zorder=0)
    axes[1].set_axisbelow(True)

    fig.suptitle("ST Depression (Oldpeak) Analysis", fontsize=17, fontweight="bold", color="#E0E0E0", y=1.02)
    plt.tight_layout()
    return _save(fig, "10_oldpeak_analysis")


# ── 11. Feature-Target Correlations ──
def plot_target_correlations(df):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=PLOT_STYLE["figure.facecolor"])
    corr = df.select_dtypes(include=[np.number]).corr()["HeartDisease"].drop("HeartDisease").sort_values()
    colors_bar = [COLORS["positive"] if v > 0 else COLORS["secondary"] for v in corr.values]

    bars = ax.barh(corr.index, corr.values, color=colors_bar, edgecolor="#ffffff22", height=0.6, zorder=3)
    for bar, val in zip(bars, corr.values):
        x = val + 0.01 if val >= 0 else val - 0.01
        ax.text(x, bar.get_y()+bar.get_height()/2, f"{val:.3f}",
                ha="left" if val >= 0 else "right", va="center", fontsize=10, color="#E0E0E0")

    ax.axvline(x=0, color="#AAAACC", linewidth=1, zorder=2)
    ax.set_title("Feature Correlation with Heart Disease", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Pearson Correlation", fontsize=13)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return _save(fig, "11_target_correlations")


# ══════════════════════════════════════════════
def run_full_eda(df=None):
    if df is None:
        df = get_clean_data()

    print_section("EXPLORATORY DATA ANALYSIS")
    print(f"📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"📁 Plots → {EDA_PLOTS_DIR}\n")

    funcs = [
        ("Target Distribution", plot_target_distribution),
        ("Age Distribution", plot_age_distribution),
        ("Correlation Heatmap", plot_correlation_heatmap),
        ("Chest Pain Analysis", plot_chest_pain_analysis),
        ("Gender Analysis", plot_gender_analysis),
        ("Numerical Box Plots", plot_numerical_boxplots),
        ("Heart Rate vs Age", plot_heart_rate_vs_age),
        ("Cholesterol & BP", plot_cholesterol_bp),
        ("Exercise Features", plot_exercise_features),
        ("Oldpeak Analysis", plot_oldpeak_analysis),
        ("Target Correlations", plot_target_correlations),
    ]

    plots = []
    for i, (name, func) in enumerate(funcs, 1):
        print(f"\n📈 [{i}/{len(funcs)}] {name}:")
        try:
            plots.append(func(df))
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n✅ EDA complete! {len(plots)} visualizations generated.")
    return plots


if __name__ == "__main__":
    run_full_eda()
