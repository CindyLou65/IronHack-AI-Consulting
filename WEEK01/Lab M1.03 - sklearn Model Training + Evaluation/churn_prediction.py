"""
churn_prediction.py
Telco Customer Churn — KNN model (Part 2 Unguided Exercise)

What this script includes (per deliverables):
- Data loading + exploration (shape, dtypes, missing values, target distribution)
- Simple visualizations (saved as PNGs)
- Data preprocessing (TotalCharges fix, missing values, encoding, feature/target split)
- Train/test split (80/20, stratify, random_state=42)
- KNN training + evaluation (accuracy, precision, recall, confusion matrix, classification report)
- Experiment with different K values and pick best K (by accuracy and by recall)
- Short analysis/conclusions printed at the end

How to run:
python churn_prediction.py

Make sure Telco-Customer-Churn.csv is in the same folder as this script,
or change DATA_PATH below.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

DATA_PATH = "Telco-Customer-Churn.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def explore_data(df: pd.DataFrame) -> None:
    print("=== DATA LOADING & EXPLORATION ===")
    print("Shape (rows, columns):", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column (NaN only):")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing > 0] if (missing > 0).any() else "✓ No missing values found (as NaN).")

    print("\nTarget distribution (Churn):")
    print(df["Churn"].value_counts())
    print("\nTarget distribution (%):")
    print((df["Churn"].value_counts(normalize=True) * 100).round(2))

    # Visual 1: Churn distribution
    plt.figure()
    df["Churn"].value_counts().plot(kind="bar")
    plt.title("Churn distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("churn_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Visual 2: Tenure distribution by churn
    plt.figure()
    df[df["Churn"] == "No"]["tenure"].hist(bins=30, alpha=0.5, label="No")
    df[df["Churn"] == "Yes"]["tenure"].hist(bins=30, alpha=0.5, label="Yes")
    plt.title("Tenure distribution by churn")
    plt.xlabel("tenure")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tenure_by_churn.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nSaved plots: churn_distribution.png, tenure_by_churn.png")


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print("\n=== DATA PREPROCESSING ===")

    # 1) Fix TotalCharges (string -> numeric, blanks -> NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 2) Handle missing TotalCharges (small number of rows -> fill with median)
    n_missing = df["TotalCharges"].isnull().sum()
    print(f"Missing TotalCharges after conversion: {n_missing}")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3) Convert target to 0/1 (Churn: Yes -> 1, No -> 0)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 4) Drop customerID (identifier, not predictive)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # 5) One-hot encode categoricals
    df_encoded = pd.get_dummies(df, drop_first=True)

    # 6) Separate X/y
    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]

    print("Encoded data shape:", df_encoded.shape)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # Quick sanity check: no missing values
    remaining_missing = df_encoded.isnull().sum().sum()
    print("Remaining missing values (total across all cols):", remaining_missing)

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series):
    print("\n=== TRAIN / TEST SPLIT ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])

    print("\nTraining churn distribution (%):")
    print((y_train.value_counts(normalize=True) * 100).round(2))

    print("\nTest churn distribution (%):")
    print((y_test.value_counts(normalize=True) * 100).round(2))

    return X_train, X_test, y_train, y_test


def train_and_evaluate_knn(X_train, X_test, y_train, y_test, k: int) -> dict:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_test_pred = knn.predict(X_test)

    # Positive class is churn (1)
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, pos_label=1)
    rec = recall_score(y_test, y_test_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_test_pred)

    print("\n=== MODEL EVALUATION (KNN) ===")
    print(f"K = {k}")
    print(f"Test Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")

    print("\nConfusion Matrix (rows=Actual, cols=Predicted)")
    print("               Predicted")
    print("            No Churn  Churn")
    print(f"Actual No Churn   {cm[0,0]:4d}     {cm[0,1]:4d}")
    print(f"Actual Churn      {cm[1,0]:4d}     {cm[1,1]:4d}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["No Churn", "Churn"]))

    return {
        "K": k,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "ConfusionMatrix": cm
    }


def experiment_k_values(X_train, X_test, y_train, y_test, k_values: list[int]) -> pd.DataFrame:
    print("\n=== EXPERIMENTING WITH DIFFERENT K VALUES ===")
    results = []

    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train, y_train)
        y_pred = knn_temp.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)

        results.append({"K": k, "Accuracy": acc, "Precision": prec, "Recall": rec})
        print(f"K={k:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

    results_df = pd.DataFrame(results)

    # Choose best K by accuracy and by recall (both reported)
    best_k_acc = int(results_df.loc[results_df["Accuracy"].idxmax(), "K"])
    best_acc = float(results_df["Accuracy"].max())

    best_k_rec = int(results_df.loc[results_df["Recall"].idxmax(), "K"])
    best_rec = float(results_df["Recall"].max())

    print(f"\nBest K based on Accuracy: {best_k_acc} (Accuracy={best_acc:.4f})")
    print(f"Best K based on Recall:   {best_k_rec} (Recall={best_rec:.4f})")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["K"], results_df["Accuracy"], marker="o", label="Accuracy")
    plt.plot(results_df["K"], results_df["Precision"], marker="s", label="Precision")
    plt.plot(results_df["K"], results_df["Recall"], marker="^", label="Recall")
    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel("Score")
    plt.title("KNN Performance vs K (Churn Prediction)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_k_comparison_churn.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nSaved plot: knn_k_comparison_churn.png")

    # Save table for evidence
    results_df.to_csv("knn_k_results_churn.csv", index=False)
    print("Saved table: knn_k_results_churn.csv")

    return results_df


def print_analysis_summary(best_by_acc: dict, best_k_results_df: pd.DataFrame) -> None:
    """
    Short written-style summary printed to terminal (you can paste into your report).
    """
    print("\n=== ANALYSIS & RECOMMENDATIONS (SHORT SUMMARY) ===")

    k = best_by_acc["K"]
    acc = best_by_acc["Accuracy"]
    prec = best_by_acc["Precision"]
    rec = best_by_acc["Recall"]
    cm = best_by_acc["ConfusionMatrix"]

    print(f"- Chosen K (by accuracy): {k}")
    print(f"- Test Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"- Precision (Churn=1): {prec:.4f}")
    print(f"- Recall (Churn=1):    {rec:.4f}")
    print("\nConfusion Matrix (rows=Actual, cols=Predicted):")
    print(cm)

    print("\nKey findings about churn (high-level patterns to look for):")
    print("- Short tenure customers tend to churn more often.")
    print("- Month-to-month contracts often correlate with higher churn.")
    print("- Higher monthly charges can be associated with churn risk.")
    print("- Add-on services (e.g., tech support / security) may relate to lower churn.")

    print("\nBusiness recommendations (based on model + typical churn patterns):")
    print("- Focus retention offers on high-risk segments (short tenure, month-to-month, high monthly charges).")
    print("- Consider discounts or contract upgrades to reduce churn.")
    print("- Use the model as an early warning system to prioritize outreach.")

    print("\nLimitations / future improvements:")
    print("- Recall for churn may be limited due to class imbalance and KNN’s distance-based nature.")
    print("- Feature scaling could improve KNN performance (KNN is sensitive to scale).")
    print("- Try other models (logistic regression, random forest, gradient boosting) for stronger churn recall and interpretability.")
    print("- Use cross-validation when selecting K for more reliable tuning.")


def main():
    # 1) Load + explore
    df = load_data(DATA_PATH)
    print("Dataset loaded successfully!")
    explore_data(df)

    # 2) Preprocess
    X, y = preprocess_data(df)

    # 3) Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4–5) Train baseline KNN (k=5) + evaluate
    baseline = train_and_evaluate_knn(X_train, X_test, y_train, y_test, k=5)

    # 6) Experiment with different K values
    k_values = [1, 3, 5, 7, 9, 11, 15]
    results_df = experiment_k_values(X_train, X_test, y_train, y_test, k_values)

    # Choose K=9 if you want to match your earlier best-by-accuracy result
    best_k_acc = int(results_df.loc[results_df["Accuracy"].idxmax(), "K"])
    best_model_summary = train_and_evaluate_knn(X_train, X_test, y_train, y_test, k=best_k_acc)

    # 7) Print short analysis/recommendations summary
    print_analysis_summary(best_model_summary, results_df)


if __name__ == "__main__":
    main()
