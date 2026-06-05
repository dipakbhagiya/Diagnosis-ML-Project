# ZeroError Diagnosis — evaluate.py
# Full evaluation: metrics, confusion matrix, feature importance

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, accuracy_score
)

def evaluate_model(model, X_test, y_test, feature_names):
    """Run full evaluation suite and print/plot results."""

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n📊 ZeroError Evaluation Report")
    print("=" * 40)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("=" * 40)
    print(classification_report(y_test, y_pred,
          target_names=["No Disease", "Disease"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title("ZeroError — Confusion Matrix")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.show()

    # ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ZeroError — ROC Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=150)
    plt.show()

    # Feature importance (for tree models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(feature_names)),
                importances[indices], color="steelblue")
        plt.xticks(range(len(feature_names)),
                   [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.title("ZeroError — Feature Importance")
        plt.tight_layout()
        plt.savefig("results/feature_importance.png", dpi=150)
        plt.show()