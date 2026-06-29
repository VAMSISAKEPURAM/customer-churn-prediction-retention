"""
Evaluation module.
Computes all business and statistical model performance metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ==============================================================
# Core Metrics
# ==============================================================

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Compute a full suite of classification and calibration metrics.

    Returns:
        dict with keys: auroc, brier_score, avg_precision, precision, recall, f1,
                        top_decile_lift, confusion_matrix
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auroc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "avg_precision": round(float(average_precision_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "top_decile_lift": round(float(_top_decile_lift(y_true, y_prob)), 4),
        "threshold": threshold,
    }

    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k:<25}: {v}")
    logger.info("=" * 50)
    return metrics


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute the Population Stability Index (PSI) to detect feature/score drift.
    PSI < 0.1: No shift | 0.1–0.2: Moderate shift | >0.2: Significant shift
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = (expected_counts / len(expected)).clip(min=1e-6)
    actual_pct = (actual_counts / len(actual)).clip(min=1e-6)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    logger.info(f"PSI: {psi:.4f}")
    return psi


def _top_decile_lift(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute top-decile lift: how much better than random in the top 10% of scored customers."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df = df.sort_values("y_prob", ascending=False)
    top_decile_n = max(1, int(len(df) * 0.10))
    top_decile_rate = df.head(top_decile_n)["y_true"].mean()
    overall_rate = df["y_true"].mean()
    if overall_rate == 0:
        return 0.0
    return top_decile_rate / overall_rate


# ==============================================================
# Report Generation
# ==============================================================

def save_metrics_report(metrics: dict, output_dir: str, split: str = "test"):
    """Save metrics dict to JSON and a human-readable text report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{split}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved: {json_path}")

    txt_path = output_dir / f"{split}_metrics.txt"
    with open(txt_path, "w") as f:
        f.write(f"Model Evaluation Report — {split.upper()} SET\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"  {k:<25}: {v}\n")
    logger.info(f"Text report saved: {txt_path}")


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str, split: str = "test"):
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}", color="#2196F3", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {split} Set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = Path(output_dir) / f"{split}_roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved: {path}")


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: str, split: str = "test"):
    """Generate and save probability calibration (reliability) curve."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, "s-", color="#E91E63", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve — {split} Set")
    plt.legend()
    plt.tight_layout()
    path = Path(output_dir) / f"{split}_calibration_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Calibration curve saved: {path}")
