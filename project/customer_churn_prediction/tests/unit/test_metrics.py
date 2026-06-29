"""
Unit tests for the evaluation metrics module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.evaluation.metrics import compute_metrics, compute_psi, _top_decile_lift


class TestComputeMetrics:
    def setup_method(self):
        np.random.seed(42)
        self.y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        self.y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.4, 0.75, 0.15])

    def test_metrics_returns_dict(self):
        metrics = compute_metrics(self.y_true, self.y_prob)
        assert isinstance(metrics, dict)

    def test_auroc_range(self):
        metrics = compute_metrics(self.y_true, self.y_prob)
        assert 0.0 <= metrics["auroc"] <= 1.0

    def test_brier_score_range(self):
        metrics = compute_metrics(self.y_true, self.y_prob)
        assert 0.0 <= metrics["brier_score"] <= 1.0

    def test_perfect_predictions(self):
        metrics = compute_metrics(
            np.array([0, 1, 0, 1]),
            np.array([0.01, 0.99, 0.01, 0.99])
        )
        assert metrics["auroc"] == 1.0

    def test_required_keys(self):
        metrics = compute_metrics(self.y_true, self.y_prob)
        required = ["auroc", "brier_score", "precision", "recall", "f1", "top_decile_lift"]
        for key in required:
            assert key in metrics, f"Missing key: {key}"


class TestPSI:
    def test_no_drift(self):
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.1, "PSI should be near 0 for identical distributions"

    def test_high_drift(self):
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(5, 1, 1000)
        psi = compute_psi(expected, actual)
        assert psi > 0.2, "PSI should be high for clearly different distributions"


class TestTopDecileLift:
    def test_lift_above_one(self):
        y_true = np.array([1] * 20 + [0] * 80)
        y_prob = np.concatenate([np.random.uniform(0.8, 1.0, 20), np.random.uniform(0, 0.4, 80)])
        lift = _top_decile_lift(y_true, y_prob)
        assert lift > 1.0, "Top-decile lift should be > 1 for a good model"

    def test_lift_is_float(self):
        y_true = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.1, 0.7, 0.4, 0.3, 0.1])
        lift = _top_decile_lift(y_true, y_prob)
        assert isinstance(lift, float)
