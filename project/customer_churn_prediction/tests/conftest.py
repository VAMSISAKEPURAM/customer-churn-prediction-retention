"""
conftest.py — Shared test fixtures for the entire test suite.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_binary_data():
    """Returns a simple y_true / y_prob pair for metric tests."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 200)
    y_prob = np.clip(y_true * 0.5 + np.random.normal(0, 0.3, 200), 0, 1)
    return y_true, y_prob


@pytest.fixture(scope="session")
def mock_config():
    return {
        "data": {
            "target_column": "churned",
            "id_column": "customer_id",
            "raw_dir": "data/raw",
            "train_test_split": {"test_size": 0.2, "stratify": True},
        },
        "features": {
            "encoding": {
                "target_encoding_threshold": 0.15,
                "ohe_cardinality_limit": 10,
                "frequency_cardinality_limit": 50,
                "hash_bins": 100,
            },
            "ordinal_columns": {
                "engagement_tier": ["dormant", "low", "medium", "high"]
            },
            "drop_columns": [],
        },
        "project": {"random_seed": 42},
    }


@pytest.fixture(scope="session")
def sample_customer_df():
    """Returns a small synthetic customer feature DataFrame."""
    np.random.seed(0)
    n = 100
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "churned": np.random.randint(0, 2, n),
        "tenure_months": np.random.randint(1, 60, n),
        "monthly_spend": np.random.uniform(10, 500, n),
        "num_support_tickets": np.random.randint(0, 20, n),
        "engagement_tier": np.random.choice(["dormant", "low", "medium", "high"], n),
        "payment_failures": np.random.randint(0, 5, n),
    })
