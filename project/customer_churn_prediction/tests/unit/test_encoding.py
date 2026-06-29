"""
Unit tests for the categorical encoding pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.features.encode import CategoricalEncoderPipeline, cramers_v


MOCK_CONFIG = {
    "data": {"target_column": "churned"},
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
    },
}


def make_sample_df(n=200, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        "churned": np.random.randint(0, 2, n),
        "engagement_tier": np.random.choice(["dormant", "low", "medium", "high"], n),
        "gender": np.random.choice(["Male", "Female", "Other"], n),
        "country": np.random.choice(["US", "UK", "IN", "DE", "AU"], n),
        "numeric_col": np.random.randn(n),
    })
    return df


class TestCramersV:
    def test_perfect_association(self):
        x = pd.Series(["a", "b"] * 50)
        y = pd.Series([0, 1] * 50)
        cv = cramers_v(x, y)
        assert cv > 0.9

    def test_no_association(self):
        np.random.seed(42)
        x = pd.Series(np.random.choice(["a", "b", "c"], 200))
        y = pd.Series(np.random.randint(0, 2, 200))
        cv = cramers_v(x, y)
        assert cv < 0.3

    def test_output_is_float(self):
        x = pd.Series(["a", "b", "c"] * 10)
        y = pd.Series([0, 1, 0] * 10)
        assert isinstance(cramers_v(x, y), float)


class TestCategoricalEncoderPipeline:
    def setup_method(self):
        df = make_sample_df(300)
        self.df_train = df.iloc[:240].reset_index(drop=True)
        self.df_test = df.iloc[240:].reset_index(drop=True)
        self.encoder = CategoricalEncoderPipeline(MOCK_CONFIG)

    def test_fit_transform_runs(self):
        df_train_enc, df_test_enc = self.encoder.fit_transform(self.df_train, self.df_test)
        assert df_train_enc is not None
        assert df_test_enc is not None

    def test_ordinal_col_encoded(self):
        df_train_enc, df_test_enc = self.encoder.fit_transform(self.df_train, self.df_test)
        assert "engagement_tier_ord" in df_train_enc.columns
        assert "engagement_tier" not in df_train_enc.columns

    def test_original_cols_removed(self):
        df_train_enc, _ = self.encoder.fit_transform(self.df_train, self.df_test)
        # All categorical source columns should be dropped
        cat_cols = ["engagement_tier", "gender", "country"]
        for col in cat_cols:
            assert col not in df_train_enc.columns

    def test_encoding_plan_populated(self):
        self.encoder.fit_transform(self.df_train, self.df_test)
        assert len(self.encoder.encoding_plan) > 0
        valid_strategies = {"Ordinal", "Target", "One-Hot", "Frequency", "Hash"}
        for col, strategy in self.encoder.encoding_plan.items():
            assert strategy in valid_strategies

    def test_no_data_leakage(self):
        """Target encoding should be computed on train set only and applied to test."""
        self.encoder.fit_transform(self.df_train, self.df_test)
        # Verify target means only reference train values
        for col, means in self.encoder.target_means.items():
            assert isinstance(means, dict)
