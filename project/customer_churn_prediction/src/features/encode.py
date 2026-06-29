"""
Feature engineering module.
Handles encoding of categorical variables, numerical transformations,
and feature selection for the churn prediction pipeline.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

logger = logging.getLogger(__name__)


# ==============================================================
# Statistical Helpers
# ==============================================================

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Cramer's V statistic for categorical-categorical association.
    Used to decide if a column is strongly correlated with the target.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    if min((kcorr - 1), (rcorr - 1)) == 0:
        return 0.0
    return float(np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))))


# ==============================================================
# Production Categorical Encoder
# ==============================================================

class CategoricalEncoderPipeline:
    """
    Intelligent, production-safe categorical encoding pipeline.

    Encoding Strategy:
        - Ordinal     → Columns with inherent rank order (e.g., rfm_segment)
        - Target      → High Cramer's V with target (>threshold), fit only on train
        - One-Hot     → Low cardinality (≤ ohe_limit)
        - Frequency   → Medium cardinality (≤ freq_limit)
        - Hash        → High cardinality catch-all

    Usage:
        encoder = CategoricalEncoderPipeline(config)
        df_train_enc, df_test_enc = encoder.fit_transform(df_train, df_test)
        encoder.save("models/registry/categorical_encoders.pkl")
    """

    def __init__(self, config: dict):
        enc_cfg = config["features"]["encoding"]
        self.target_col: str = config["data"]["target_column"]
        self.target_threshold: float = enc_cfg.get("target_encoding_threshold", 0.15)
        self.ohe_limit: int = enc_cfg.get("ohe_cardinality_limit", 10)
        self.freq_limit: int = enc_cfg.get("frequency_cardinality_limit", 50)
        self.hash_bins: int = enc_cfg.get("hash_bins", 100)

        self.ordinal_cols: List[str] = list(config["features"].get("ordinal_columns", {}).keys())
        self.ordinal_categories: Dict[str, list] = config["features"].get("ordinal_columns", {})

        # Fitted state (persisted as artifacts)
        self.encoding_plan: Dict[str, str] = {}
        self.sklearn_encoders: Dict[str, object] = {}
        self.target_means: Dict[str, dict] = {}
        self.freq_maps: Dict[str, dict] = {}
        self.global_mean: float = 0.0

    def fit_transform(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit encoders on train, transform both train and test."""
        df_train = df_train.copy()
        df_test = df_test.copy()

        self.global_mean = float(df_train[self.target_col].mean())
        logger.info(f"Global churn rate (train): {self.global_mean:.4f}")

        cat_cols = df_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if self.target_col in cat_cols:
            cat_cols.remove(self.target_col)
        logger.info(f"Encoding {len(cat_cols)} categorical columns...")

        for col in cat_cols:
            df_train[col] = df_train[col].astype(str).replace("nan", "Unknown")
            df_test[col] = df_test[col].astype(str).replace("nan", "Unknown")
            unique_vals = df_train[col].nunique()

            if col in self.ordinal_cols:
                df_train, df_test = self._ordinal_encode(df_train, df_test, col)

            elif self._compute_cv(df_train, col) > self.target_threshold:
                df_train, df_test = self._target_encode(df_train, df_test, col)

            elif unique_vals <= self.ohe_limit:
                df_train, df_test = self._ohe_encode(df_train, df_test, col)

            elif unique_vals <= self.freq_limit:
                df_train, df_test = self._frequency_encode(df_train, df_test, col)

            else:
                df_train, df_test = self._hash_encode(df_train, df_test, col)

            logger.debug(f"  [{col}] → {self.encoding_plan[col]} encoding")

        logger.info(f"Encoding complete. Encoding plan: {self.encoding_plan}")
        return df_train, df_test

    def save(self, path: str):
        """Persist all fitted encoder state to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "encoding_plan": self.encoding_plan,
            "sklearn_encoders": self.sklearn_encoders,
            "target_means": self.target_means,
            "freq_maps": self.freq_maps,
            "global_mean": self.global_mean,
            "hash_bins": self.hash_bins,
        }
        joblib.dump(artifact, path)
        logger.info(f"Encoder artifacts saved to: {path}")

    @classmethod
    def load(cls, path: str, config: dict) -> "CategoricalEncoderPipeline":
        """Load a previously fitted encoder from disk."""
        artifact = joblib.load(path)
        enc = cls(config)
        enc.encoding_plan = artifact["encoding_plan"]
        enc.sklearn_encoders = artifact["sklearn_encoders"]
        enc.target_means = artifact["target_means"]
        enc.freq_maps = artifact["freq_maps"]
        enc.global_mean = artifact["global_mean"]
        enc.hash_bins = artifact["hash_bins"]
        logger.info(f"Encoder loaded from: {path}")
        return enc

    # ---- Private Encoding Methods ----

    def _compute_cv(self, df: pd.DataFrame, col: str) -> float:
        unique_vals = df[col].nunique()
        if len(df) / unique_vals > 5 and unique_vals > 1:
            return cramers_v(df[col], df[self.target_col])
        return 0.0

    def _ordinal_encode(self, train, test, col):
        cats = self.ordinal_categories.get(col, sorted(train[col].unique().tolist()))
        enc = OrdinalEncoder(
            categories=[cats],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        train[f"{col}_ord"] = enc.fit_transform(train[[col]])
        test[f"{col}_ord"] = enc.transform(test[[col]])
        self.sklearn_encoders[col] = enc
        self.encoding_plan[col] = "Ordinal"
        return train.drop(columns=[col]), test.drop(columns=[col])

    def _target_encode(self, train, test, col):
        means = train.groupby(col)[self.target_col].mean().to_dict()
        self.target_means[col] = means
        train[f"{col}_te"] = train[col].map(means).fillna(self.global_mean)
        test[f"{col}_te"] = test[col].map(means).fillna(self.global_mean)
        self.encoding_plan[col] = "Target"
        return train.drop(columns=[col]), test.drop(columns=[col])

    def _ohe_encode(self, train, test, col):
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_enc = enc.fit_transform(train[[col]])
        test_enc = enc.transform(test[[col]])
        col_names = [f"{col}_ohe_{cat}" for cat in enc.categories_[0]]
        train = pd.concat([train, pd.DataFrame(train_enc, columns=col_names, index=train.index)], axis=1)
        test = pd.concat([test, pd.DataFrame(test_enc, columns=col_names, index=test.index)], axis=1)
        self.sklearn_encoders[col] = enc
        self.encoding_plan[col] = "One-Hot"
        return train.drop(columns=[col]), test.drop(columns=[col])

    def _frequency_encode(self, train, test, col):
        freq = train[col].value_counts(normalize=True).to_dict()
        self.freq_maps[col] = freq
        train[f"{col}_freq"] = train[col].map(freq).fillna(0)
        test[f"{col}_freq"] = test[col].map(freq).fillna(0)
        self.encoding_plan[col] = "Frequency"
        return train.drop(columns=[col]), test.drop(columns=[col])

    def _hash_encode(self, train, test, col):
        def _hash(x):
            return int(hashlib.md5(str(x).encode("utf-8")).hexdigest(), 16) % self.hash_bins

        train[f"{col}_hash"] = train[col].apply(_hash)
        test[f"{col}_hash"] = test[col].apply(_hash)
        self.encoding_plan[col] = "Hash"
        return train.drop(columns=[col]), test.drop(columns=[col])
