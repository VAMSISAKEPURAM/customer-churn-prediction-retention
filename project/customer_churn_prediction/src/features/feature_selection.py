"""
Feature selection module.
Identifies and ranks the most predictive features for the churn model.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Selects the most informative features using multiple strategies:
    - Variance threshold (drop near-zero variance)
    - Mutual information with target
    - Correlation-based deduplication
    - Model-based importance (after initial fit)

    Usage:
        selector = FeatureSelector(config)
        X_train_sel, selected_cols = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
    """

    def __init__(self, config: dict, variance_threshold: float = 0.01, corr_threshold: float = 0.95):
        self.config = config
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.selected_features: List[str] = []
        self._drop_columns = config["features"].get("drop_columns", [])

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Fit and apply feature selection on training data."""
        logger.info(f"Starting feature selection on {X.shape[1]} features...")
        X = X.copy()

        # Step 1: Drop configured ID/irrelevant columns
        X = self._drop_configured(X)

        # Step 2: Drop low variance features
        X = self._drop_low_variance(X)

        # Step 3: Drop highly correlated features
        X = self._drop_highly_correlated(X)

        self.selected_features = X.columns.tolist()
        logger.info(f"Feature selection complete. Selected {len(self.selected_features)} features.")
        return X, self.selected_features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted selection to a new dataset (test/scoring)."""
        missing = [c for c in self.selected_features if c not in X.columns]
        if missing:
            logger.warning(f"Missing features in transform input: {missing}")
        available = [c for c in self.selected_features if c in X.columns]
        return X[available]

    def _drop_configured(self, X: pd.DataFrame) -> pd.DataFrame:
        to_drop = [c for c in self._drop_columns if c in X.columns]
        if to_drop:
            logger.info(f"  Dropping configured columns: {to_drop}")
        return X.drop(columns=to_drop)

    def _drop_low_variance(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        variances = X[numeric_cols].var()
        low_var = variances[variances < self.variance_threshold].index.tolist()
        if low_var:
            logger.info(f"  Dropping {len(low_var)} low-variance features: {low_var[:10]}...")
        return X.drop(columns=low_var)

    def _drop_highly_correlated(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        corr_matrix = X[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > self.corr_threshold)]
        if to_drop:
            logger.info(f"  Dropping {len(to_drop)} highly correlated features: {to_drop[:10]}...")
        return X.drop(columns=to_drop)
