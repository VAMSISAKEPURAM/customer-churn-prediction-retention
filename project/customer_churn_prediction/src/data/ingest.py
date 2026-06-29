"""
Data ingestion module.
Loads, validates, and joins all raw source CSV files.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads raw data files as defined in the config and performs basic schema validation.
    
    Usage:
        loader = DataLoader(config)
        dfs = loader.load_all()
    """

    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = Path(config["data"]["raw_dir"])

    def load_file(self, filename: str) -> pd.DataFrame:
        """Load a single CSV file with logging."""
        filepath = self.raw_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
        logger.info(f"Loading: {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"  → Shape: {df.shape}")
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all source files defined in the config."""
        source_files = self.config["data"]["source_files"]
        dfs = {}
        for name, filename in source_files.items():
            dfs[name] = self.load_file(filename)
        logger.info(f"Loaded {len(dfs)} source files.")
        return dfs


class DataValidator:
    """
    Validates loaded DataFrames for schema, missing values, and type correctness.
    
    Usage:
        validator = DataValidator()
        validator.validate(df, name="customer_churn")
    """

    REQUIRED_SCHEMAS = {
        "customer_churn": {
            "required_columns": ["customer_id", "churned"],
            "id_column": "customer_id",
        },
        "rfm": {
            "required_columns": ["customer_id", "rfm_segment"],
            "id_column": "customer_id",
        },
    }

    def validate(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Run all validation checks for a given DataFrame."""
        logger.info(f"Validating '{name}'...")
        schema = self.REQUIRED_SCHEMAS.get(name)
        if schema:
            self._check_required_columns(df, schema["required_columns"], name)
            self._check_duplicates(df, schema["id_column"], name)
        self._check_missing_rate(df, name, threshold=0.5)
        logger.info(f"  '{name}' passed validation.")
        return df

    def _check_required_columns(self, df: pd.DataFrame, required: list, name: str):
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"[{name}] Missing required columns: {missing}")

    def _check_duplicates(self, df: pd.DataFrame, id_col: str, name: str):
        dup_count = df[id_col].duplicated().sum()
        if dup_count > 0:
            logger.warning(f"[{name}] Found {dup_count} duplicate '{id_col}' values.")

    def _check_missing_rate(self, df: pd.DataFrame, name: str, threshold: float = 0.5):
        high_missing = df.columns[df.isnull().mean() > threshold].tolist()
        if high_missing:
            logger.warning(f"[{name}] Columns with >{threshold*100:.0f}% missing: {high_missing}")


def join_all_sources(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merges all source dataframes on customer_id into a single feature table.
    
    Args:
        dfs: Dictionary of {name: DataFrame} as returned by DataLoader.load_all()
    
    Returns:
        Merged DataFrame (final customer feature table)
    """
    logger.info("Joining all source tables on customer_id...")
    base = dfs["customer_churn"].copy()

    join_map = {
        "rfm": "customer_id",
        "engagement_metrics": "customer_id",
        "billing": "customer_id",
        "orders": "customer_id",
        "support_tickets": "customer_id",
        "campaign_responses": "customer_id",
    }

    for table_name, join_key in join_map.items():
        if table_name in dfs:
            before = len(base)
            base = base.merge(dfs[table_name], on=join_key, how="left", suffixes=("", f"_{table_name}"))
            logger.info(f"  Joined '{table_name}': {before} → {len(base)} rows")

    logger.info(f"Final merged shape: {base.shape}")
    return base
