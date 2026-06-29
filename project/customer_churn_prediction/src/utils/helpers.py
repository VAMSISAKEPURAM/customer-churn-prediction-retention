"""
Shared utility functions used across the pipeline.
"""

import json
import logging
import logging.config
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def set_seed(seed: int = 42):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(logging_config_path: str = "configs/logging_config.yaml"):
    """Initialize logging from YAML config."""
    Path("reports/logs").mkdir(parents=True, exist_ok=True)
    if Path(logging_config_path).exists():
        with open(logging_config_path) as f:
            log_cfg = yaml.safe_load(f)
        logging.config.dictConfig(log_cfg)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )


def save_json(data: dict, path: str):
    """Save a dict to JSON with pretty printing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)
