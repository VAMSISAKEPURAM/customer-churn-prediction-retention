"""
FastAPI serving application for the Customer Churn Prediction model.
Exposes REST endpoints for real-time and batch scoring.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health       → Health check
    GET  /model/info   → Model metadata
    POST /predict      → Score a single customer
    POST /predict/batch → Score a batch of customers
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.predict import ChurnScorer
from src.utils.helpers import load_config

logger = logging.getLogger("api")

# ---- App Setup ----
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production REST API for real-time and batch churn probability scoring.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load Artifacts at Startup ----
CONFIG_PATH = "configs/predict_config.yaml"
config = load_config(CONFIG_PATH)

try:
    scorer = ChurnScorer.from_config(config)
    logger.info("Model loaded successfully at API startup.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    scorer = None


# ---- Pydantic Schemas ----
class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    features: Dict[str, float] = Field(..., description="Feature name-value pairs")


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_decile: int
    risk_label: str


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


# ---- Endpoints ----

@app.get("/health", tags=["System"])
def health_check():
    """Returns API health status and model load state."""
    return {
        "status": "ok",
        "model_loaded": scorer is not None,
        "api_version": "0.1.0",
    }


@app.get("/model/info", tags=["System"])
def model_info():
    """Returns model metadata."""
    meta_path = Path("models/registry/model_metadata.json")
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    with open(meta_path) as f:
        metadata = json.load(f)
    return metadata


@app.post("/predict", response_model=PredictionResponse, tags=["Scoring"])
def predict_single(customer: CustomerFeatures):
    """Score a single customer for churn probability."""
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training pipeline first.")

    df = pd.DataFrame([customer.features])
    df.insert(0, "customer_id", customer.customer_id)

    try:
        scored = scorer.score(df)
        row = scored.iloc[0]
        prob = float(row["churn_probability"])
        decile = int(row["risk_decile"])
        label = "High Risk" if prob >= 0.7 else "Medium Risk" if prob >= 0.4 else "Low Risk"
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_probability=prob,
            risk_decile=decile,
            risk_label=label,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Scoring"])
def predict_batch(request: BatchRequest):
    """Score a batch of customers for churn probability."""
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    rows = []
    for c in request.customers:
        row = {"customer_id": c.customer_id}
        row.update(c.features)
        rows.append(row)

    df = pd.DataFrame(rows)

    try:
        scored = scorer.score(df)
        results = scored.to_dict(orient="records")
        return {"total": len(results), "predictions": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def start():
    """Entry point for CLI launch via setup.py console_scripts."""
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
