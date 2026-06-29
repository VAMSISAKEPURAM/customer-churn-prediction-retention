<div align="center">

# 🔮 Customer Churn Prediction & Retention Model 

**A production-grade machine learning system that predicts customer churn probability with calibrated confidence scores — enabling data-driven retention campaigns and CRM prioritization.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Why This Project Matters](#-why-this-project-matters)
- [Architecture Overview](#️-architecture-overview)
- [Project Structure](#-project-structure)
- [Data Sources](#-data-sources)
- [ML Pipeline Design](#-ml-pipeline-design)
- [Encoding Strategy](#-encoding-strategy)
- [Model & Calibration](#-model--calibration)
- [Evaluation Metrics](#-evaluation-metrics)
- [REST API](#-rest-api)
- [Quick Start](#-quick-start)
- [Configuration](#️-configuration)
- [Running Tests](#-running-tests)
- [Docker Deployment](#-docker-deployment)
- [Tech Stack](#-tech-stack)
- [Project Roadmap](#-project-roadmap)

---

## 🎯 Problem Statement

> *"It costs 5× more to acquire a new customer than to retain an existing one."*

Customer churn is one of the most costly problems in subscription-based and e-commerce businesses. A single percentage point reduction in churn can translate to millions in recovered annual revenue.

Most churn models are one-off notebook experiments — they produce a static accuracy number but fail in production because they:
- Are not re-trainable without re-writing code
- Leak future information into training features
- Output uncalibrated probabilities that mislead business decisions
- Have no API for CRM integration
- Drift silently without alerting anyone

**This project solves all of that.** It delivers a fully reproducible, calibrated, API-served, drift-monitored churn scoring system ready for production deployment.

---

## 💡 Why This Project Matters

| Capability | Naive Notebook | This Project |
|---|:---:|:---:|
| Reproducible pipeline from raw CSV | ❌ | ✅ |
| No data leakage between train/test | ⚠️ | ✅ |
| Calibrated probability scores | ❌ | ✅ |
| Intelligent categorical encoding | ❌ | ✅ |
| Production REST API | ❌ | ✅ |
| Drift monitoring (PSI) | ❌ | ✅ |
| Docker containerized | ❌ | ✅ |
| Unit tested | ❌ | ✅ |
| CI/CD pipeline | ❌ | ✅ |
| Config-driven (no hardcoded values) | ❌ | ✅ |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER (7 CSVs)                          │
│  customer_churn • engagement_metrics • rfm • billing • orders       │
│  support_tickets • campaign_responses                               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  join on customer_id
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING LAYER                        │
│  DataLoader → DataValidator → join_all_sources()                    │
│  CategoricalEncoderPipeline (5 strategies) → FeatureSelector        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  stratified train/test split
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODELING LAYER                                │
│  build_model() → train_model() (with class balancing)               │
│  → calibrate_model() (isotonic/Platt) → save_model()               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐     ┌────────────────────────┐
│   EVALUATION LAYER  │     │    SERVING LAYER         │
│  AUROC • Brier      │     │  FastAPI REST API        │
│  PSI • Top-Decile   │     │  POST /predict           │
│  ROC Curve Plots    │     │  POST /predict/batch     │
│  Calibration Plots  │     │  GET  /health            │
└─────────────────────┘     └────────────────────────┘
```

---

## 📁 Project Structure

```
customer_churn_prediction/
│
├── 📄 README.md                          ← You are here
├── 📄 requirements.txt                   ← Runtime dependencies
├── 📄 requirements-dev.txt               ← Dev & test dependencies
├── 📄 setup.py                           ← Installable package + CLI entry points
├── 📄 Makefile                           ← One-command task runner
├── 📄 pytest.ini                         ← Test discovery config
├── 📄 .gitignore                         ← Excludes data, models, logs
│
├── 📁 configs/                           ← All configuration, no hardcoded values
│   ├── train_config.yaml                 ← Hyperparams, data paths, feature settings
│   ├── predict_config.yaml               ← Scoring pipeline config
│   └── logging_config.yaml              ← Rotating file + structured console logging
│
├── 📁 src/                               ← Production Python package (importable)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── ingest.py                     ← DataLoader, DataValidator, join_all_sources()
│   ├── features/
│   │   ├── __init__.py
│   │   ├── encode.py                     ← CategoricalEncoderPipeline (5 strategies)
│   │   └── feature_selection.py          ← FeatureSelector (variance + correlation)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                      ← build_model(), train_model(), calibrate_model()
│   │   └── predict.py                    ← ChurnScorer — batch inference class
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                    ← AUROC, Brier, PSI, top-decile lift, plots
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                    ← set_seed(), load_config(), setup_logging()
│
├── 📁 pipelines/                         ← Orchestration scripts (CLI entry points)
│   ├── __init__.py
│   ├── train_pipeline.py                 ← End-to-end training orchestration
│   └── predict_pipeline.py              ← Batch scoring orchestration
│
├── 📁 api/                               ← FastAPI serving layer
│   ├── __init__.py
│   └── main.py                           ← REST endpoints: /health, /predict, /predict/batch
│
├── 📁 tests/
│   ├── conftest.py                       ← Shared pytest fixtures
│   ├── unit/
│   │   ├── test_metrics.py               ← Tests for AUROC, Brier, PSI, top-decile lift
│   │   └── test_encoding.py              ← Tests for encoding strategies & Cramer's V
│   └── integration/                      ← End-to-end pipeline tests (coming soon)
│
├── 📁 notebooks/                         ← Exploratory analysis (non-production)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_model_evaluation.ipynb
│
├── 📁 docker/
│   ├── Dockerfile                        ← Python 3.11 slim, non-root, health check
│   └── docker-compose.yml               ← API container with model/config volume mounts
│
├── 📁 .github/workflows/
│   └── ci.yml                            ← GitHub Actions: lint + unit tests on push/PR
│
├── 📁 data/
│   ├── raw/                              ← Immutable source CSVs (not committed to git)
│   ├── interim/                          ← Intermediate transformed data
│   ├── processed/                        ← Final features ready for modeling
│   └── external/                         ← Third-party reference data
│
├── 📁 models/
│   └── registry/                         ← Saved model artifacts after training
│       ├── churn_model_latest.pkl
│       ├── categorical_encoders.pkl
│       ├── feature_names.json
│       └── model_metadata.json
│
└── 📁 reports/                           ← Auto-generated reports & plots
    ├── test_metrics.json
    ├── test_metrics.txt
    ├── test_roc_curve.png
    ├── test_calibration_curve.png
    ├── scored_customers.csv
    └── logs/pipeline.log
```

---

## 📊 Data Sources

This project merges **7 relational CSV files** on `customer_id` to build a rich feature table:

| File | Description | Key Features |
|---|---|---|
| `customer_churn.csv` | **Target table** — churn labels + behavioral data | `churned`, `tenure_months`, `nps_score`, `support_contacts` |
| `customer_rfm.csv` | Recency, Frequency, Monetary segmentation | `rfm_segment`, `recency_days`, `frequency`, `monetary_value` |
| `customer_engagement_metrics.csv` | Rolling engagement KPIs | `engagement_tier`, `logins_30d`, `email_opens`, `feature_adoption_score` |
| `subscription_billing.csv` | Billing & contract data | `plan_type`, `billing_cadence`, `payment_failures`, `contract_type` |
| `orders.csv` | Purchase history & velocity | `order_count`, `total_spend`, `last_order_date` |
| `support_tickets.csv` | Customer support case data | `ticket_count`, `open_tickets`, `avg_resolution_days` |
| `campaign_responses.csv` | Retention campaign exposure & response | `offer_accepted`, `campaign_type`, `discount_amount` |

> **Note:** All data is **synthetic** and intended for educational / portfolio use only. No real PII is present.

---

## 🔧 ML Pipeline Design

The training pipeline is orchestrated end-to-end in [`pipelines/train_pipeline.py`](pipelines/train_pipeline.py):

```
Step 1: Load & Validate Raw Data
       ↓  DataLoader reads all 7 CSVs from data/raw/
       ↓  DataValidator checks: required columns, duplicate IDs, missing rate

Step 2: Join All Sources
       ↓  Left-join all tables on customer_id
       ↓  Final merged table: ~50+ features

Step 3: Stratified Train / Test Split  (80% / 20%)
       ↓  Stratified on 'churned' to preserve class balance

Step 4: Categorical Encoding  ← fit ONLY on train, apply to both
       ↓  5-strategy intelligent encoder (see below)

Step 5: Feature Selection
       ↓  Drop low-variance + highly correlated features

Step 6: Model Training  (80% of train set)
       ↓  XGBoost with balanced class weights + early stopping

Step 7: Probability Calibration  (20% of train set)
       ↓  Isotonic regression calibration (CalibratedClassifierCV)

Step 8: Evaluation on Test Set
       ↓  AUROC, Brier score, top-decile lift, PSI
       ↓  ROC curve + calibration curve plots saved to reports/

Step 9: Save All Artifacts
       ↓  churn_model_latest.pkl
       ↓  categorical_encoders.pkl
       ↓  feature_names.json
       ↓  model_metadata.json
```

---

## 🧬 Encoding Strategy

The [`CategoricalEncoderPipeline`](src/features/encode.py) applies the most appropriate encoding strategy per column automatically — **fit only on train data** to prevent leakage:

| Strategy | Trigger | Example Columns | Output |
|---|---|---|---|
| **Ordinal** | Column is in `ordinal_columns` config | `rfm_segment`, `engagement_tier` | `col_ord` (integer rank) |
| **Target** | Cramer's V with target > 0.15 | High-signal categoricals | `col_te` (churn rate per category) |
| **One-Hot** | Cardinality ≤ 10 | `gender`, `contract_type` | `col_ohe_Male`, `col_ohe_Female` … |
| **Frequency** | Cardinality 11–50 | `country`, `plan_type` | `col_freq` (proportion of train rows) |
| **Hash** | Cardinality > 50 | Free-text fields, IDs | `col_hash` (MD5 mod 100) |

**Cramer's V** (a bias-corrected version of χ²) is used to measure categorical-to-target association without being fooled by cardinality:

```
V = √(φ²_corrected / min(k-1, r-1))

Where φ²_corrected removes bias from contingency table dimensions
```

All encoders are **serialized to `models/registry/categorical_encoders.pkl`** and loaded identically at scoring time — no transformation drift between training and production.

---

## 🤖 Model & Calibration

### Algorithm: XGBoost Gradient Boosting

Chosen for its:
- Robustness to mixed feature types and missing values
- Built-in `scale_pos_weight` for class imbalance
- Early stopping to prevent overfitting
- Fast inference for batch scoring

### Handling Class Imbalance

Churn datasets are heavily imbalanced (often 5–15% churn rate). This is handled in two ways:
1. **`compute_sample_weight(class_weight='balanced')`** — reweights minority class during training
2. **Stratified splits** — preserves class ratio across train/val/test/calibration splits

### Probability Calibration

Raw XGBoost scores are uncalibrated — a score of 0.7 does **not** mean "70% chance of churning". Without calibration, business thresholds are meaningless.

**Isotonic Regression calibration** is applied on a held-out calibration set (20% of train):

```python
CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
```

A well-calibrated model means: *among all customers scored 0.7, approximately 70% will actually churn.*

---

## 📈 Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| **AUROC** | Area under the ROC curve — discriminatory power | > 0.80 |
| **Brier Score** | Calibration quality (lower is better) | < 0.15 |
| **Avg Precision** | Precision-Recall AUC — better for imbalanced classes | > 0.60 |
| **Top-Decile Lift** | Churn rate in top 10% vs. random — business value | > 3× |
| **PSI** | Population Stability Index — detects feature/score drift | < 0.10 |

### Understanding PSI (Drift Monitoring)

| PSI Range | Interpretation | Action |
|---|---|---|
| < 0.10 | No significant shift | Monitor normally |
| 0.10 – 0.20 | Moderate shift detected | Investigate features |
| > 0.20 | Significant population drift | Retrain model |

---

## 🌐 REST API

The FastAPI scoring service ([`api/main.py`](api/main.py)) exposes the following endpoints:

### `GET /health`
Returns the API status and model load state.
```json
{
  "status": "ok",
  "model_loaded": true,
  "api_version": "0.1.0"
}
```

### `GET /model/info`
Returns training metadata — algorithm, metrics, version.
```json
{
  "algorithm": "xgboost",
  "calibration": "isotonic",
  "num_features": 47,
  "train_metrics": { "auroc": 0.8812, "brier_score": 0.1034 },
  "project_version": "0.1.0"
}
```

### `POST /predict`
Score a single customer in real time.
```json
// Request
{
  "customer_id": "CUST_0042",
  "features": {
    "tenure_months": 14,
    "monthly_spend": 89.50,
    "payment_failures": 2,
    "num_support_tickets": 5,
    "logins_30d": 1
  }
}

// Response
{
  "customer_id": "CUST_0042",
  "churn_probability": 0.7834,
  "risk_decile": 9,
  "risk_label": "High Risk"
}
```

### `POST /predict/batch`
Score hundreds of customers in a single API call.
```json
// Response
{
  "total": 500,
  "predictions": [
    { "customer_id": "CUST_0001", "churn_probability": 0.23, "risk_decile": 3 },
    { "customer_id": "CUST_0042", "churn_probability": 0.78, "risk_decile": 9 }
  ]
}
```

**Interactive API Docs** (auto-generated by FastAPI):
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip
- (Optional) Docker Desktop

### 1. Clone & Setup

```bash
# Navigate to the project directory
cd customer_churn_prediction

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install runtime + dev dependencies
make install-dev

# OR manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 3. Add Your Data

Copy the 7 source CSV files into `data/raw/`:

```
data/raw/
  ├── customer_churn.csv
  ├── customer_engagement_metrics.csv
  ├── customer_rfm.csv
  ├── subscription_billing.csv
  ├── orders.csv
  ├── support_tickets.csv
  └── campaign_responses.csv
```

### 4. Train the Model

```bash
make train
# OR:
python pipelines/train_pipeline.py --config configs/train_config.yaml
```

**What happens:**
- All 7 CSVs are loaded, validated, and joined
- Categorical features are intelligently encoded
- XGBoost is trained with early stopping + class balancing
- Probabilities are calibrated with isotonic regression
- AUROC, Brier score, top-decile lift are printed and saved to `reports/`
- All model artifacts are saved to `models/registry/`

### 5. Score New Customers (Batch)

```bash
make predict
# OR:
python pipelines/predict_pipeline.py --config configs/predict_config.yaml
```

Output: `reports/scored_customers.csv` with columns:
`customer_id | churn_probability | risk_decile | score_date`

### 6. Start the Scoring API

```bash
make serve
# OR:
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` to explore the API interactively.

---

## ⚙️ Configuration

All pipeline behavior is controlled through YAML files in `configs/` — **no hardcoded values anywhere in the codebase**.

### `configs/train_config.yaml` — Key Settings

```yaml
model:
  algorithm: "xgboost"            # Switch to lightgbm or logistic_regression
  calibration_method: "isotonic"  # or "sigmoid" or "none"
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.05
    scale_pos_weight: auto         # Auto-balances class imbalance

features:
  encoding:
    target_encoding_threshold: 0.15  # Cramer's V threshold for target encoding
    ohe_cardinality_limit: 10
    frequency_cardinality_limit: 50
    hash_bins: 100

  ordinal_columns:
    rfm_segment: [Lost, At Risk, New, Promising, Loyal, Champions]
    engagement_tier: [dormant, low, medium, high]

data:
  train_test_split:
    test_size: 0.20
    stratify: true
```

### Switching Algorithms

To switch from XGBoost to LightGBM, change one line in `train_config.yaml`:

```yaml
model:
  algorithm: "lightgbm"   # was: xgboost
```

No code changes required.

---

## 🧪 Running Tests

```bash
# Run all unit tests with coverage
make test

# Run only unit tests
make test-unit

# Run with verbose output
pytest tests/ -v --tb=short

# Run a specific test file
pytest tests/unit/test_encoding.py -v
```

### Test Coverage

| Module | Tests |
|---|---|
| `src/evaluation/metrics.py` | AUROC range, Brier range, PSI drift detection, top-decile lift, required keys |
| `src/features/encode.py` | Cramer's V perfect/no association, ordinal encoding, no data leakage, encoding plan |

---

## 🐳 Docker Deployment

### Build & Run with Docker

```bash
# Build the image
make docker-build

# Run the scoring API container
make docker-run

# Check the API is healthy
curl http://localhost:8000/health
```

### Docker Compose (Recommended)

```bash
cd docker/
docker compose up --build
```

The container:
- Uses Python 3.11 slim base
- Runs as a **non-root user** (security best practice)
- Has a built-in **health check** (`/health` endpoint)
- Starts `uvicorn` with 2 workers for concurrency
- Mounts `models/` and `configs/` as read-only volumes

---

## 🛠️ Tech Stack

| Category | Library | Purpose |
|---|---|---|
| **Data** | `pandas`, `numpy` | DataFrame manipulation, numerical operations |
| **Statistics** | `scipy` | Cramer's V (chi-squared contingency) |
| **Modeling** | `xgboost`, `lightgbm` | Gradient boosting classifiers |
| **ML Utilities** | `scikit-learn` | Encoding, calibration, metrics, selection |
| **Explainability** | `shap` | Feature importance & SHAP values |
| **Serialization** | `joblib` | Efficient model & encoder persistence |
| **Configuration** | `pyyaml` | YAML config loading |
| **API** | `fastapi`, `uvicorn` | REST API serving |
| **Validation** | `pydantic` | Request/response schema validation |
| **Visualization** | `matplotlib` | ROC curve & calibration curve plots |
| **Testing** | `pytest`, `pytest-cov` | Unit tests + coverage reporting |
| **Code Quality** | `black`, `flake8`, `isort` | Formatting & linting |
| **CI/CD** | GitHub Actions | Automated test & lint on every push |
| **Container** | Docker, docker-compose | Reproducible deployment |

---

## 🗺️ Project Roadmap

### ✅ Phase 1 — Foundation (Complete)
- [x] Production folder structure
- [x] 7-source data ingestion & validation
- [x] Intelligent 5-strategy categorical encoder
- [x] Feature selection pipeline
- [x] XGBoost training with class balancing
- [x] Isotonic probability calibration
- [x] AUROC, Brier, PSI, top-decile lift metrics
- [x] ROC curve & calibration plots
- [x] FastAPI scoring service
- [x] Docker containerization
- [x] GitHub Actions CI pipeline
- [x] Unit tests for metrics & encoding

### 🔄 Phase 2 — Advanced Modeling (In Progress)
- [ ] SHAP feature importance plots & explanations
- [ ] Hyperparameter tuning with Optuna
- [ ] Cross-validation framework (StratifiedKFold)
- [ ] LightGBM comparison & model selection
- [ ] Campaign response propensity modeling
- [ ] Integration tests for full pipeline

### 🔮 Phase 3 — MLOps & Monitoring (Planned)
- [ ] MLflow experiment tracking
- [ ] Automated PSI drift alerts
- [ ] Scheduled retraining pipeline (Prefect / Airflow)
- [ ] Feature store integration
- [ ] A/B testing framework for model versions
- [ ] Grafana dashboard for score distribution monitoring

---

## 📖 References

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*
- Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting Good Probabilities with Supervised Learning*
- Cramér, H. (1946). *Mathematical Methods of Statistics* — Cramer's V statistic
- Gini, C. (1912). *Variabilità e mutabilità* — Population Stability Index basis

---

## 👤 Author

**Vamsi** — Machine Learning Engineer

Built as part of the EDUKRON ML Portfolio Series.

---

<div align="center">

**⭐ Star this repo if it helped you build production ML systems!**

*"The difference between a model that works in a notebook and one that works in production is everything."*

</div>
