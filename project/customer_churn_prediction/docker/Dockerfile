# ==============================================================
# Dockerfile — Customer Churn Prediction API
# ==============================================================

FROM python:3.11-slim

LABEL maintainer="Vamsi"
LABEL description="Production container for Customer Churn Prediction scoring API"

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# ---- System Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ---- Python Dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Application Code ----
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY models/ ./models/
COPY setup.py .

RUN pip install --no-cache-dir -e .

# ---- Non-root User for Security ----
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# ---- Expose Port ----
EXPOSE 8000

# ---- Health Check ----
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# ---- Start Server ----
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
