# Docker image for IRF Kriging FastAPI service.
# Build from repository root with:
#   docker build -t irf-kriging-api project
# Run with:
#   docker run --rm -p 8000:8000 irf-kriging-api

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system packages for scientific Python wheels and health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first to improve Docker build caching.
COPY requirements.txt /app/project/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /app/project/requirements.txt

# Copy the project package, including src/, artifacts/, tests/ if present.
COPY . /app/project

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "project.src.service.main:app", "--host", "0.0.0.0", "--port", "8000"]
