# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Healthcare AI Assistant
#
# Multi-stage build:
#   Stage 1 (builder): Install all Python deps into /opt/venv
#   Stage 2 (runtime): Slim image — copies only venv + app code
#
# Fixes applied vs original:
#   - pydantic bumped to 2.9.2 (ollama>=0.4.8 requires pydantic>=2.9)
#   - pip upgraded to latest before dependency resolution
#   - faiss-cpu version relaxed to >=1.7.4 to avoid manylinux conflicts
#   - torch installed as CPU-only variant (no CUDA, smaller image)
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for Docker layer cache
COPY requirements.txt .

# Upgrade pip to latest (resolves backtracking issues in older pip versions)
RUN pip install --upgrade pip wheel setuptools

# Install CPU-only torch first (avoids pulling CUDA wheels — saves ~3 GB)
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install all remaining dependencies
# --no-cache-dir keeps image lean
# pip's backtracking resolver handles transitive conflicts automatically
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Healthcare AI Team"
LABEL description="Healthcare AI Assistant — RAG-powered clinical information system"
LABEL version="1.0.0"

WORKDIR /app

# Minimal runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Copy application source
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/documents/ ./data/documents/

# Writable directories for index + logs
RUN mkdir -p data/faiss_index logs && \
    chown -R appuser:appuser /app

COPY .env.example .env.example

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default: start FastAPI backend
# Frontend is started as a separate service in docker-compose
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
