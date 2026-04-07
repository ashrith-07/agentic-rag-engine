# ── Stage 1: builder ─────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml .

RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir --prefix=/install .

# ── Stage 2: runtime ─────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install system deps needed by pymupdf + spacy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/
COPY dashboard/ ./dashboard/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default: FastAPI. Override for dashboard.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
