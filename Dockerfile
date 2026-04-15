# Dockerfile  (root of repo — for Hugging Face Spaces)
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 curl supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install project
COPY pyproject.toml .
RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir . \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY dashboard/ ./dashboard/
COPY data/evaluation/ ./data/evaluation/

# Create required directories
RUN mkdir -p data/raw data/processed data/logs

# Supervisor config — runs FastAPI + Streamlit together
RUN mkdir -p /etc/supervisor/conf.d
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# HF Spaces requires port 7860
# We run Streamlit on 7860 (public) and FastAPI on 8000 (internal)
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]