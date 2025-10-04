
# Multi-stage build using uv for efficient Python dependency management
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Set environment variables
ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY .python-version .
COPY uv.lock .

# Create virtual environment and install dependencies
RUN uv sync --frozen --no-cache

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=appuser:appuser /app/.venv /opt/venv

# Copy application files
COPY --chown=appuser:appuser app/app.py ./
COPY --chown=appuser:appuser app/A2modelandprep.py ./
COPY --chown=appuser:appuser app/A3model.py ./
COPY --chown=appuser:appuser app/LoadA3model.py ./
COPY --chown=appuser:appuser app/model/ ./model/
COPY --chown=appuser:appuser data/ ./data/

# Copy tests for CI/CD
COPY --chown=appuser:appuser tests/conftest.py ./tests/
COPY --chown=appuser:appuser tests/test_model_staging.py ./tests/
COPY --chown=appuser:appuser tests/test_app_callbacks.py ./tests/

# Copy transition script for staging workflow
COPY --chown=appuser:appuser transition.py ./

# Copy notebooks and data (commented out for now)
# COPY --chown=appuser:appuser notebook/ ./notebook/
# COPY --chown=appuser:appuser data/ ./data/

# Switch to non-root user
USER appuser

# Add virtual environment to PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Set environment variables
ENV PORT=8050
ENV DEBUG=False
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=https://mlflow.ml.brain.cs.ait.ac.th/
ENV MLFLOW_TRACKING_USERNAME=admin
ENV MLFLOW_TRACKING_PASSWORD=password
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Expose port
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8050')" || exit 1

# Run the application
CMD ["python", "app.py"]