# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install curl for uv installer and clean up apt
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY orchestrator/requirements.txt /app/orchestrator/requirements.txt

# Install Python dependencies using uv inside the venv
RUN uv pip install --no-cache -r /app/orchestrator/requirements.txt

# Copy application code
COPY orchestrator /app/orchestrator

EXPOSE 3002

# Environment-configurable; defaults are handled by the application
ENV PORT=3002

# Start the FastAPI app
CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "3002"]

