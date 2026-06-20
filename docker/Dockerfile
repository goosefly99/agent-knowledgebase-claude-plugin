# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    AGENT_KB_SAVES_DIR=/data/saves

# Minimal system deps. sentence-transformers needs a C++ runtime via torch,
# but the slim image already ships libstdc++; no extra apt installs needed
# for the wheel-only deps we ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies. requirements.lock is generated from
# pyproject + the embed-local-st extra so sentence-transformers ships
# in the image (the openai provider just uses httpx; ollama is a separate
# sidecar service).
COPY requirements.lock /app/requirements.lock
RUN pip install --timeout 120 -r requirements.lock

# Copy source.
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/

# Editable install so `python -m agent_knowledgebase.server` resolves the
# package from /app/src.
RUN pip install --no-deps -e /app

# Persistent state lives here; mounted as a volume by docker-compose.
RUN mkdir -p /data/saves

# Entry script keeps the container alive so `docker exec` works.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
