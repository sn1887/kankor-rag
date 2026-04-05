FROM node:20-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     NEXT_TELEMETRY_DISABLED=1

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-dev python3-pip python3-venv build-essential git supervisor curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY packages /app/packages
COPY apps/api /app/apps/api
COPY scripts /app/scripts
RUN mkdir -p /app/data
COPY data/sample_corpus /app/data/sample_corpus
COPY docs /app/docs
COPY docker /app/docker
COPY README.md /app/README.md

RUN python3 -m venv /app/.venv \
  && /app/.venv/bin/python -m pip install -U pip setuptools wheel \
  && /app/.venv/bin/python -m pip install -e '/app/packages/rag_core[serve]' -e /app/apps/api \
  && /app/.venv/bin/python -m pip install 'google-genai>=1.0.0' 'sentencepiece>=0.2.0' 'FlagEmbedding>=1.3.4' 'transformers>=4.49.0,<5' 'torch>=2.6.0,<2.8'
ENV PATH="/app/.venv/bin:${PATH}"

WORKDIR /app/apps/web
COPY apps/web/package.json /app/apps/web/package.json
COPY apps/web/package-lock.json* /app/apps/web/
RUN npm install
COPY apps/web /app/apps/web
RUN npm run build

WORKDIR /app
RUN chmod +x /app/docker/start.sh
EXPOSE 7860 8000
CMD ["bash", "/app/docker/start.sh"]
