FROM node:20-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     NEXT_TELEMETRY_DISABLED=1

RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip python3-venv build-essential git supervisor curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY packages /app/packages
COPY apps/api /app/apps/api
COPY scripts /app/scripts
COPY data /app/data
COPY docs /app/docs
COPY docker /app/docker
COPY README.md /app/README.md

RUN python3 -m pip install -U pip &&     python3 -m pip install -e '/app/packages/rag_core[serve]' -e /app/apps/api

WORKDIR /app/apps/web
COPY apps/web/package.json /app/apps/web/package.json
COPY apps/web/package-lock.json* /app/apps/web/
RUN npm install
COPY apps/web /app/apps/web
RUN npm run build

WORKDIR /app
RUN chmod +x /app/docker/start.sh
EXPOSE 7860
CMD ["bash", "/app/docker/start.sh"]
