#!/usr/bin/env bash
set -euo pipefail

export BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export PORT="${PORT:-7860}"
export HOST="${HOST:-0.0.0.0}"
export RAG_ARTIFACT_DIR="${RAG_ARTIFACT_DIR:-/tmp/kankor-artifacts}"

if [[ "${RAG_DOWNLOAD_ON_BOOT:-false}" == "true" ]] && [[ -n "${RAG_DATASET_REPO_ID:-}" ]]; then
  mkdir -p "${RAG_ARTIFACT_DIR}"
  python3 - <<'BOOTPY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
repo_id = os.environ.get('RAG_DATASET_REPO_ID', '').strip()
subfolder = os.environ.get('RAG_DATASET_SUBFOLDER', '').strip() or None
local_dir = Path(os.environ.get('RAG_ARTIFACT_DIR', '/tmp/kankor-artifacts'))
token = os.environ.get('HF_TOKEN') or None
if repo_id:
    allow_patterns = [f"{subfolder}/*"] if subfolder else None
    snapshot_download(repo_id=repo_id, repo_type='dataset', local_dir=str(local_dir), token=token, allow_patterns=allow_patterns)
BOOTPY

  if [[ -n "${RAG_DATASET_SUBFOLDER:-}" ]]; then
    export RAG_INDEX_PATH="${RAG_INDEX_PATH:-${RAG_ARTIFACT_DIR}/${RAG_DATASET_SUBFOLDER}/index.faiss}"
    export RAG_DOCSTORE_PATH="${RAG_DOCSTORE_PATH:-${RAG_ARTIFACT_DIR}/${RAG_DATASET_SUBFOLDER}/metadata.json}"
  else
    export RAG_INDEX_PATH="${RAG_INDEX_PATH:-${RAG_ARTIFACT_DIR}/index.faiss}"
    export RAG_DOCSTORE_PATH="${RAG_DOCSTORE_PATH:-${RAG_ARTIFACT_DIR}/metadata.json}"
  fi
fi

exec /usr/bin/supervisord -c /app/docker/supervisord.conf
