#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker/docker-compose.openwebui.yml"
DEFAULT_ENV_FILE="${ROOT_DIR}/docker/.env"
if [[ -f "${ROOT_DIR}/docker/.env.gemini" ]]; then
  DEFAULT_ENV_FILE="${ROOT_DIR}/docker/.env.gemini"
fi
WAIT_TIMEOUT_SECONDS="${OPENWEBUI_LOCAL_WAIT_TIMEOUT_SECONDS:-120}"
WAIT_INTERVAL_SECONDS="${OPENWEBUI_LOCAL_WAIT_INTERVAL_SECONDS:-2}"

ENV_FILE="${DEFAULT_ENV_FILE}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  ENV_FILE="$1"
  shift
fi

BUILD_FLAG=""
if [[ "${1:-}" == "--build" ]]; then
  BUILD_FLAG="--build"
  shift
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Env file not found: ${ENV_FILE}" >&2
  exit 1
fi

# Ensure selected env file values win over any previously exported shell vars.
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

# Force timing logs on for local eval/debug runs launched via this helper.
export RAG_TIMING_DEBUG=true

echo "Using env file: ${ENV_FILE}"
echo "Effective RAG runtime config:"
echo "  RAG_LLM_BACKEND=${RAG_LLM_BACKEND:-openai}"
echo "  RAG_EMBEDDING_BACKEND=${RAG_EMBEDDING_BACKEND:-hash}"
echo "  RAG_CONTEXT_MODE=${RAG_CONTEXT_MODE:-text}"
echo "  RAG_TIMING_DEBUG=${RAG_TIMING_DEBUG}"
echo "  RAG_INDEX_PATH=${RAG_INDEX_PATH:-/app/data/sample_index/index.faiss}"
echo "  RAG_DOCSTORE_PATH=${RAG_DOCSTORE_PATH:-/app/data/sample_index/metadata.jsonl}"
echo "  RAG_TOC_MANIFEST_PATH=${RAG_TOC_MANIFEST_PATH:-<auto-or-empty>}"

PASSTHRU_ARGS=("$@")
DETACHED=0
for arg in "${PASSTHRU_ARGS[@]}"; do
  if [[ "${arg}" == "-d" || "${arg}" == "--detach" ]]; then
    DETACHED=1
    break
  fi
done

docker compose \
  -f "${COMPOSE_FILE}" \
  --env-file "${ENV_FILE}" \
  up --force-recreate ${BUILD_FLAG} "${PASSTHRU_ARGS[@]}" api openwebui

if [[ "${DETACHED}" -eq 1 ]]; then
  echo "Waiting for api to become ready on http://127.0.0.1:8000/v1/health ..."
  deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl --silent --show-error --max-time 2 http://127.0.0.1:8000/v1/health >/dev/null 2>&1; then
      echo "api is ready."
      exit 0
    fi
    sleep "${WAIT_INTERVAL_SECONDS}"
  done

  echo "api was not ready within ${WAIT_TIMEOUT_SECONDS}s. Recent logs:" >&2
  docker compose \
    -f "${COMPOSE_FILE}" \
    --env-file "${ENV_FILE}" \
    logs --tail=120 api openwebui >&2 || true
  exit 1
fi
