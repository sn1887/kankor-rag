#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker/docker-compose.openwebui.yml"
DEFAULT_ENV_FILE="${ROOT_DIR}/.env.whatsapp.example"
WAIT_TIMEOUT_SECONDS="${WHATSAPP_LOCAL_WAIT_TIMEOUT_SECONDS:-120}"
WAIT_INTERVAL_SECONDS="${WHATSAPP_LOCAL_WAIT_INTERVAL_SECONDS:-2}"

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

echo "Using env file: ${ENV_FILE}"

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
  --profile whatsapp-local \
  up ${BUILD_FLAG} "${PASSTHRU_ARGS[@]}" api-whatsapp redis

if [[ "${DETACHED}" -eq 1 ]]; then
  echo "Waiting for api-whatsapp to become ready on http://127.0.0.1:8100/v1/health ..."
  deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl --silent --show-error --max-time 2 http://127.0.0.1:8100/v1/health >/dev/null 2>&1; then
      echo "api-whatsapp is ready."
      exit 0
    fi
    sleep "${WAIT_INTERVAL_SECONDS}"
  done

  echo "api-whatsapp was not ready within ${WAIT_TIMEOUT_SECONDS}s. Recent logs:" >&2
  docker compose \
    -f "${COMPOSE_FILE}" \
    --env-file "${ENV_FILE}" \
    --profile whatsapp-local \
    logs --tail=120 api-whatsapp redis >&2 || true
  exit 1
fi
