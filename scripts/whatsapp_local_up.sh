#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker/docker-compose.openwebui.yml"
DEFAULT_ENV_FILE="${ROOT_DIR}/.env.whatsapp.example"

ENV_FILE="${DEFAULT_ENV_FILE}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  ENV_FILE="$1"
  shift
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Env file not found: ${ENV_FILE}" >&2
  exit 1
fi

echo "Using env file: ${ENV_FILE}"

docker compose \
  -f "${COMPOSE_FILE}" \
  --env-file "${ENV_FILE}" \
  --profile whatsapp-local \
  up --build "$@" api-whatsapp redis
