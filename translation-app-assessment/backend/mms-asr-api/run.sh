#!/usr/bin/env sh
set -e

export PORT="${PORT:-8080}"
export HF_HOME="${HF_HOME:-/tmp/models}"
export MODEL_LOCAL_DIR="${HF_HOME}"
export ENABLE_LLM_CORRECTION=1

echo "[run] HF_HOME=${HF_HOME}"
echo "[run] MODEL_LOCAL_DIR=${MODEL_LOCAL_DIR}"

exec uvicorn app:app --host 0.0.0.0 --port "${PORT}" --workers 1 --log-level info --access-log