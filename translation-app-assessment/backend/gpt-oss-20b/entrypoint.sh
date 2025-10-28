#!/usr/bin/env bash
set -euo pipefail

: "${PORT:=8080}"
: "${API_KEY:=secretemploi}"
: "${N_CTX:=8192}"
: "${N_PARALLEL:=4}"
: "${N_GPU_LAYERS:=999}"
: "${MODEL_DIR:=/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/model.gguf}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "[ERROR] Modèle introuvable : ${MODEL_PATH}"
  ls -lah "${MODEL_DIR}" || true
  exit 2
fi

SERVER_CMD=""
if command -v llama-server >/dev/null 2>&1; then
  SERVER_CMD="llama-server"
elif command -v llama >/dev/null 2>&1; then
  SERVER_CMD="llama --server"
else
  echo "[ERROR] Binaire serveur introuvable"; echo "PATH=$PATH"; exit 127
fi

echo "[INFO] Using: ${SERVER_CMD}"
exec ${SERVER_CMD} --host 0.0.0.0 --port "${PORT}" \
  -m "${MODEL_PATH}" -c "${N_CTX}" -np "${N_PARALLEL}" -ngl "${N_GPU_LAYERS}" \
  --api-key "${API_KEY}"