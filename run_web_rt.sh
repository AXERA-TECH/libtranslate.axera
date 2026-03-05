#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TRANS_MODEL_DIR=${TRANS_MODEL_DIR:-/home/axera/ax-llm/build_650/axllm-models/HY-MT1.5-1.8B_GPTQ_INT4}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8001}

CERT_DIR="${ROOT_DIR}/web_rt/certs"
CERT_FILE="${CERT_DIR}/server.crt"
KEY_FILE="${CERT_DIR}/server.key"
mkdir -p "${CERT_DIR}"

LOCAL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
SAN="DNS:localhost,IP:127.0.0.1"
if [ -n "${LOCAL_IP}" ]; then
  SAN="${SAN},IP:${LOCAL_IP}"
fi

NEED_REGEN_CERT=0
if [ ! -f "${CERT_FILE}" ] || [ ! -f "${KEY_FILE}" ]; then
  NEED_REGEN_CERT=1
else
  CERT_SAN="$(openssl x509 -in "${CERT_FILE}" -noout -ext subjectAltName 2>/dev/null || true)"
  if ! echo "${CERT_SAN}" | grep -q "DNS:localhost"; then
    NEED_REGEN_CERT=1
  elif [ -n "${LOCAL_IP}" ] && ! echo "${CERT_SAN}" | grep -q "IP Address:${LOCAL_IP}"; then
    NEED_REGEN_CERT=1
  fi
fi

if [ "${NEED_REGEN_CERT}" = "1" ]; then
  echo "Generating self-signed cert..."
  openssl req -x509 -newkey rsa:2048 -sha256 -nodes \
    -keyout "${KEY_FILE}" -out "${CERT_FILE}" -days 365 \
    -subj "/CN=localhost" \
    -addext "subjectAltName=${SAN}"
fi

echo "TRANS_MODEL_DIR = ${TRANS_MODEL_DIR}"
echo "HOST            = ${HOST}"
echo "PORT            = ${PORT}"
echo "SSL_CERT        = ${CERT_FILE}"
echo "SSL_KEY         = ${KEY_FILE}"

SSL_CERT="${CERT_FILE}" SSL_KEY="${KEY_FILE}" python3 "${ROOT_DIR}/web_rt/server.py"
