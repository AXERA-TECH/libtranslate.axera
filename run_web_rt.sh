#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AX_MEETING_ROOT=${AX_MEETING_ROOT:-/home/axera/AXERA-TECH/3D-Speaker-MT.Axera}
export ASR_MODEL_DIR=${ASR_MODEL_DIR:-${AX_MEETING_ROOT}/ax_meeting/ax_model}
export TRANS_MODEL_DIR=${TRANS_MODEL_DIR:-/home/axera/ax-llm/build_650/axllm-models/HY-MT1.5-1.8B_GPTQ_INT4}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8001}

CERT_DIR="${ROOT_DIR}/web_rt/certs"
CERT_FILE="${CERT_DIR}/server.crt"
KEY_FILE="${CERT_DIR}/server.key"
mkdir -p "${CERT_DIR}"

if [ ! -f "${CERT_FILE}" ] || [ ! -f "${KEY_FILE}" ]; then
  echo "Generating self-signed cert..."
  openssl req -x509 -newkey rsa:2048 -sha256 -nodes \
    -keyout "${KEY_FILE}" -out "${CERT_FILE}" -days 365 \
    -subj "/CN=localhost"
fi

echo "AX_MEETING_ROOT = ${AX_MEETING_ROOT}"
echo "ASR_MODEL_DIR   = ${ASR_MODEL_DIR}"
echo "TRANS_MODEL_DIR = ${TRANS_MODEL_DIR}"
echo "HOST            = ${HOST}"
echo "PORT            = ${PORT}"
echo "SSL_CERT        = ${CERT_FILE}"
echo "SSL_KEY         = ${KEY_FILE}"

SSL_CERT="${CERT_FILE}" SSL_KEY="${KEY_FILE}" python3 "${ROOT_DIR}/web_rt/server.py"
