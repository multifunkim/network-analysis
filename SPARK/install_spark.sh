#!/bin/bash
set -euo pipefail

SPARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SPARK_DIR}/.venv"

if command -v module >/dev/null 2>&1; then
    module --force purge
    module load StdEnv/2023 python/3.10.13
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${SPARK_DIR}/requirements.txt"

python -m ipykernel install --user \
    --name spark-python \
    --display-name "SPARK Python"

echo "SPARK environment installed successfully."
echo "Activate with:"
echo "source ${VENV_DIR}/bin/activate"
