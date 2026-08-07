#!/bin/bash
set -euo pipefail

SPARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SPARK_DIR}/.venv"
MODE="user"

if [[ "${1:-}" == "--developer" ]]; then
    MODE="developer"
elif [[ $# -gt 0 ]]; then
    echo "Usage: bash install_spark.sh [--developer]"
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 was not found."
    echo "Load or install Python 3.10 or newer, then rerun."
    exit 1
fi

python3 - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        f"ERROR: Python 3.10 or newer is required; found {sys.version.split()[0]}"
    )
PY

echo "SPARK directory : ${SPARK_DIR}"
echo "Environment     : ${VENV_DIR}"
echo "Installation mode: ${MODE}"

if [[ -d "${VENV_DIR}" ]]; then
    echo "Existing virtual environment found."
else
    echo "Creating virtual environment..."
    python3.11 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "${MODE}" == "developer" ]]; then
    echo "Installing runtime and developer dependencies..."
    python -m pip install -r "${SPARK_DIR}/requirements-dev.txt"

    echo "Registering Jupyter kernel..."
    python -m ipykernel install --user \
        --name spark-python \
        --display-name "SPARK Python"
else
    echo "Installing runtime dependencies..."
    python -m pip install -r "${SPARK_DIR}/requirements.txt"
fi

python - <<'PY'
import joblib
import nibabel
import numpy
import scipy
import sklearn

print("")
print("Dependency verification passed.")
print("NumPy        :", numpy.__version__)
print("SciPy        :", scipy.__version__)
print("scikit-learn :", sklearn.__version__)
print("NiBabel      :", nibabel.__version__)
print("Joblib       :", joblib.__version__)
PY

echo ""
echo "SPARK installation completed."
echo "Activate the environment with:"
echo "source ${VENV_DIR}/bin/activate"

if [[ "${MODE}" == "developer" ]]; then
    echo "Jupyter kernel registered as: SPARK Python"
fi
