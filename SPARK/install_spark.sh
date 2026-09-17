#!/bin/bash
set -euo pipefail

# ============================================================
# SPARK INSTALLER
#
# Core SPARK installation only.
#
# Python runtime:
#   Python 3.11
#
# Usage:
#
#   bash install_spark.sh
#
# Developer installation:
#
#   bash install_spark.sh --developer
#
# Notes:
# - If an existing .venv uses an older Python version, it is
#   rebuilt automatically with Python 3.11.
# - Optional components such as the SPARK Viewer manage their
#   own dependencies separately.
# ============================================================


# ============================================================
# PATHS / MODE
# ============================================================

SPARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SPARK_DIR}/.venv"

MODE="user"

if [[ "${1:-}" == "--developer" ]]; then

    MODE="developer"

elif [[ $# -gt 0 ]]; then

    echo "Usage: bash install_spark.sh [--developer]"
    exit 1

fi


# ============================================================
# PYTHON CONFIGURATION
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python3.11}"


# ------------------------------------------------------------
# Alliance convenience:
#
# If Python 3.11 is not already available, try loading the
# standard Alliance Python 3.11 module automatically.
# ------------------------------------------------------------

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then

    if command -v module >/dev/null 2>&1; then

        echo "Python 3.11 not currently available."
        echo "Trying Alliance modules..."

        module --force purge
        module load StdEnv/2023
        module load python/3.11

    fi

fi


# ------------------------------------------------------------
# Final Python availability check
# ------------------------------------------------------------

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then

    echo "ERROR: Python 3.11 was not found."
    echo ""
    echo "Load or install Python 3.11, then rerun:"
    echo ""
    echo "  bash install_spark.sh"
    echo ""

    exit 1

fi


# ------------------------------------------------------------
# Verify Python version
# ------------------------------------------------------------

"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"ERROR: SPARK currently requires Python 3.11; "
        f"found {sys.version.split()[0]}"
    )
PY


# ============================================================
# INFORMATION
# ============================================================

echo ""
echo "============================================================"
echo "SPARK installation"
echo "============================================================"
echo "SPARK directory   : ${SPARK_DIR}"
echo "Environment       : ${VENV_DIR}"
echo "Python executable : $(command -v "${PYTHON_BIN}")"
echo "Python version    : $("${PYTHON_BIN}" --version 2>&1)"
echo "Installation mode : ${MODE}"
echo "============================================================"
echo ""


# ============================================================
# CHECK EXISTING VIRTUAL ENVIRONMENT
# ============================================================

REBUILD_ENV=false

if [[ -d "${VENV_DIR}" ]]; then

    echo "Existing SPARK virtual environment found."

    if [[ -x "${VENV_DIR}/bin/python" ]]; then

        EXISTING_PYTHON_VERSION="$(
            "${VENV_DIR}/bin/python" - <<'PY' 2>/dev/null || true
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
        )"

    else

        EXISTING_PYTHON_VERSION="unknown"

    fi

    echo "Existing environment Python: ${EXISTING_PYTHON_VERSION}"

    if [[ "${EXISTING_PYTHON_VERSION}" != "3.11" ]]; then

        echo ""
        echo "SPARK now uses Python 3.11."
        echo "The existing environment must be rebuilt."

        REBUILD_ENV=true

    fi

fi


# ============================================================
# REBUILD OLD ENVIRONMENT IF NECESSARY
# ============================================================

if [[ "${REBUILD_ENV}" == "true" ]]; then

    echo ""
    echo "Removing old SPARK environment:"
    echo "  ${VENV_DIR}"

    rm -rf "${VENV_DIR}"

fi


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

if [[ ! -d "${VENV_DIR}" ]]; then

    echo ""
    echo "Creating SPARK Python 3.11 environment..."

    "${PYTHON_BIN}" -m venv "${VENV_DIR}"

else

    echo "Using existing Python 3.11 SPARK environment."

fi


# ============================================================
# ACTIVATE ENVIRONMENT
# ============================================================

source "${VENV_DIR}/bin/activate"

echo ""
echo "Active Python:"
echo "  $(which python)"
echo "  $(python --version)"


# ============================================================
# INSTALLATION TOOLS
# ============================================================

echo ""
echo "Updating installation tools..."

python -m pip install --upgrade \
    pip \
    setuptools \
    wheel


# ============================================================
# DEPENDENCIES
# ============================================================

if [[ "${MODE}" == "developer" ]]; then

    echo ""
    echo "Installing SPARK runtime and developer dependencies..."

    python -m pip install \
        -r "${SPARK_DIR}/requirements-dev.txt"

else

    echo ""
    echo "Installing SPARK runtime dependencies..."

    python -m pip install \
        -r "${SPARK_DIR}/requirements.txt"

fi


# ============================================================
# DEPENDENCY VERIFICATION
# ============================================================

echo ""
echo "Verifying SPARK dependencies..."

python - <<'PY'
import sys

import joblib
import nibabel
import numpy
import openpyxl
import scipy
import sklearn

print("")
print("Dependency verification passed.")
print("")
print("Python        :", sys.version.split()[0])
print("NumPy         :", numpy.__version__)
print("SciPy         :", scipy.__version__)
print("scikit-learn  :", sklearn.__version__)
print("NiBabel       :", nibabel.__version__)
print("Joblib        :", joblib.__version__)
print("openpyxl      :", openpyxl.__version__)
PY


# ============================================================
# DEVELOPER SETUP
# ============================================================

if [[ "${MODE}" == "developer" ]]; then

    echo ""
    echo "Registering Jupyter kernel..."

    python -m ipykernel install --user \
        --name spark-python \
        --display-name "SPARK Python 3.11"

fi


# ============================================================
# FINISH
# ============================================================

echo ""
echo "============================================================"
echo "SPARK installation completed successfully."
echo "============================================================"
echo ""
echo "Activate the environment with:"
echo ""
echo "  source ${VENV_DIR}/bin/activate"
echo ""

if [[ "${MODE}" == "developer" ]]; then

    echo "Jupyter kernel:"
    echo ""
    echo "  SPARK Python 3.11"
    echo ""

fi