#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=spark_qc
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

set -uo pipefail

# ============================================================
# USER CONFIGURATION
# ============================================================

# Can point to:
#
#   1. One atom_qc_*.xlsx workbook
#   2. One subject/KMAP directory
#   3. The root SPARK results directory
#
# In root-directory mode all atom_qc_*.xlsx files are found
# recursively.
qc_path="/path/to/SPARK/results"

# false = keep existing clean k-hubness outputs
# true  = replace them
overwrite=false

# Alliance module configuration
python_module="python/3.10.13"

# ============================================================
# AUTOMATIC PATHS
# ============================================================

SPARK_DIR="${SLURM_SUBMIT_DIR}"
VENV_DIR="${SPARK_DIR}/.venv"

STEP7="${SPARK_DIR}/brick/step7_atom_qc.py"

if [[ ! -f "${STEP7}" ]]; then

    echo "ERROR: Submit this job from the SPARK directory."
    echo "Expected:"
    echo "  ${STEP7}"

    exit 1
fi

if [[ ! -e "${qc_path}" ]]; then

    echo "ERROR: QC path not found:"
    echo "  ${qc_path}"

    exit 1
fi

# ============================================================
# ENVIRONMENT
# ============================================================

if command -v module >/dev/null 2>&1; then

    module --force purge
    module load StdEnv/2023
    module load "${python_module}"

fi

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then

    echo "ERROR: SPARK environment not found:"
    echo "  ${VENV_DIR}"
    echo ""
    echo "Run:"
    echo "  bash install_spark.sh"

    exit 1
fi

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SPARK_DIR}"

# ============================================================
# JOB INFO
# ============================================================

echo "============================================================"
echo "SPARK atom QC"
echo "============================================================"
echo "Host      : $(hostname)"
echo "Python    : $(which python)"
echo "QC path   : ${qc_path}"
echo "Overwrite : ${overwrite}"
echo "============================================================"

# ============================================================
# BUILD COMMAND
# ============================================================

qc_cmd=(
    python
    "${STEP7}"
    --qc_path
    "${qc_path}"
)

if [[ "${overwrite}" == "true" ]]; then
    qc_cmd+=(
        --overwrite
    )
fi

# ============================================================
# RUN
# ============================================================

"${qc_cmd[@]}"

exit_code=$?

deactivate

if [[ ${exit_code} -eq 0 ]]; then

    echo ""
    echo "SPARK atom QC completed."

else

    echo ""
    echo "SPARK atom QC failed with exit code ${exit_code}."

fi

exit "${exit_code}"
