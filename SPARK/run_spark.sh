#!/bin/bash
set -uo pipefail

# ============================================================
# USER CONFIGURATION
# ============================================================

input_dir="/path/to/input"
mask_path="/path/to/mask.nii.gz"
output_base="/path/to/output"

# Filename suffix before .nii or .nii.gz
suffix="_processed"

# SPARK parameters
network_scales=(10 2 40)
subsample_factor=8
nb_samps=200
block_window_length=(10 1 30)
max_parallel_jobs=8
n_iter=30
pvalue=0.35
min_voxels=30

# ============================================================
# AUTOMATIC PATHS
# ============================================================

SPARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SPARK_DIR}/.venv"
PIPELINE="${SPARK_DIR}/pipeline_steps1_6.py"

# ============================================================
# VALIDATION
# ============================================================

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "ERROR: SPARK environment was not found:"
    echo "  ${VENV_DIR}"
    echo ""
    echo "Install it first:"
    echo "  cd ${SPARK_DIR}"
    echo "  bash install_spark.sh"
    exit 1
fi

if [[ ! -d "${input_dir}" ]]; then
    echo "ERROR: Input directory not found: ${input_dir}"
    exit 1
fi

if [[ ! -f "${mask_path}" ]]; then
    echo "ERROR: Mask not found: ${mask_path}"
    exit 1
fi

mkdir -p "${output_base}"

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# The current pipeline calls the step scripts using relative paths.
cd "${SPARK_DIR}"

# ============================================================
# FIND INPUT FILES
# ============================================================

mapfile -t fmri_files < <(
    find "${input_dir}" -maxdepth 1 -type f \
        \( -name "*${suffix}.nii" -o -name "*${suffix}.nii.gz" \) \
        | sort
)

if [[ ${#fmri_files[@]} -eq 0 ]]; then
    echo "ERROR: No input fMRI files found in: ${input_dir}"
    echo "Searched for:"
    echo "  *${suffix}.nii"
    echo "  *${suffix}.nii.gz"
    deactivate
    exit 1
fi

echo "Found ${#fmri_files[@]} fMRI file(s)."

# ============================================================
# MAIN LOOP
# ============================================================

for fmri_file_path in "${fmri_files[@]}"; do

    fmri_file="$(basename "${fmri_file_path}")"

    subject_label="$(printf '%s' "${fmri_file}" | sed -E 's/\.nii(\.gz)?$//')"
    subject_label="$(printf '%s' "${subject_label}" | sed -E "s/${suffix}$//")"

    if [[ -z "${subject_label}" ]]; then
        echo "ERROR: Could not extract subject label from ${fmri_file}"
        continue
    fi

    subject_outdir="${output_base}/${subject_label}"
    kmap_file="${subject_outdir}/KMAP_${subject_label}/k_hubness_${subject_label}.nii.gz"
    lock_file="${subject_outdir}/.lock"

    echo ""
    echo "------------------------------------------------------------"
    echo "Subject : ${subject_label}"
    echo "Input   : ${fmri_file_path}"
    echo "Output  : ${subject_outdir}"
    echo "------------------------------------------------------------"

    if [[ -f "${kmap_file}" ]]; then
        echo "Skipping: k-hubness output already exists."
        continue
    fi

    if [[ -f "${lock_file}" ]]; then
        echo "Skipping: lock file exists."
        continue
    fi

    mkdir -p "${subject_outdir}"
    touch "${lock_file}"

    python "${PIPELINE}" \
        --fmri_path            "${fmri_file_path}" \
        --mask_path            "${mask_path}" \
        --output_dir           "${subject_outdir}" \
        --network_scales       "${network_scales[@]}" \
        --subsample_factor     "${subsample_factor}" \
        --nb_samps             "${nb_samps}" \
        --block_window_length  "${block_window_length[@]}" \
        --max_parallel_jobs    "${max_parallel_jobs}" \
        --n_iter               "${n_iter}" \
        --pvalue               "${pvalue}" \
        --min_voxels           "${min_voxels}" \
        --steps                all \
        --step2_extra          --coding omp --c_bits 8 --rowmean

    exit_code=$?

    rm -f "${lock_file}"

    if [[ ${exit_code} -eq 0 ]]; then
        echo "Completed: ${subject_label}"
    else
        echo "Failed: ${subject_label}, exit code ${exit_code}"
    fi
done

deactivate

echo ""
echo "SPARK processing finished."