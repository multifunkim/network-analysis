#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=spark_batch
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL

set -uo pipefail

# ============================================================
# USER CONFIGURATION
# ============================================================

input_dir="/path/to/input"
mask_path="/path/to/mask.nii.gz"    # Required only for NIfTI input and Ignored for GIFTI input.
output_base="/path/to/output"

# Filename suffix before:
#   .nii
#   .nii.gz
#   .func.gii
#
# Example:
#   sub-01_processed.nii.gz
#   sub-01_processed.func.gii
suffix="_processed"

# SPARK parameters
network_scales=(10 2 40)
subsample_factor=8
nb_samps=200
block_window_length=(10 1 30)
n_iter=30
pvalue=0.05
min_voxels=30

# Alliance module configuration
python_module="python/3.10.13"

# ============================================================
# AUTOMATIC PATHS
# ============================================================

SPARK_DIR="${SLURM_SUBMIT_DIR}"
VENV_DIR="${SPARK_DIR}/.venv"
PIPELINE="${SPARK_DIR}/pipeline_steps1_6.py"

if [[ ! -f "${PIPELINE}" ]]; then
    echo "ERROR: Submit this job from the SPARK directory."
    echo "Current submission directory: ${SPARK_DIR}"
    exit 1
fi

# Use the CPUs assigned by SLURM
max_parallel_jobs="${SLURM_CPUS_PER_TASK:-1}"

# ============================================================
# ENVIRONMENT
# ============================================================

if command -v module >/dev/null 2>&1; then
    module --force purge
    module load StdEnv/2023
    module load "${python_module}"
fi

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

mkdir -p "${output_base}"

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Required because the current pipeline uses relative step-script paths.
cd "${SPARK_DIR}"

echo "============================================================"
echo "SPARK SLURM job"
echo "============================================================"
echo "Host            : $(hostname)"
echo "Python          : $(which python)"
echo "Input directory : ${input_dir}"
echo "Mask            : ${mask_path} (NIfTI only)"
echo "Output          : ${output_base}"
echo "Parallel jobs   : ${max_parallel_jobs}"
echo "============================================================"

# ============================================================
# FIND INPUT FILES
# ============================================================

mapfile -t fmri_files < <(
    find "${input_dir}" -maxdepth 1 -type f \
        \( \
            -name "*${suffix}.nii" \
            -o -name "*${suffix}.nii.gz" \
            -o -name "*${suffix}.func.gii" \
        \) \
        | sort
)

if [[ ${#fmri_files[@]} -eq 0 ]]; then
    echo "ERROR: No supported fMRI files found in: ${input_dir}"
    echo "Searched for:"
    echo "  *${suffix}.nii"
    echo "  *${suffix}.nii.gz"
    echo "  *${suffix}.func.gii"
    deactivate
    exit 1
fi

echo "Found ${#fmri_files[@]} fMRI file(s)."

# ============================================================
# MAIN LOOP
# ============================================================
pvalue=0.2
for fmri_file_path in "${fmri_files[@]}"; do

    fmri_file="$(basename "${fmri_file_path}")"

    # --------------------------------------------------------
    # DETECT INPUT FORMAT
    # --------------------------------------------------------

    if [[ "${fmri_file}" == *.func.gii ]]; then

        input_format="gifti"
        output_extension="func.gii"

        subject_label="$(printf '%s' "${fmri_file}" \
            | sed -E 's/\.func\.gii$//')"

    elif [[ "${fmri_file}" == *.nii.gz ]]; then

        input_format="nifti"
        output_extension="nii.gz"

        subject_label="$(printf '%s' "${fmri_file}" \
            | sed -E 's/\.nii\.gz$//')"

    elif [[ "${fmri_file}" == *.nii ]]; then

        input_format="nifti"
        output_extension="nii.gz"

        subject_label="$(printf '%s' "${fmri_file}" \
            | sed -E 's/\.nii$//')"

    else

        echo "ERROR: Unsupported file format: ${fmri_file}"
        continue

    fi

    # Remove configured suffix from subject label
    subject_label="$(printf '%s' "${subject_label}" \
        | sed -E "s/${suffix}$//")"

    if [[ -z "${subject_label}" ]]; then
        echo "ERROR: Could not extract subject label from ${fmri_file}"
        continue
    fi

    # --------------------------------------------------------
    # NIFTI MASK CHECK
    # --------------------------------------------------------

    if [[ "${input_format}" == "nifti" && ! -f "${mask_path}" ]]; then
        echo "ERROR: NIfTI input requires a valid mask:"
        echo "  ${mask_path}"
        continue
    fi

    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    subject_outdir="${output_base}/${subject_label}"

    kmap_file="${subject_outdir}/KMAP_${subject_label}/k_hubness_${subject_label}.${output_extension}"

    lock_file="${subject_outdir}/.lock"

    echo ""
    echo "------------------------------------------------------------"
    echo "Subject : ${subject_label}"
    echo "Format  : ${input_format}"
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

    # --------------------------------------------------------
    # RUN SPARK
    # --------------------------------------------------------

    if [[ "${input_format}" == "nifti" ]]; then

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

    else

        python "${PIPELINE}" \
            --fmri_path            "${fmri_file_path}" \
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

    fi

    exit_code=$?

    rm -f "${lock_file}"

    if [[ ${exit_code} -eq 0 ]]; then

        if [[ -f "${kmap_file}" ]]; then
            echo "Completed: ${subject_label}"
        else
            echo "WARNING: Pipeline completed but expected output was not found:"
            echo "  ${kmap_file}"
        fi

    else
        echo "Failed: ${subject_label}, exit code ${exit_code}"
    fi

done

deactivate

echo ""
echo "SPARK SLURM processing finished."