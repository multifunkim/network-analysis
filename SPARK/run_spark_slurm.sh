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

UPDATE_BRANCH="main"   

# Can be either:
#   1. A single .nii / .nii.gz / .func.gii file
#   2. A directory containing supported input files
input_path="/path/to/inputs"
mask_path="/path/to/mask.nii.gz"   # NIfTI only
output_base="/path/to/outputs"

suffix="_processed"

network_scales=(10 2 40)
subsample_factor=8
nb_samps=200
block_window_length=(10 1 30)
n_iter=30
pvalue=0.05
min_voxels=30

python_module="python/3.10.13"


# ============================================================
# LOGIN-NODE UPDATE AND SUBMISSION
# ============================================================

if [[ -z "${SLURM_JOB_ID:-}" ]]; then

    launcher_path="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || printf '%s' "${BASH_SOURCE[0]}")"
    launcher_dir="$(cd "$(dirname "${launcher_path}")" && pwd)"

    if ! git -C "${launcher_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "ERROR: The SPARK launcher is not inside a Git repository."
        exit 1
    fi

    repo_root="$(git -C "${launcher_dir}" rev-parse --show-toplevel)"
    launcher_relative="${launcher_path#"${repo_root}/"}"

    case "${UPDATE_BRANCH}" in
        main|development)
            ;;
        *)
            echo "ERROR: UPDATE_BRANCH must be 'main' or 'development'."
            exit 1
            ;;
    esac

    echo "Checking origin/${UPDATE_BRANCH} for SPARK updates..."

    if ! git -C "${repo_root}" diff --quiet || \
       ! git -C "${repo_root}" diff --cached --quiet; then

        echo "WARNING: Local tracked modifications were found."
        echo "Automatic update skipped to protect local work."

    elif git -C "${repo_root}" fetch --quiet origin "${UPDATE_BRANCH}"; then

        if git -C "${repo_root}" switch "${UPDATE_BRANCH}" >/dev/null 2>&1 || \
           git -C "${repo_root}" checkout "${UPDATE_BRANCH}" >/dev/null 2>&1; then

            if git -C "${repo_root}" merge --ff-only "origin/${UPDATE_BRANCH}" >/dev/null; then
                echo "SPARK ${UPDATE_BRANCH} is up to date."
            else
                echo "WARNING: Local ${UPDATE_BRANCH} cannot be fast-forwarded."
                echo "Automatic update skipped."
            fi

        else
            echo "WARNING: Could not switch to ${UPDATE_BRANCH}."
            echo "Automatic update skipped."
        fi

    else
        echo "WARNING: Could not reach origin/${UPDATE_BRANCH}."
        echo "Continuing with the currently installed SPARK version."
    fi

    spark_commit="$(git -C "${repo_root}" rev-parse HEAD)" || exit 1
    current_branch="$(git -C "${repo_root}" branch --show-current)"
    submitted_launcher="${repo_root}/${launcher_relative}"

    echo "Branch       : ${current_branch}"
    echo "SPARK commit : ${spark_commit}"
    echo "Submitting SPARK job..."

    sbatch \
        --chdir="$(dirname "${submitted_launcher}")" \
        --export="ALL,SPARK_COMMIT=${spark_commit}" \
        "${submitted_launcher}"

    exit $?
fi


# ============================================================
# AUTOMATIC PATHS
# ============================================================

SPARK_DIR="$(pwd)"
VENV_DIR="${SPARK_DIR}/.venv"
PIPELINE="${SPARK_DIR}/pipeline_steps1_6.py"

if [[ ! -f "${PIPELINE}" ]]; then
    echo "ERROR: Could not find:"
    echo "  ${PIPELINE}"
    exit 1
fi

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

mkdir -p "${output_base}"

source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${SPARK_DIR}"


# ============================================================
# JOB INFORMATION
# ============================================================

echo "============================================================"
echo "SPARK SLURM job"
echo "Branch           : ${UPDATE_BRANCH}"
echo "Commit           : ${SPARK_COMMIT:-unknown}"
echo "Host             : $(hostname)"
echo "Python           : $(which python)"
echo "Input path       : ${input_path}"
echo "Mask             : ${mask_path}"
echo "Output           : ${output_base}"
echo "CPUs per task    : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Parallel jobs    : ${max_parallel_jobs}"
echo "============================================================"


# ============================================================
# INPUT DISCOVERY
# ============================================================

fmri_files=()

if [[ -f "${input_path}" ]]; then

    case "${input_path}" in

        *.nii|*.nii.gz|*.func.gii)
            fmri_files=("${input_path}")
            ;;

        *)
            echo "ERROR: Unsupported input file:"
            echo "  ${input_path}"
            echo ""
            echo "Supported formats:"
            echo "  .nii"
            echo "  .nii.gz"
            echo "  .func.gii"

            deactivate
            exit 1
            ;;
    esac

elif [[ -d "${input_path}" ]]; then

    mapfile -t fmri_files < <(
        find "${input_path}" -maxdepth 1 -type f \
            \( \
                -name "*${suffix}.nii" \
                -o -name "*${suffix}.nii.gz" \
                -o -name "*${suffix}.func.gii" \
            \) \
            | sort
    )

else

    echo "ERROR: Input path not found:"
    echo "  ${input_path}"

    deactivate
    exit 1
fi


# ============================================================
# CHECK INPUT FILES
# ============================================================

if [[ ${#fmri_files[@]} -eq 0 ]]; then

    echo "ERROR: No supported fMRI files found."
    echo "Input path: ${input_path}"

    if [[ -d "${input_path}" ]]; then
        echo ""
        echo "Searched for:"
        echo "  *${suffix}.nii"
        echo "  *${suffix}.nii.gz"
        echo "  *${suffix}.func.gii"
    fi

    deactivate
    exit 1
fi

echo "Found ${#fmri_files[@]} fMRI file(s)."


# ============================================================
# MAIN LOOP
# ============================================================

for fmri_file_path in "${fmri_files[@]}"; do

    fmri_file="$(basename "${fmri_file_path}")"


    # --------------------------------------------------------
    # INPUT FORMAT
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

        echo "ERROR: Unsupported file format:"
        echo "  ${fmri_file}"

        continue
    fi


    # --------------------------------------------------------
    # SUBJECT LABEL
    # --------------------------------------------------------

    subject_label="$(printf '%s' "${subject_label}" \
        | sed -E "s/${suffix}$//")"

    if [[ -z "${subject_label}" ]]; then
        echo "ERROR: Could not extract subject label from:"
        echo "  ${fmri_file}"
        continue
    fi


    # --------------------------------------------------------
    # MASK CHECK
    # --------------------------------------------------------

    if [[ "${input_format}" == "nifti" ]]; then

        if [[ ! -f "${mask_path}" ]]; then
            echo "ERROR: NIfTI input requires a valid mask:"
            echo "  ${mask_path}"
            continue
        fi

    fi


    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    subject_outdir="${output_base}/${subject_label}"

    kmap_file="${subject_outdir}/KMAP_${subject_label}/k_hubness_${subject_label}.${output_extension}"

    lock_file="${subject_outdir}/.lock"

    echo ""
    echo "------------------------------------------------------------"
    echo "Subject       : ${subject_label}"
    echo "Format        : ${input_format}"
    echo "Input         : ${fmri_file_path}"
    echo "Output        : ${subject_outdir}"
    echo "Parallel jobs : ${max_parallel_jobs}"
    echo "------------------------------------------------------------"


    # --------------------------------------------------------
    # SKIP COMPLETED / LOCKED
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if [[ ${exit_code} -eq 0 ]]; then

        if [[ -f "${kmap_file}" ]]; then

            echo "Completed: ${subject_label}"

        else

            echo "WARNING: Pipeline completed successfully,"
            echo "but expected k-hubness output was not found:"
            echo "  ${kmap_file}"

        fi

    else

        echo "Failed: ${subject_label}, exit code ${exit_code}"

    fi

done


deactivate

echo ""
echo "SPARK SLURM processing finished."