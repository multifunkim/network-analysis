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

UPDATE_BRANCH="${UPDATE_BRANCH:-main}"   

# Can be either:
#   1. A single .nii / .nii.gz / .func.gii file
#   2. A directory containing supported input files
input_path="/path/to/inputs"
mask_path="/path/to/mask.nii.gz"   # NIfTI only
output_base="/path/to/outputs"

suffix="_processed"

network_scales=(10 2 40)
cortex_scales=(10 2 20)
noncortex_scales=(2 2 10)
cifti_region='["cortex","subcortex"]'
joint_l=5
bootstrap_seed=0

# ============================================================
# PIPELINE EXECUTION OPTIONS
# ============================================================

# SPARK execution
# Options:
#   "all" : Steps 1-6 (skip completed steps)
#   "1"   : Step 1 only
#   "3,4" : Steps 3 and 4
#   "2-5" : Steps 2 through 5
# RUN_DYNO=true requires "all".
steps_to_run="all"

# dynoSPARK execution
# Options:
#   false : Standard SPARK only
#   true  : SPARK + dynoSPARK Step 8
# Completed SPARK steps are skipped.
# Existing dynoSPARK outputs are preserved.
RUN_DYNO=false

# Temporal reconstruction L1 regularization
# Examples: 0.00, 0.01, 0.05, 0.10
# 0.05 is provisional, not scientifically optimized.
# Used only when RUN_DYNO=true.
DYNO_LAMBDA_RATIO=0.05

subsample_factor=8
nb_samps=200
block_window_length=(10 1 30)
n_iter=30
pvalue=0.05
min_voxels=30

python_module="python/3.11"


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

SPARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

    if [[ "${fmri_file}" == *.dtseries.nii ]]; then

        input_format="cifti"
        output_extension="dscalar.nii"
        subject_label="${fmri_file%.dtseries.nii}"

    elif [[ "${fmri_file}" == *.func.gii ]]; then

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

    # Expected SPARK analysis units.
    if [[ "${input_format}" == "cifti" ]]; then
        case "${cifti_region}" in
            whole|'["whole"]')
                analysis_dirs=("${subject_outdir}")
                ;;
            both|'["cortex","subcortex"]'|'["subcortex","cortex"]')
                analysis_dirs=(
                    "${subject_outdir}/cortex"
                    "${subject_outdir}/noncortex"
                )
                ;;
            cortex|'["cortex"]')
                analysis_dirs=("${subject_outdir}/cortex")
                ;;
            subcortex|'["subcortex"]')
                analysis_dirs=("${subject_outdir}/noncortex")
                ;;
            *)
                echo "ERROR: Unsupported CIFTI selection: ${cifti_region}"
                exit 1
                ;;
        esac
    else
        analysis_dirs=("${subject_outdir}")
    fi

    kmap_extension="${output_extension}"
    if [[ "${input_format}" == "cifti" ]]; then
        kmap_extension="dscalar.nii"
    fi

    spark_complete=true

    [[ -s "${subject_outdir}/tseries.mat" ]] || spark_complete=false

    for analysis_dir in "${analysis_dirs[@]}"; do

        kmap_file="${analysis_dir}/KMAP_${subject_label}/k_hubness_${subject_label}.${kmap_extension}"

        if [[ ! -s "${analysis_dir}/clusters.mat" ]] ||
           [[ ! -s "${analysis_dir}/dict/boot_000_dict.mat" ]] ||
           [[ ! -s "${kmap_file}" ]]; then
            spark_complete=false
        fi

    done

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

    if [[ "${steps_to_run}" == "all" &&
          "${spark_complete}" == "true" &&
          "${RUN_DYNO}" == "false" ]]; then
        echo "Skipping: SPARK Steps 1–6 already complete."
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

    if [[ "${steps_to_run}" == "all" &&
          "${spark_complete}" == "true" ]]; then

        echo "SPARK already complete; skipping Steps 1-6."
        exit_code=0

    else

    if [[ "${input_format}" == "nifti" ]]; then

        python "${PIPELINE}" \
            --fmri_path            "${fmri_file_path}" \
            --mask_path            "${mask_path}" \
            --output_dir           "${subject_outdir}" \
            --network_scales       "${network_scales[@]}" \
            --cifti_region "${cifti_region}" \
            --joint_l "${joint_l}" \
            --bootstrap_seed "${bootstrap_seed}" \
            --cortex_scales        "${cortex_scales[@]}" \
            --noncortex_scales     "${noncortex_scales[@]}" \
            --subsample_factor     "${subsample_factor}" \
            --nb_samps             "${nb_samps}" \
            --block_window_length  "${block_window_length[@]}" \
            --max_parallel_jobs    "${max_parallel_jobs}" \
            --n_iter               "${n_iter}" \
            --pvalue               "${pvalue}" \
            --min_voxels           "${min_voxels}" \
            --steps                "${steps_to_run}" \
            --step2_extra          --coding omp --c_bits 8 --rowmean

    else

        python "${PIPELINE}" \
            --fmri_path            "${fmri_file_path}" \
            --output_dir           "${subject_outdir}" \
            --network_scales       "${network_scales[@]}" \
            --cifti_region "${cifti_region}" \
            --joint_l "${joint_l}" \
            --bootstrap_seed "${bootstrap_seed}" \
            --cortex_scales        "${cortex_scales[@]}" \
            --noncortex_scales     "${noncortex_scales[@]}" \
            --subsample_factor     "${subsample_factor}" \
            --nb_samps             "${nb_samps}" \
            --block_window_length  "${block_window_length[@]}" \
            --max_parallel_jobs    "${max_parallel_jobs}" \
            --n_iter               "${n_iter}" \
            --pvalue               "${pvalue}" \
            --min_voxels           "${min_voxels}" \
            --steps                "${steps_to_run}" \
            --step2_extra          --coding omp --c_bits 8 --rowmean

    fi

    exit_code=$?
    fi


    # --------------------------------------------------------
    # RESULT
    # ========================================================

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "ERROR: SPARK failed for ${subject_label}."
        rm -f "${lock_file}"
        continue
    fi

    if [[ "${steps_to_run}" != "all" ]]; then
        echo "Selected SPARK steps finished."
        rm -f "${lock_file}"
        continue
    fi

    # Recheck outputs after the SPARK run.
    spark_complete=true

    [[ -s "${subject_outdir}/tseries.mat" ]] ||
        spark_complete=false

    for analysis_dir in "${analysis_dirs[@]}"; do

        expected_kmap="${analysis_dir}/KMAP_${subject_label}/k_hubness_${subject_label}.${kmap_extension}"

        if [[ ! -s "${analysis_dir}/clusters.mat" ]] ||
           [[ ! -s "${analysis_dir}/dict/boot_000_dict.mat" ]] ||
           [[ ! -s "${expected_kmap}" ]]; then
            spark_complete=false
        fi
    done

    if [[ "${spark_complete}" != "true" ]]; then
        echo "ERROR: SPARK outputs are incomplete."
        rm -f "${lock_file}"
        continue
    fi

    echo "SPARK Steps 1-6 complete: ${subject_label}"

    # --------------------------------------------------------
    # Phase 3 -- dynoSPARK
    # --------------------------------------------------------

    if [[ "${RUN_DYNO}" == "true" ]]; then

        STEP8="${SPARK_DIR}/brick/step8_dynospark.py"
        dyno_dir="${subject_outdir}/dyno"

        if [[ ! -f "${STEP8}" ]]; then
            echo "ERROR: Step 8 script not found: ${STEP8}"
            rm -f "${lock_file}"
            continue
        fi

        unit_names=()

        for analysis_dir in "${analysis_dirs[@]}"; do
            if [[ "${analysis_dir}" == "${subject_outdir}" ]]; then
                if [[ "${input_format}" == "cifti" ]]; then
                    unit_names+=("whole")
                else
                    unit_names+=("standard")
                fi
            else
                unit_names+=("$(basename "${analysis_dir}")")
            fi
        done

        # Phase 2: signed spatial atoms.
        signed_count=0
        for unit in "${unit_names[@]}"; do
            [[ -e "${dyno_dir}/signed_atoms_${unit}.mat" ]] &&
                signed_count=$((signed_count + 1))
        done

        if [[ "${signed_count}" -eq 0 ]]; then

            echo "Running dynoSPARK signed atom reconstruction."

            if ! python "${STEP8}" \\
                --subject_dir "${subject_outdir}" \\
                --signed-atoms; then
                echo "ERROR: Signed atom reconstruction failed."
                rm -f "${lock_file}"
                continue
            fi

        elif [[ "${signed_count}" -eq "${#unit_names[@]}" ]]; then
            echo "Signed atoms already exist; skipping."
        else
            echo "ERROR: Partial signed-atom outputs detected."
            echo "Inspect dyno/ before rerunning."
            rm -f "${lock_file}"
            continue
        fi

        signed_ok=true
        for unit in "${unit_names[@]}"; do
            [[ -s "${dyno_dir}/signed_atoms_${unit}.mat" ]] ||
                signed_ok=false
        done

        if [[ "${signed_ok}" != "true" ]]; then
            echo "ERROR: Signed-atom outputs are incomplete."
            rm -f "${lock_file}"
            continue
        fi

        # Phase 3: kappa and final temporal expression.
        recon_files=(
            "${dyno_dir}/dyno_timeseries.mat"
            "${dyno_dir}/reconstruction_metrics.json"
            "${dyno_dir}/dyno_metadata.json"
        )

        for unit in "${unit_names[@]}"; do
            recon_files+=("${dyno_dir}/dyno_${unit}.mat")
        done

        recon_count=0
        for file in "${recon_files[@]}"; do
            [[ -e "${file}" ]] &&
                recon_count=$((recon_count + 1))
        done

        if [[ "${recon_count}" -eq 0 ]]; then

            echo "Running dynoSPARK temporal reconstruction."

            if ! python "${STEP8}" \\
                --subject_dir "${subject_outdir}" \\
                --reconstruct \\
                --lambda-ratio 0.05; then
                echo "ERROR: Temporal reconstruction failed."
                rm -f "${lock_file}"
                continue
            fi

        elif [[ "${recon_count}" -eq "${#recon_files[@]}" ]]; then
            echo "dynoSPARK outputs already exist; skipping."
        else
            echo "ERROR: Partial dynoSPARK outputs detected."
            echo "Inspect dyno/ before rerunning."
            rm -f "${lock_file}"
            continue
        fi

        recon_ok=true
        for file in "${recon_files[@]}"; do
            [[ -s "${file}" ]] || recon_ok=false
        done

        if [[ "${recon_ok}" != "true" ]]; then
            echo "ERROR: dynoSPARK outputs are incomplete."
            rm -f "${lock_file}"
            continue
        fi

        echo "dynoSPARK completed: ${subject_label}"
    fi

    rm -f "${lock_file}"

done


deactivate

echo ""
echo "SPARK SLURM processing finished."