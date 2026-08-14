#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline_steps1_6.py
====================

SPARK pipeline Steps 1–6 with format-aware input handling.

Supported inputs
----------------
NIfTI:
    .nii
    .nii.gz

GIFTI:
    .func.gii

The input format is detected automatically.

NIfTI:
    --mask_path is required

GIFTI:
    --mask_path is not required

Steps 2–5 remain format-independent because they operate on
SPARK's internal T × V representation.
"""

import os
import re
import glob
import shlex
import logging
import argparse
import subprocess

from spatial_io import detect_format


# ============================================================
# LOGGING
# ============================================================

def setup_logger():

    logging.basicConfig(
        level=logging.INFO,
        format="[pipeline] %(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


# ============================================================
# STEP PARSING
# ============================================================

def parse_steps(arg):

    if arg.lower() in ("all", ""):
        return set(range(1, 7))

    steps = set()

    for part in arg.split(","):

        m = re.match(
            r"(\d+)(?:-(\d+))?$",
            part.strip()
        )

        if not m:
            raise ValueError(
                f"Invalid --steps: {part}"
            )

        a = int(m.group(1))
        b = int(
            m.group(2) or m.group(1)
        )

        steps.update(
            range(a, b + 1)
        )

    return steps


# ============================================================
# STEP EXECUTION
# ============================================================

def run_step(
    cmd,
    marker,
    do_run,
    step
):

    if not do_run:

        logging.info(
            "SKIP step %d",
            step
        )

        return

    if os.path.exists(marker):

        logging.info(
            "✔ step %d already completed (%s)",
            step,
            os.path.basename(marker)
        )

        return

    logging.info(
        "→ step %d: %s",
        step,
        " ".join(
            shlex.quote(c)
            for c in cmd
        )
    )

    subprocess.check_call(cmd)


# ============================================================
# MAIN
# ============================================================

def main():

    setup_logger()

    # --------------------------------------------------------
    # ARGUMENTS
    # --------------------------------------------------------

    p = argparse.ArgumentParser(
        description=(
            "SPARK pipeline Steps 1–6 "
            "for NIfTI or GIFTI input"
        )
    )

    p.add_argument(
        "--fmri_path",
        required=True,
        help="Input .nii/.nii.gz or .func.gii file"
    )

    p.add_argument(
        "--mask_path",
        default=None,
        help=(
            "NIfTI mask. Required for NIfTI input; "
            "not required for GIFTI."
        )
    )

    p.add_argument(
        "--output_dir",
        required=True
    )

    p.add_argument(
        "--network_scales",
        nargs=3,
        type=int,
        required=True,
        metavar=(
            "KMIN",
            "KSTEP",
            "KMAX"
        )
    )

    p.add_argument(
        "--l-max",
        type=int,
        default=None,
        help=(
            "Cap L grid in Step 2 "
            "(default None → K/2)"
        )
    )

    p.add_argument(
        "--subsample_factor",
        type=int,
        default=1
    )

    p.add_argument(
        "--nb_samps",
        type=int,
        default=30
    )

    p.add_argument(
        "--block_window_length",
        nargs=3,
        type=int,
        default=[30, 1, 30]
    )

    p.add_argument(
        "--max_parallel_jobs",
        type=int,
        default=8
    )

    p.add_argument(
        "--n_iter",
        type=int,
        default=8
    )

    p.add_argument(
        "--pvalue",
        type=float,
        default=0.05
    )

    p.add_argument(
        "--min_voxels",
        type=int,
        default=30,
        help=(
            "Minimum number of spatial elements retained "
            "per atom. Kept for backward compatibility; "
            "for GIFTI this corresponds to vertices."
        )
    )

    p.add_argument(
        "--steps",
        type=str,
        default="all"
    )

    # Additional flags forwarded to Step 2
    p.add_argument(
        "--step2_extra",
        nargs=argparse.REMAINDER,
        help=(
            "Additional arguments forwarded to "
            "step2_estimate_scale.py"
        )
    )

    args = p.parse_args()


    # ========================================================
    # DETECT INPUT FORMAT
    # ========================================================

    input_format = detect_format(
        args.fmri_path
    )

    logging.info(
        "Input file: %s",
        args.fmri_path
    )

    logging.info(
        "Detected format: %s",
        input_format.upper()
    )


    # ========================================================
    # VALIDATE FORMAT-SPECIFIC REQUIREMENTS
    # ========================================================

    if input_format == "nifti":

        if args.mask_path is None:

            raise ValueError(
                "NIfTI input requires --mask_path."
            )

        logging.info(
            "NIfTI mask: %s",
            args.mask_path
        )

    elif input_format == "gifti":

        if args.mask_path is not None:

            logging.warning(
                "--mask_path was provided for GIFTI input "
                "but will not be used."
            )

        logging.info(
            "GIFTI surface input: no volumetric mask required"
        )


    # ========================================================
    # STEPS TO RUN
    # ========================================================

    to_run = parse_steps(
        args.steps
    )


    # ========================================================
    # WORKING DIRECTORY
    # ========================================================

    wd = args.output_dir

    os.makedirs(
        wd,
        exist_ok=True
    )


    # ========================================================
    # OUTPUT FILENAMES
    # ========================================================

    mat1 = os.path.join(
        wd,
        "tseries.mat"
    )

    mat2 = os.path.join(
        wd,
        "scale.mat"
    )

    boot_dir = os.path.join(
        wd,
        "boot"
    )

    dict_dir = os.path.join(
        wd,
        "dict"
    )

    mat5 = os.path.join(
        wd,
        "clusters.mat"
    )


    # ========================================================
    # STEP 1
    # Load and preprocess
    # ========================================================

    cmd1 = [
        "python",
        "step1_load_data.py",

        "--fmri",
        args.fmri_path,

        "--subsample",
        str(args.subsample_factor),

        "--out",
        mat1
    ]

    # Only NIfTI needs a mask
    if input_format == "nifti":

        cmd1 += [
            "--mask",
            args.mask_path
        ]

    run_step(
        cmd1,
        mat1,
        1 in to_run,
        1
    )


    # ========================================================
    # STEP 2
    # MDL scale estimation
    # ========================================================

    Kmin, Kstep, Kmax = (
        args.network_scales
    )

    cmd2 = [
        "python",
        "step2_estimate_scale.py",

        "--tseries",
        mat1,

        "--k-min",
        str(Kmin),

        "--k-step",
        str(Kstep),

        "--k-max",
        str(Kmax),

        "--out",
        mat2
    ]

    if (
        args.l_max is not None
        and args.l_max > 0
    ):

        cmd2 += [
            "--l-max",
            str(args.l_max)
        ]

    if args.step2_extra:

        cmd2 += args.step2_extra

    run_step(
        cmd2,
        mat2,
        2 in to_run,
        2
    )


    # ========================================================
    # STEP 3
    # Bootstrap
    # ========================================================

    last_boot = os.path.join(
        boot_dir,
        f"boot_{args.nb_samps - 1:03d}.mat"
    )

    cmd3 = [
        "python",
        "step3_bootstrap.py",

        "--tseries",
        mat1,

        "--block-len",
        str(
            args.block_window_length[0]
        ),

        "--n-boot",
        str(args.nb_samps),

        "--outdir",
        boot_dir
    ]

    run_step(
        cmd3,
        last_boot,
        3 in to_run,
        3
    )


    # ========================================================
    # STEP 4
    # Dictionary learning
    # ========================================================

    sample_dict = os.path.join(
        dict_dir,
        "boot_000_dict.mat"
    )

    cmd4 = [
        "python",
        "step4_dictionary.py",

        "--bootstrap_dir",
        boot_dir,

        "--scale",
        mat2,

        "--outdir",
        dict_dir,

        "--n-iter",
        str(args.n_iter),

        "--n-jobs",
        str(args.max_parallel_jobs)
    ]

    run_step(
        cmd4,
        sample_dict,
        4 in to_run,
        4
    )


    # ========================================================
    # STEP 5
    # Spatial clustering
    # ========================================================

    dicts = sorted(
        glob.glob(
            os.path.join(
                dict_dir,
                "*_dict.mat"
            )
        )
    )

    if (
        5 in to_run
        and not dicts
    ):

        raise FileNotFoundError(
            f"No dictionary files found in {dict_dir}"
        )

    cmd5 = [
        "python",
        "step5_clustering.py",
        "--dicts"
    ] + dicts + [

        "--scale",
        mat2,

        "--out",
        mat5
    ]

    run_step(
        cmd5,
        mat5,
        5 in to_run,
        5
    )


    # ========================================================
    # STEP 6
    # Atoms + k-hubness
    # ========================================================

    subj = os.path.basename(
        os.path.normpath(wd)
    )

    kmap_dir = os.path.join(
        wd,
        f"KMAP_{subj}"
    )

    os.makedirs(
        kmap_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Expected Step-6 output depends on input format
    # --------------------------------------------------------

    if input_format == "nifti":

        kmap_path = os.path.join(
            kmap_dir,
            f"k_hubness_{subj}.nii.gz"
        )

    elif input_format == "gifti":

        kmap_path = os.path.join(
            kmap_dir,
            f"k_hubness_{subj}.func.gii"
        )


    # --------------------------------------------------------
    # Step-6 command
    # --------------------------------------------------------

    cmd6 = [
        "python",
        "step6_kmap_atoms.py",

        "--clusters",
        mat5,

        "--tseries",
        mat1,

        "--pvalue",
        str(args.pvalue),

        "--min_voxels",
        str(args.min_voxels),

        "--outdir",
        kmap_dir,

        "--subject_label",
        subj
    ]


    # NIfTI reconstruction requires the mask
    if input_format == "nifti":

        cmd6 += [
            "--mask",
            args.mask_path
        ]


    run_step(
        cmd6,
        kmap_path,
        6 in to_run,
        6
    )


    # ========================================================
    # FINISHED
    # ========================================================

    logging.info(
        "✅ SPARK Steps 1–6 complete"
    )

    logging.info(
        "Format: %s",
        input_format
    )

    logging.info(
        "Results: %s",
        wd
    )


if __name__ == "__main__":
    main()