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
import sys

from brick.spatial_io import detect_format


SPARK_DIR = os.path.dirname(os.path.abspath(__file__))
BRICK_DIR = os.path.join(SPARK_DIR, "brick")


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
        "--cortex_scales",
        nargs=3,
        type=int,
        default=[10, 2, 20],
        metavar=("KMIN", "KSTEP", "KMAX"),
        help="Cortical K grid for full-brain CIFTI"
    )

    p.add_argument(
        "--noncortex_scales",
        nargs=3,
        type=int,
        default=[2, 2, 10],
        metavar=("KMIN", "KSTEP", "KMAX"),
        help="Noncortical K grid for full-brain CIFTI"
    )

    p.add_argument(
        "--cifti_region",
        choices=["whole", "cortex", "subcortex"],
        default="whole",
        help="CIFTI analysis region"
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
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Shared random seed for regional bootstrapping"
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

    elif input_format in ("gifti", "cifti"):

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
        sys.executable,
        os.path.join(BRICK_DIR, "step1_load_data.py"),

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
    # MDL scale estimation (regional for full-brain CIFTI)
    # ========================================================

    regional_cifti = False

    if input_format == "cifti" and (
        2 in to_run or any(step in to_run for step in (3, 4, 5, 6))
    ):
        import nibabel as nib
        from nibabel.cifti2.cifti2_axes import BrainModelAxis

        img = nib.load(args.fmri_path)
        brain_axis = img.header.get_axis(1)

        if not isinstance(brain_axis, BrainModelAxis):
            raise ValueError("Expected CIFTI BrainModelAxis.")

        names = brain_axis.name
        cortex_names = {
            "CIFTI_STRUCTURE_CORTEX_LEFT",
            "CIFTI_STRUCTURE_CORTEX_RIGHT",
        }

        has_cortex = any(name in cortex_names for name in names)
        has_noncortex = any(name not in cortex_names for name in names)

        regional_cifti = has_cortex and has_noncortex

    if regional_cifti:

        # Steps 3–6 will be upgraded subsequently.
        # Never run the legacy whole-brain downstream stages
        # accidentally with regional model-selection outputs.
        if any(step in to_run for step in (5, 6)):
            raise NotImplementedError(
                "Regional CIFTI Steps 3–6 are not implemented yet. "
                "Use --steps 1 or --steps 2."
            )

        logging.info(
            "Full-brain CIFTI detected: regional MDL enabled"
        )

        selected_regions = {
            "whole": (
                ("cortex", args.cortex_scales),
                ("noncortex", args.noncortex_scales),
            ),
            "cortex": (
                ("cortex", args.cortex_scales),
            ),
            "subcortex": (
                ("noncortex", args.noncortex_scales),
            ),
        }[args.cifti_region]

        logging.info(
            "CIFTI analysis mode: %s",
            args.cifti_region
        )

        for region, scales in selected_regions:

            kmin, kstep, kmax = scales

            if kmin < 2 or kstep < 1 or kmax < kmin:
                raise ValueError(
                    f"Invalid {region} K grid: {scales}"
                )

            region_dir = os.path.join(wd, region)
            region_mat = os.path.join(region_dir, "tseries.mat")
            region_scale = os.path.join(region_dir, "scale.mat")

            if 2 in to_run and not os.path.isfile(region_mat):
                raise FileNotFoundError(
                    f"Missing {region_mat}. Run Step 1 first."
                )

            logging.info(
                "Regional MDL: %s | K=%d:%d:%d | input=%s",
                region, kmin, kstep, kmax, region_mat
            )

            cmd_region = [
                sys.executable,
                os.path.join(BRICK_DIR, "step2_estimate_scale.py"),
                "--tseries", region_mat,
                "--k-min", str(kmin),
                "--k-step", str(kstep),
                "--k-max", str(kmax),
                "--out", region_scale,
            ]

            if args.l_max is not None and args.l_max > 0:
                cmd_region += ["--l-max", str(args.l_max)]

            if args.step2_extra:
                cmd_region += args.step2_extra

            run_step(
                cmd_region,
                region_scale,
                2 in to_run,
                2
            )

    else:
        # Original SPARK behavior for NIfTI, GIFTI,
        # and single-compartment CIFTI.
        Kmin, Kstep, Kmax = args.network_scales

        cmd2 = [
            sys.executable,
            os.path.join(BRICK_DIR, "step2_estimate_scale.py"),
            "--tseries", mat1,
            "--k-min", str(Kmin),
            "--k-step", str(Kstep),
            "--k-max", str(Kmax),
            "--out", mat2,
        ]

        if args.l_max is not None and args.l_max > 0:
            cmd2 += ["--l-max", str(args.l_max)]

        if args.step2_extra:
            cmd2 += args.step2_extra

        run_step(cmd2, mat2, 2 in to_run, 2)


    # ========================================================
    # ========================================================
    # STEP 3
    # Matched circular block bootstrap for regional CIFTI
    # ========================================================

    if 3 in to_run:

        if input_format == "cifti":

            from brick.utils import get_cifti_region_indices

            regions, _ = get_cifti_region_indices(args.fmri_path)

            selected = {
                "whole": ("cortex", "noncortex"),
                "cortex": ("cortex",),
                "subcortex": ("noncortex",),
            }[args.cifti_region]

            for region in selected:

                if len(regions[region]) == 0:
                    raise ValueError(
                        f"CIFTI contains no {region} grayordinates."
                    )

                other = (
                    "noncortex" if region == "cortex"
                    else "cortex"
                )

                region_dir = os.path.join(wd, region)
                region_mat = os.path.join(
                    region_dir, "tseries.mat"
                )

                # For a single-compartment CIFTI,
                # Step 1 may produce only the full matrix.
                if not os.path.isfile(region_mat):
                    if len(regions[other]) == 0:
                        region_mat = mat1
                    else:
                        raise FileNotFoundError(
                            f"Missing {region_mat}. Run Step 1."
                        )

                region_boot = os.path.join(region_dir, "boot")
                marker = os.path.join(
                    region_boot,
                    f"boot_{args.nb_samps - 1:03d}.mat"
                )

                cmd = [
                    sys.executable,
                    os.path.join(
                        BRICK_DIR, "step3_bootstrap.py"
                    ),
                    "--tseries", region_mat,
                    "--block-len",
                    str(args.block_window_length[0]),
                    "--n-boot", str(args.nb_samps),
                    "--seed", str(args.bootstrap_seed),
                    "--outdir", region_boot,
                ]

                logging.info(
                    "Regional Step 3: %s | seed=%d",
                    region, args.bootstrap_seed
                )

                run_step(cmd, marker, True, 3)

        else:

            last_boot = os.path.join(
                boot_dir,
                f"boot_{args.nb_samps - 1:03d}.mat"
            )

            cmd3 = [
                sys.executable,
                os.path.join(
                    BRICK_DIR, "step3_bootstrap.py"
                ),
                "--tseries", mat1,
                "--block-len",
                str(args.block_window_length[0]),
                "--n-boot", str(args.nb_samps),
                "--outdir", boot_dir,
            ]

            run_step(cmd3, last_boot, True, 3)


    # ========================================================
    # STEP 4
    # K-SVD dictionary learning
    # ========================================================

    if 4 in to_run and input_format == "cifti":

        from brick.utils import get_cifti_region_indices

        regions, _ = get_cifti_region_indices(args.fmri_path)

        selected = {
            "whole": ("cortex", "noncortex"),
            "cortex": ("cortex",),
            "subcortex": ("noncortex",),
        }[args.cifti_region]

        for region in selected:

            if len(regions[region]) == 0:
                raise ValueError(
                    f"CIFTI contains no {region} grayordinates."
                )

            region_dir = os.path.join(wd, region)
            region_boot = os.path.join(region_dir, "boot")
            region_scale = os.path.join(region_dir, "scale.mat")
            region_dict = os.path.join(region_dir, "dict")

            if not os.path.isdir(region_boot):
                raise FileNotFoundError(
                    f"Missing bootstrap directory: {region_boot}"
                )

            if not os.path.isfile(region_scale):
                raise FileNotFoundError(
                    f"Missing regional scale: {region_scale}"
                )

            boots = sorted(glob.glob(
                os.path.join(region_boot, "boot_*.mat")
            ))

            if len(boots) != args.nb_samps:
                raise ValueError(
                    f"{region}: found {len(boots)} bootstrap "
                    f"files, expected {args.nb_samps}"
                )

            expected = [
                f"boot_{i:03d}.mat"
                for i in range(args.nb_samps)
            ]

            if [os.path.basename(x) for x in boots] != expected:
                raise ValueError(
                    f"{region}: bootstrap filenames are incomplete"
                )

            cmd = [
                sys.executable,
                os.path.join(
                    BRICK_DIR, "step4_dictionary.py"
                ),
                "--bootstrap_dir", region_boot,
                "--scale", region_scale,
                "--outdir", region_dict,
                "--n-iter", str(args.n_iter),
                "--n-jobs", str(args.max_parallel_jobs),
            ]

            logging.info(
                "Regional Step 4: %s | %d bootstraps",
                region, len(boots)
            )

            # Step 4 already skips completed dictionaries
            # internally. Always invoke it to resume partial runs.
            subprocess.check_call(cmd)

            missing = [
                f"boot_{i:03d}_dict.mat"
                for i in range(args.nb_samps)
                if not os.path.isfile(os.path.join(
                    region_dict, f"boot_{i:03d}_dict.mat"
                ))
            ]

            if missing:
                raise RuntimeError(
                    f"{region}: missing dictionaries: {missing}"
                )

    else:

        sample_dict = os.path.join(
            dict_dir, "boot_000_dict.mat"
        )

        cmd4 = [
            sys.executable,
            os.path.join(
                BRICK_DIR, "step4_dictionary.py"
            ),
            "--bootstrap_dir", boot_dir,
            "--scale", mat2,
            "--outdir", dict_dir,
            "--n-iter", str(args.n_iter),
            "--n-jobs", str(args.max_parallel_jobs)
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
        sys.executable,
        os.path.join(BRICK_DIR, "step5_clustering.py"),
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

    elif input_format == "cifti":

        kmap_path = os.path.join(
            kmap_dir,
            f"k_hubness_{subj}.dscalar.nii"
        )


    # --------------------------------------------------------
    # Step-6 command
    # --------------------------------------------------------

    cmd6 = [
        sys.executable,
        os.path.join(BRICK_DIR, "step6_kmap_atoms.py"),

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
